import os
import sys
import json
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import paths
from interface import TableTennisPipeline, BallDetector, TableDetector
from uplifting.helper import table_points, table_connections, TABLE_HEIGHT, TABLE_WIDTH, TABLE_LENGTH
from inference.utils import KEYPOINT_VISIBLE, world2cam, cam2img

def run_pipeline_analysis(image_folder='example_images', output_dir='results', fps=60.0):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load image sequence
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"[1/5] Loading {len(image_files)} frames from '{image_folder}'...", flush=True)
    images = [cv2.imread(os.path.join(image_folder, f)) for f in image_files]
    orig_h, orig_w = images[0].shape[:2]
    
    # 2. Run Full Pipeline
    print("[2/5] Initializing models (SegFormer++ B2, WASB, HRNet, ConnectStage Uplifting Transformer)...", flush=True)
    pipeline = TableTennisPipeline()
    
    print("[3/5] Running end-to-end inference (2D Ball + 13 Table Keypoints + 3D Uplifting + Spin)...", flush=True)
    results = pipeline.predict_full(images, fps, calibrate=True)
    
    pred_spin = results['pred_spin']
    pred_pos_3d = results['pred_pos_3d']
    filtered_ball_pos = results['filtered_ball_positions']
    valid_ball_indices = results['valid_indices_ball']
    ball_times = results['times_ball']
    filtered_table_kps = results['filtered_table_keypoints']
    Mint = results['Mint']
    Mext = results['Mext']
    reprojected_2d = results['reprojected_2d']
    
    # 3. Analyze Physics & Kinematics
    spin_np = pred_spin.cpu().numpy() if torch.is_tensor(pred_spin) else np.array(pred_spin)
    spin_rad_s = spin_np.tolist()
    spin_hz = (spin_np / (2 * np.pi)).tolist()
    
    # Spin Classification
    top_back_hz = spin_hz[1]
    side_hz = spin_hz[2]
    cork_hz = spin_hz[0]
    
    if top_back_hz > 2.0:
        spin_type = "Heavy Topspin"
    elif top_back_hz > 0.5:
        spin_type = "Topspin"
    elif top_back_hz < -2.0:
        spin_type = "Heavy Backspin (Underspin)"
    elif top_back_hz < -0.5:
        spin_type = "Backspin (Chop/Push)"
    else:
        spin_type = "Float / Minimal Spin"
        
    if abs(side_hz) > 1.5:
        spin_type += f" with {'Right' if side_hz > 0 else 'Left'} Sidespin"
        
    # Velocity calculation
    dt = 1.0 / fps
    velocities_3d = np.linalg.norm(np.diff(pred_pos_3d, axis=0), axis=1) / dt # m/s
    velocities_kmh = velocities_3d * 3.6
    avg_velocity_kmh = float(np.mean(velocities_kmh)) if len(velocities_kmh) > 0 else 0.0
    max_velocity_kmh = float(np.max(velocities_kmh)) if len(velocities_kmh) > 0 else 0.0
    
    # Heights
    peak_height_m = float(np.max(pred_pos_3d[:, 2]))
    min_height_m = float(np.min(pred_pos_3d[:, 2]))
    
    # Reprojection Error (between detected 2D and reprojected 3D)
    min_len = min(len(filtered_ball_pos), len(reprojected_2d))
    reproj_errors = np.linalg.norm(filtered_ball_pos[:min_len] - reprojected_2d[:min_len], axis=1)
    mean_reproj_error_px = float(np.mean(reproj_errors))
    
    print(f"\n====================== RESULTS SUMMARY ======================")
    print(f"Predicted 3D trajectory points: {len(pred_pos_3d)}")
    print(f"Spin Classification:            {spin_type}")
    print(f"Spin Vector (rad/s):           [wx={spin_rad_s[0]:.2f}, wy={spin_rad_s[1]:.2f}, wz={spin_rad_s[2]:.2f}]")
    print(f"Spin Frequencies (Hz):         [Top/Back={spin_hz[1]:.2f} Hz, Side={spin_hz[2]:.2f} Hz, Cork={spin_hz[0]:.2f} Hz]")
    print(f"Average Ball Speed:            {avg_velocity_kmh:.1f} km/h ({avg_velocity_kmh/3.6:.1f} m/s)")
    print(f"Maximum Ball Speed:            {max_velocity_kmh:.1f} km/h ({max_velocity_kmh/3.6:.1f} m/s)")
    print(f"Peak Trajectory Height:        {peak_height_m:.3f} m (Table Height: {TABLE_HEIGHT} m)")
    print(f"Mean 2D Reprojection Error:    {mean_reproj_error_px:.2f} pixels")
    print(f"=============================================================\n")
    
    # 4. Save JSON results for Web UI & Professor Presentation
    results_data = {
        "metadata": {
            "paper": "Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation (WACV 2026)",
            "arxiv": "2511.20250",
            "fps": fps,
            "total_frames": len(images),
            "image_resolution": [orig_w, orig_h]
        },
        "physics": {
            "spin_type": spin_type,
            "spin_vector_rad_s": spin_rad_s,
            "spin_frequencies_hz": {
                "corkscrew_x": spin_hz[0],
                "top_back_y": spin_hz[1],
                "sidespin_z": spin_hz[2]
            },
            "avg_speed_kmh": round(avg_velocity_kmh, 2),
            "max_speed_kmh": round(max_velocity_kmh, 2),
            "peak_height_meters": round(peak_height_m, 4),
            "table_height_meters": TABLE_HEIGHT,
            "mean_reprojection_error_pixels": round(mean_reproj_error_px, 2)
        },
        "camera_calibration": {
            "intrinsic_matrix": Mint.tolist(),
            "extrinsic_matrix": Mext.tolist()
        },
        "trajectory_3d": pred_pos_3d.tolist(),
        "trajectory_2d_detected": filtered_ball_pos.tolist(),
        "trajectory_2d_reprojected": reprojected_2d.tolist(),
        "valid_frame_indices": valid_ball_indices.tolist() if isinstance(valid_ball_indices, np.ndarray) else valid_ball_indices,
        "table_keypoints_3d": table_points.tolist(),
        "table_keypoints_2d_filtered": filtered_table_kps.tolist()
    }
    
    json_path = os.path.join(output_dir, 'pipeline_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"[4/5] Saved structured results to '{json_path}'")
    
    # 5. Generate Visual Charts & Overlay Video
    print("[5/5] Generating presentation figures and overlay visualization...")
    generate_presentation_plots(images, filtered_table_kps, filtered_ball_pos, pred_pos_3d, reprojected_2d, Mint, Mext, spin_hz, spin_type, velocities_kmh, output_dir)
    generate_annotated_video(images, filtered_table_kps, filtered_ball_pos, reprojected_2d, valid_ball_indices, output_dir, fps)
    
    return results_data

def generate_presentation_plots(images, table_kps, ball_2d, pos_3d, reproj_2d, Mint, Mext, spin_hz, spin_type, velocities_kmh, output_dir):
    # Figure 1: 3D Trajectory & Table Mesh in World Metric Space
    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Table surface
    table_x = [-TABLE_LENGTH/2, TABLE_LENGTH/2, TABLE_LENGTH/2, -TABLE_LENGTH/2, -TABLE_LENGTH/2]
    table_y = [-TABLE_WIDTH/2, -TABLE_WIDTH/2, TABLE_WIDTH/2, TABLE_WIDTH/2, -TABLE_WIDTH/2]
    table_z = [TABLE_HEIGHT] * 5
    ax.plot(table_x, table_y, table_z, 'b-', linewidth=2.5, label='Table Boundary')
    
    # Center line & Net
    ax.plot([-TABLE_LENGTH/2, TABLE_LENGTH/2], [0, 0], [TABLE_HEIGHT, TABLE_HEIGHT], 'w--', linewidth=1.2, label='Center Line')
    ax.plot([0, 0], [-TABLE_WIDTH/2 - 0.1525, TABLE_WIDTH/2 + 0.1525], [TABLE_HEIGHT, TABLE_HEIGHT], 'k-', linewidth=2)
    ax.plot([0, 0], [-TABLE_WIDTH/2 - 0.1525, TABLE_WIDTH/2 + 0.1525], [TABLE_HEIGHT + 0.1525, TABLE_HEIGHT + 0.1525], 'k-', linewidth=2, label='Net (15.25cm)')
    
    # 3D Ball Trajectory
    x_3d, y_3d, z_3d = pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2]
    ax.plot(x_3d, y_3d, z_3d, color='#ff3366', linewidth=3.5, label='Predicted 3D Trajectory')
    scatter = ax.scatter(x_3d, y_3d, z_3d, c=np.arange(len(x_3d)), cmap='autumn', s=45, edgecolors='black', linewidth=0.5)
    
    # Trajectory ground shadow
    ax.plot(x_3d, y_3d, np.full_like(z_3d, TABLE_HEIGHT), color='gray', linestyle=':', alpha=0.6, label='Trajectory Projection on Table')
    
    ax.set_title(f"3D Table Tennis Ball Trajectory & Spin Estimation (WACV 2026)\nEstimated Spin: {spin_type} (Topspin={spin_hz[1]:.2f} Hz, Sidespin={spin_hz[2]:.2f} Hz)", fontsize=12, fontweight='bold')
    ax.set_xlabel('X [meters] (Length)', fontweight='bold')
    ax.set_ylabel('Y [meters] (Width)', fontweight='bold')
    ax.set_zlabel('Z [meters] (Height)', fontweight='bold')
    ax.set_xlim([-1.6, 1.6])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([0.7, 1.5])
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_trajectory_world.png'), dpi=200)
    plt.close()
    
    # Figure 2: Multi-panel Presentation Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    
    # Panel A: 2D Detections & Table Keypoints on Mid-flight Frame
    mid_idx = min(len(images) // 2, len(images) - 1)
    img_vis = cv2.cvtColor(images[mid_idx], cv2.COLOR_BGR2RGB)
    
    # Draw table keypoints
    for idx, kp in enumerate(table_kps):
        x, y, v = kp
        if v == KEYPOINT_VISIBLE:
            cv2.circle(img_vis, (int(x), int(y)), 7, (0, 255, 128), -1)
            cv2.putText(img_vis, str(idx), (int(x)+8, int(y)+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    # Draw 2D trajectory streak
    for pos in ball_2d:
        cv2.circle(img_vis, (int(pos[0]), int(pos[1])), 6, (255, 50, 50), -1)
    for pos in reproj_2d:
        cv2.circle(img_vis, (int(pos[0]), int(pos[1])), 4, (0, 220, 255), -1)
        
    axes[0, 0].imshow(img_vis)
    axes[0, 0].set_title(f"A: 2D Video Frame Overlay (Frame #{mid_idx})\nGreen: 13 Table Keypoints | Red: 2D Detected | Cyan: Reprojected 3D", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Panel B: Height Profile vs Time
    time_axis = np.linspace(0, len(pos_3d) / 60.0, len(pos_3d))
    axes[0, 1].plot(time_axis, z_3d, 'r-o', linewidth=2.5, markersize=5, label='Ball Height Z(t)')
    axes[0, 1].axhline(TABLE_HEIGHT, color='blue', linestyle='--', linewidth=1.5, label='Table Surface (0.76m)')
    axes[0, 1].axhline(TABLE_HEIGHT + 0.1525, color='black', linestyle=':', linewidth=1.5, label='Net Height (0.9125m)')
    axes[0, 1].set_title("B: Ball Height vs Time (Trajectory Arc)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Time [seconds]")
    axes[0, 1].set_ylabel("Height [meters]")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    
    # Panel C: Ball Speed Profile
    if len(velocities_kmh) > 0:
        v_time = time_axis[:-1]
        axes[1, 0].plot(v_time, velocities_kmh, 'g-s', linewidth=2, markersize=4, label='Speed [km/h]')
        axes[1, 0].set_title("C: Ball Velocity Profile over Flight", fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel("Time [seconds]")
        axes[1, 0].set_ylabel("Speed [km/h]")
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)
        axes[1, 0].legend()
        
    # Panel D: Spin Frequency & Physics Breakdown
    spin_names = ['Corkscrew (ωx)', 'Top/Backspin (ωy)', 'Sidespin (ωz)']
    colors = ['#4f46e5', '#ef4444', '#06b6d4']
    bars = axes[1, 1].bar(spin_names, spin_hz, color=colors, edgecolor='black', width=0.5)
    axes[1, 1].axhline(0, color='gray', linewidth=1)
    axes[1, 1].set_title(f"D: Spin Vector Breakdown: {spin_type}", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Rotational Frequency [Hz / rev/s]")
    axes[1, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
    for bar in bars:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + (0.1 if yval >= 0 else -0.3), f"{yval:+.2f} Hz", ha='center', va='bottom' if yval >= 0 else 'top', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'presentation_summary_dashboard.png'), dpi=200)
    plt.close()
    print(f"Generated presentation plots in '{output_dir}'")

def generate_annotated_video(images, table_kps, ball_2d, reproj_2d, valid_indices, output_dir, fps):
    video_path = os.path.join(output_dir, 'annotated_trajectory.mp4')
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    
    reproj_dict = {}
    if isinstance(valid_indices, (list, np.ndarray)):
        for idx_order, f_idx in enumerate(valid_indices):
            if idx_order < len(reproj_2d):
                reproj_dict[int(f_idx) + 1] = reproj_2d[idx_order] # offset by 1 because ball detection uses triples
                
    history_detected = []
    history_reprojected = []
    
    for f_idx, img in enumerate(images):
        frame = img.copy()
        
        # Draw table connections & keypoints
        for conn in table_connections:
            pt1, pt2 = table_kps[conn[0]], table_kps[conn[1]]
            if pt1[2] == KEYPOINT_VISIBLE and pt2[2] == KEYPOINT_VISIBLE:
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 120), 2, cv2.LINE_AA)
                
        for kp in table_kps:
            if kp[2] == KEYPOINT_VISIBLE:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 255), -1)
                
        # Draw 3D reprojected point for current frame
        if f_idx in reproj_dict:
            cur_reproj = reproj_dict[f_idx]
            history_reprojected.append(cur_reproj)
            
        # Draw trajectory tails
        for pt in history_reprojected[-15:]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 220, 255), -1)
        if len(history_reprojected) > 1:
            for k in range(1, len(history_reprojected)):
                p1 = (int(history_reprojected[k-1][0]), int(history_reprojected[k-1][1]))
                p2 = (int(history_reprojected[k][0]), int(history_reprojected[k][1]))
                cv2.line(frame, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)
                
        # HUD overlay
        cv2.rectangle(frame, (20, 20), (520, 110), (20, 20, 20), -1)
        cv2.putText(frame, "3D Ball Trajectory & Spin Uplifting (WACV 2026)", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {f_idx+1:02d}/{len(images)} | FPS: {fps:.0f}", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "Green: Table Net/Bounds | Cyan: 3D Trajectory", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        
        writer.write(frame)
        
    writer.release()
    print(f"Exported annotated video to '{video_path}'")

if __name__ == '__main__':
    run_pipeline_analysis()
