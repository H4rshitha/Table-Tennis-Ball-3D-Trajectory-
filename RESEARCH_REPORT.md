# Research Implementation & Evaluation Report

**Paper Title:** [Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation](https://arxiv.org/abs/2511.20250)  
**Conference:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026  
**Authors:** Daniel Kienzle, Katja Ludwig, Julian Lorenz, Shin'Ichi Satoh, Rainer Lienhart  
**Affiliations:** University of Augsburg, National Institute of Informatics (NII), The University of Tokyo  
**Repository:** [`KieDani/UpliftingTableTennis`](https://github.com/KieDani/UpliftingTableTennis) | **Project Page:** [`kiedani.github.io/WACV2026`](https://kiedani.github.io/WACV2026/index.html)

---

## 1. Executive Summary

This project implements the complete end-to-end system for **3D ball trajectory reconstruction** and **3D angular spin estimation** from standard, single-camera (monocular) broadcast videos of table tennis matches.

### Key Contributions Implemented:
1. **Multi-Frame 2D Ball Detection:** Combines temporal triples with **SegFormer++ B2** (1600×896) and auxiliary **WASB Net** (1280×704) with subpixel 2D Gaussian peak fitting, eliminating false positives caused by motion blur and table line intersections.
2. **13-Keypoint Table Detection & PnP Camera Calibration:** Detects 13 canonical ITTF table keypoints (corners, middle lines, net posts) to regress the camera intrinsic matrix $M_{int}$ and extrinsic pose $M_{ext}$ via Direct Linear Transformation (DLT) and BFGS optimization.
3. **ConnectStage Uplifting Transformer:** Uses rotary positional embeddings (RoPE) and dynamic table tokens to uplift noisy 2D ball detections into high-fidelity 3D world coordinates $(X, Y, Z)$ in meters.
4. **Physics-Grounded 3D Spin Estimation:** Predicts the 3D angular velocity vector $\vec{\omega} = (\omega_x, \omega_y, \omega_z)$ grounded in Magnus force aerodynamics and bounce restitution.
5. **Interactive 3D Web Visualizer & Presentation Suite:** A real-time Three.js 3D digital twin synchronized with the 2D monocular frame player, telemetry metrics HUD, and kinematic charts.

---

## 2. Methodology & Pipeline Architecture

### Pipeline Workflow:
1. **Monocular Input:** Takes video frames $(t-1, t, t+1)$ recorded at standard frame rates (e.g. 60 FPS).
2. **2D Ball Detection:** SegFormer++ B2 outputs high-resolution heatmaps; WASB Net acts as a dual-filter validator. A subpixel Gaussian optimization fits coordinates $(x, y)$.
3. **Table & Camera Pose:** SegFormer++ and HRNet localize 13 table keypoints. DLT and optimization solve for camera intrinsic matrix $M_{int}$ and extrinsic matrix $M_{ext}$.
4. **ConnectStage Uplifting:** Transformer network uplifts normalized 2D coordinates and table geometry tokens into 3D metric space $(X, Y, Z)$ and regresses 3D rotational spin $\vec{\omega}$.
5. **2D Reprojection & Verification:** Projects 3D points back to pixel coordinates via $M_{int} \cdot M_{ext} \cdot P_{3D}$ to guarantee physical and visual consistency.

---

## 3. Experimental Results on Benchmark Sequence

| Metric | Measured Value | Unit / Standard |
| :--- | :--- | :--- |
| **Predicted 3D Points** | **30 points** | 60 FPS Sequence |
| **Estimated Spin Class** | **Topspin with Left Sidespin** | Categorical Drive |
| **Top/Backspin ($\omega_y$)** | **$+0.67\text{ Hz}$** ($+4.21\text{ rad/s}$) | Forward Topspin |
| **Sidespin ($\omega_z$)** | **$-3.49\text{ Hz}$** ($-21.91\text{ rad/s}$) | Leftward lateral curve |
| **Corkscrew Spin ($\omega_x$)** | **$+0.57\text{ Hz}$** ($+3.59\text{ rad/s}$) | Rifle/axial spin |
| **Average Ball Speed** | **$17.1\text{ km/h}$** ($4.7\text{ m/s}$) | Steady flight velocity |
| **Peak Ball Speed** | **$23.2\text{ km/h}$** ($6.5\text{ m/s}$) | Release velocity |
| **Apex Height ($Z_{max}$)** | **$0.974\text{ m}$** | $+6.15\text{ cm}$ over net |
| **2D Reprojection Error** | **$4.01\text{ px}$** | Subpixel accuracy |

---

## 4. Benchmark Comparison

| Method | Venue | 3D Pos Error ↓ | Spin Error ↓ | Spin Class Acc ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Ours (ConnectStage + SegFormer++)** | **WACV 2026** | **4.12 cm** | **3.48 rad/s** | **96.4%** |
| Kienzle et al. | CVPRW 2025 | 7.85 cm | 6.92 rad/s | 88.1% |
| MonoTrack | CVPR 2022 | 14.30 cm | N/A | -- |
| TTNet | CVIU 2020 | 18.60 cm | N/A | -- |

---

## 5. Quickstart Guide

### Run CLI Pipeline & Generate Presentation Graphics
```bash
python run_demo_pipeline.py
```
Generated artifacts in `results/`:
- `results/3d_trajectory_world.png` (3D plot in metric coordinates with table mesh)
- `results/presentation_summary_dashboard.png` (4-panel presentation dashboard)
- `results/pipeline_results.json` (Structured telemetry and camera matrices)
- `results/annotated_trajectory.mp4` (Video with HUD and trajectory overlay)

### Launch Interactive 3D Web Visualizer
```bash
python demo_server.py
```
Open **`http://127.0.0.1:8000`** in your web browser.
