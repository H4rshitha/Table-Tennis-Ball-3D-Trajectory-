/**
 * 3D Table Tennis Trajectory & Spin Uplifting - Interactive Demo Engine
 * Connects Three.js 3D Digital Twin, 2D Monocular Frame Player, and Kinematic Analytics.
 */

// Global State
let resultsData = null;
let frameImages = [];
let currentFrameIndex = 0;
let isPlaying = false;
let playbackSpeed = 0.5;
let playInterval = null;

// Three.js State
let scene, camera, renderer, controls;
let ballMesh, trajectoryLine, spinArrowHelper;
let trajectoryPoints3D = [];
let tableGroup;

// Table Dimensions (ITTF Regulation in meters)
const TABLE_LENGTH = 2.74;
const TABLE_WIDTH = 1.525;
const TABLE_HEIGHT = 0.76;
const NET_HEIGHT = 0.1525;
const NET_OVERHANG = 0.1525;

// Canonical 13 Table Keypoint Connections
const TABLE_CONNECTIONS = [
  [0, 2], [2, 4], // Left sideline
  [1, 3], [3, 5], // Right sideline
  [0, 1], [4, 5], // Front & Back baseline
  [6, 2], [2, 3], [3, 7], // Center line & net bottom
  [6, 9], [7, 10], [9, 10], // Net frame & top tape
  [11, 8], [12, 8] // Longitudinal center line
];

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', async () => {
  initThreeJS();
  initEventListeners();
  await loadPipelineData();
  initCharts();
  renderFrame(0);
});

// --------------------------------------------------------------------------
// 1. Data Fetching & State Initialization
// --------------------------------------------------------------------------
async function loadPipelineData() {
  try {
    const res = await fetch('/api/results');
    resultsData = await res.json();
    
    // Update Telemetry Bar & Matrices
    updateTelemetryHUD(resultsData);
    
    // Load 3D Trajectory in Three.js
    if (resultsData.trajectory_3d && resultsData.trajectory_3d.length > 0) {
      trajectoryPoints3D = resultsData.trajectory_3d.map(p => new THREE.Vector3(p[0], p[2], p[1])); // Three.js Y is UP (Z in physics)
      build3DTrajectory(trajectoryPoints3D);
    }
    
    // Preload 2D Images
    const framesRes = await fetch('/api/frames');
    const framesInfo = await framesRes.json();
    
    frameImages = await Promise.all(
      framesInfo.frame_urls.map(url => {
        return new Promise((resolve) => {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => resolve(img);
          img.src = url;
        });
      })
    );
    
    // Setup Scrubber Range
    const scrubber = document.getElementById('timeline-scrubber');
    scrubber.max = frameImages.length - 1;
    scrubber.value = 0;
    
    // Draw initial 2D canvas
    draw2DCanvas(0);
    drawSpinCompass();
    
  } catch (err) {
    console.error('Failed to load results:', err);
  }
}

// --------------------------------------------------------------------------
// 2. Three.js 3D Digital Twin Setup
// --------------------------------------------------------------------------
function initThreeJS() {
  const container = document.getElementById('threejs-canvas-wrapper');
  const width = container.clientWidth || 600;
  const height = container.clientHeight || 450;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e17);
  scene.fog = new THREE.FogExp2(0x0a0e17, 0.08);

  // Camera
  camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
  setCameraPreset('broadcast');

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // OrbitControls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, TABLE_HEIGHT, 0);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
  mainLight.position.set(3, 6, 4);
  mainLight.castShadow = true;
  mainLight.shadow.mapSize.width = 2048;
  mainLight.shadow.mapSize.height = 2048;
  scene.add(mainLight);

  const fillLight = new THREE.DirectionalLight(0x00f2fe, 0.4);
  fillLight.position.set(-4, 3, -3);
  scene.add(fillLight);

  // Build Arena Table & Floor
  buildTableTennisArena();

  // Animated 3D Ball
  const ballGeo = new THREE.SphereGeometry(0.02, 32, 32); // 40mm diameter standard ball (2cm radius)
  const ballMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0.1,
    emissive: 0x222222
  });
  ballMesh = new THREE.Mesh(ballGeo, ballMat);
  ballMesh.castShadow = true;
  ballMesh.position.set(0, TABLE_HEIGHT + 0.1, 0);
  scene.add(ballMesh);

  // Spin Vector Arrow Helper
  const dir = new THREE.Vector3(0, 1, 0);
  spinArrowHelper = new THREE.ArrowHelper(dir, ballMesh.position, 0.18, 0xf59e0b, 0.05, 0.03);
  scene.add(spinArrowHelper);

  // Handle Resize
  window.addEventListener('resize', onWindowResize);

  // Render Loop
  animateThreeJS();
}

function buildTableTennisArena() {
  tableGroup = new THREE.Group();

  // 1. Sport Floor
  const floorGeo = new THREE.PlaneGeometry(12, 10);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x111625,
    roughness: 0.8,
    metalness: 0.2
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0;
  floor.receiveShadow = true;
  scene.add(floor);

  // Floor Grid Lines
  const grid = new THREE.GridHelper(10, 20, 0x00f2fe, 0x1e293b);
  grid.position.y = 0.001;
  scene.add(grid);

  // 2. Table Top Surface (ITTF Blue)
  const tableTopGeo = new THREE.BoxGeometry(TABLE_LENGTH, 0.04, TABLE_WIDTH);
  const tableTopMat = new THREE.MeshStandardMaterial({
    color: 0x0284c7, // ITTF Blue
    roughness: 0.4,
    metalness: 0.1
  });
  const tableTop = new THREE.Mesh(tableTopGeo, tableTopMat);
  tableTop.position.set(0, TABLE_HEIGHT - 0.02, 0);
  tableTop.castShadow = true;
  tableTop.receiveShadow = true;
  tableGroup.add(tableTop);

  // 3. White Boundary Lines
  const lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  
  // Outer perimeter lines (2cm white border)
  const lineThick = 0.02;
  const borderGeo1 = new THREE.PlaneGeometry(TABLE_LENGTH, lineThick);
  const lineSide1 = new THREE.Mesh(borderGeo1, lineMat);
  lineSide1.rotation.x = -Math.PI / 2;
  lineSide1.position.set(0, TABLE_HEIGHT + 0.001, TABLE_WIDTH / 2 - lineThick / 2);
  tableGroup.add(lineSide1);

  const lineSide2 = lineSide1.clone();
  lineSide2.position.set(0, TABLE_HEIGHT + 0.001, -TABLE_WIDTH / 2 + lineThick / 2);
  tableGroup.add(lineSide2);

  const borderGeo2 = new THREE.PlaneGeometry(lineThick, TABLE_WIDTH);
  const lineEnd1 = new THREE.Mesh(borderGeo2, lineMat);
  lineEnd1.rotation.x = -Math.PI / 2;
  lineEnd1.position.set(TABLE_LENGTH / 2 - lineThick / 2, TABLE_HEIGHT + 0.001, 0);
  tableGroup.add(lineEnd1);

  const lineEnd2 = lineEnd1.clone();
  lineEnd2.position.set(-TABLE_LENGTH / 2 + lineThick / 2, TABLE_HEIGHT + 0.001, 0);
  tableGroup.add(lineEnd2);

  // Longitudinal Center Line (for doubles, 3mm)
  const centerLineGeo = new THREE.PlaneGeometry(TABLE_LENGTH, 0.006);
  const centerLine = new THREE.Mesh(centerLineGeo, lineMat);
  centerLine.rotation.x = -Math.PI / 2;
  centerLine.position.set(0, TABLE_HEIGHT + 0.001, 0);
  tableGroup.add(centerLine);

  // 4. Net Assembly (15.25cm high, extending 15.25cm on each side)
  const netWidth = TABLE_WIDTH + 2 * NET_OVERHANG;
  const netGeo = new THREE.PlaneGeometry(netWidth, NET_HEIGHT);
  const netMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    transparent: true,
    opacity: 0.65,
    side: THREE.DoubleSide
  });
  const net = new THREE.Mesh(netGeo, netMat);
  net.rotation.y = Math.PI / 2;
  net.position.set(0, TABLE_HEIGHT + NET_HEIGHT / 2, 0);
  tableGroup.add(net);

  // White Top Tape on Net
  const netTapeGeo = new THREE.BoxGeometry(0.01, 0.015, netWidth);
  const netTape = new THREE.Mesh(netTapeGeo, lineMat);
  netTape.position.set(0, TABLE_HEIGHT + NET_HEIGHT, 0);
  tableGroup.add(netTape);

  // Net Posts
  const postGeo = new THREE.CylinderGeometry(0.015, 0.015, NET_HEIGHT + 0.05, 16);
  const postMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8 });
  const postLeft = new THREE.Mesh(postGeo, postMat);
  postLeft.position.set(0, TABLE_HEIGHT + NET_HEIGHT / 2, TABLE_WIDTH / 2 + NET_OVERHANG);
  tableGroup.add(postLeft);

  const postRight = postLeft.clone();
  postRight.position.set(0, TABLE_HEIGHT + NET_HEIGHT / 2, -TABLE_WIDTH / 2 - NET_OVERHANG);
  tableGroup.add(postRight);

  // 5. Table Legs & Undercarriage
  const legGeo = new THREE.CylinderGeometry(0.03, 0.03, TABLE_HEIGHT - 0.04, 16);
  const legMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.6, roughness: 0.3 });
  
  const legPositions = [
    [-TABLE_LENGTH * 0.35, (TABLE_HEIGHT - 0.04) / 2, TABLE_WIDTH * 0.35],
    [-TABLE_LENGTH * 0.35, (TABLE_HEIGHT - 0.04) / 2, -TABLE_WIDTH * 0.35],
    [TABLE_LENGTH * 0.35, (TABLE_HEIGHT - 0.04) / 2, TABLE_WIDTH * 0.35],
    [TABLE_LENGTH * 0.35, (TABLE_HEIGHT - 0.04) / 2, -TABLE_WIDTH * 0.35]
  ];

  legPositions.forEach(pos => {
    const leg = new THREE.Mesh(legGeo, legMat);
    leg.position.set(...pos);
    leg.castShadow = true;
    tableGroup.add(leg);
  });

  scene.add(tableGroup);
}

function build3DTrajectory(points) {
  if (trajectoryLine) scene.remove(trajectoryLine);

  if (points.length < 2) return;

  // CatmullRom Curve for smooth arc
  const curve = new THREE.CatmullRomCurve3(points);
  const tubeGeo = new THREE.TubeGeometry(curve, 100, 0.008, 8, false);
  
  // Gradient colored tube
  const tubeMat = new THREE.MeshStandardMaterial({
    color: 0xff3366,
    emissive: 0x880022,
    roughness: 0.2,
    metalness: 0.5
  });
  
  trajectoryLine = new THREE.Mesh(tubeGeo, tubeMat);
  scene.add(trajectoryLine);

  // Add waypoint spheres
  points.forEach((pt, idx) => {
    const ptGeo = new THREE.SphereGeometry(0.012, 16, 16);
    const ptMat = new THREE.MeshBasicMaterial({
      color: idx === 0 ? 0x00f2fe : (idx === points.length - 1 ? 0x10b981 : 0xffaa00)
    });
    const marker = new THREE.Mesh(ptGeo, ptMat);
    marker.position.copy(pt);
    scene.add(marker);
  });
}

function animateThreeJS() {
  requestAnimationFrame(animateThreeJS);
  
  // Rotate ball slightly along estimated spin axis for visual wow
  if (ballMesh && resultsData && resultsData.physics) {
    const spinRad = resultsData.physics.spin_vector_rad_s;
    if (spinRad) {
      ballMesh.rotation.x += spinRad[0] * 0.001;
      ballMesh.rotation.y += spinRad[2] * 0.001;
      ballMesh.rotation.z += spinRad[1] * 0.001;
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

function onWindowResize() {
  const container = document.getElementById('threejs-canvas-wrapper');
  if (!container) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function setCameraPreset(preset) {
  if (!camera) return;
  
  switch(preset) {
    case 'broadcast':
      camera.position.set(3.8, 2.4, 2.2);
      break;
    case 'side':
      camera.position.set(0, 1.3, 3.6);
      break;
    case 'top':
      camera.position.set(0, 4.5, 0.001);
      break;
    case 'receiver':
      camera.position.set(-3.2, 1.4, 0);
      break;
    case 'server':
      camera.position.set(3.2, 1.4, 0);
      break;
    case 'orbit':
      camera.position.set(2.5, 2.0, 2.8);
      break;
  }
  if (controls) controls.target.set(0, TABLE_HEIGHT, 0);
}

// --------------------------------------------------------------------------
// 3. Synchronized 2D Canvas & Video Overlay
// --------------------------------------------------------------------------
function draw2DCanvas(frameIdx) {
  const canvas = document.getElementById('canvas-2d');
  const ctx = canvas.getContext('2d');
  if (!frameImages || frameImages.length === 0) return;

  const img = frameImages[frameIdx] || frameImages[0];
  if (!img) return;

  canvas.width = img.naturalWidth || 1920;
  canvas.height = img.naturalHeight || 1080;

  // 1. Draw monocular video frame
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (!resultsData) return;

  const showTableKps = document.getElementById('toggle-table-kps').checked;
  const showBall2D = document.getElementById('toggle-ball-2d').checked;
  const showReproj = document.getElementById('toggle-reproj').checked;

  // 2. Draw 13 Table Keypoints & Wireframe
  if (showTableKps && resultsData.table_keypoints_2d_filtered) {
    const kps = resultsData.table_keypoints_2d_filtered;

    // Draw Skeleton Lines
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    ctx.beginPath();
    TABLE_CONNECTIONS.forEach(([i, j]) => {
      if (kps[i] && kps[j] && kps[i][2] === 1 && kps[j][2] === 1) {
        ctx.moveTo(kps[i][0], kps[i][1]);
        ctx.lineTo(kps[j][0], kps[j][1]);
      }
    });
    ctx.stroke();

    // Draw Keypoint Circles & Labels
    kps.forEach((kp, idx) => {
      if (kp[2] === 1) {
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(kp[0], kp[1], 7, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px Outfit, sans-serif';
        ctx.fillText(`${idx}`, kp[0] + 10, kp[1] + 5);
      }
    });
  }

  // 3. Draw 2D Ball Detection Streak
  if (showBall2D && resultsData.trajectory_2d_detected) {
    const traj2d = resultsData.trajectory_2d_detected;
    
    // Draw trail
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.8)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    traj2d.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt[0], pt[1]);
      else ctx.lineTo(pt[0], pt[1]);
    });
    ctx.stroke();

    // Draw points
    traj2d.forEach((pt, idx) => {
      ctx.fillStyle = '#f43f5e';
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // 4. Draw 3D Reprojected Trajectory (Cyan)
  if (showReproj && resultsData.trajectory_2d_reprojected) {
    const reproj = resultsData.trajectory_2d_reprojected;
    
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 3;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    reproj.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt[0], pt[1]);
      else ctx.lineTo(pt[0], pt[1]);
    });
    ctx.stroke();
    ctx.setLineDash([]); // Reset line dash

    // Current Active Frame Indicator
    const validIdxs = resultsData.valid_frame_indices || [];
    const activeOrder = validIdxs.indexOf(frameIdx - 1);
    if (activeOrder >= 0 && activeOrder < reproj.length) {
      const curPt = reproj[activeOrder];
      
      // Target Reticle
      ctx.strokeStyle = '#00f2fe';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(curPt[0], curPt[1], 14, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(curPt[0], curPt[1], 6, 0, Math.PI * 2);
      ctx.fill();

      // HUD text
      document.getElementById('hud-ball-pos').innerText = `Ball: (X: ${Math.round(curPt[0])}px, Y: ${Math.round(curPt[1])}px)`;
    }
  }
}

// --------------------------------------------------------------------------
// 4. Synchronization & Playback Controls
// --------------------------------------------------------------------------
function renderFrame(frameIdx) {
  currentFrameIndex = frameIdx;
  document.getElementById('frame-counter').innerText = `Frame ${String(frameIdx + 1).padStart(2, '0')} / ${String(frameImages.length).padStart(2, '0')}`;
  document.getElementById('timeline-scrubber').value = frameIdx;
  
  // Render 2D Frame
  draw2DCanvas(frameIdx);

  // Sync 3D Ball Position
  if (resultsData && resultsData.valid_frame_indices && resultsData.trajectory_3d) {
    const validIdxs = resultsData.valid_frame_indices;
    const activeOrder = validIdxs.indexOf(frameIdx - 1);
    
    if (activeOrder >= 0 && activeOrder < trajectoryPoints3D.length) {
      const target3D = trajectoryPoints3D[activeOrder];
      ballMesh.position.copy(target3D);
      ballMesh.visible = true;

      // Update 3D HUD
      const orig3D = resultsData.trajectory_3d[activeOrder];
      document.getElementById('hud-3d-pos').innerText = `P(3D): [X: ${orig3D[0].toFixed(2)}, Y: ${orig3D[1].toFixed(2)}, Z: ${orig3D[2].toFixed(2)}] m`;

      // Update Spin Arrow Position & Direction
      if (spinArrowHelper && resultsData.physics) {
        spinArrowHelper.position.copy(target3D);
        const spinVec = resultsData.physics.spin_vector_rad_s;
        const spinDir = new THREE.Vector3(spinVec[0], spinVec[2], spinVec[1]).normalize();
        spinArrowHelper.setDirection(spinDir);
      }
    }
  }
}

function togglePlayPause() {
  isPlaying = !isPlaying;
  const icon = document.getElementById('icon-play-pause');
  
  if (isPlaying) {
    icon.className = 'fa-solid fa-pause';
    playInterval = setInterval(() => {
      let nextIdx = (currentFrameIndex + 1) % frameImages.length;
      renderFrame(nextIdx);
    }, (1000 / 60) / playbackSpeed);
  } else {
    icon.className = 'fa-solid fa-play';
    clearInterval(playInterval);
  }
}

function initEventListeners() {
  document.getElementById('btn-play-pause').addEventListener('click', togglePlayPause);
  
  document.getElementById('btn-prev-frame').addEventListener('click', () => {
    if (isPlaying) togglePlayPause();
    let prev = Math.max(0, currentFrameIndex - 1);
    renderFrame(prev);
  });

  document.getElementById('btn-next-frame').addEventListener('click', () => {
    if (isPlaying) togglePlayPause();
    let next = Math.min(frameImages.length - 1, currentFrameIndex + 1);
    renderFrame(next);
  });

  const scrubber = document.getElementById('timeline-scrubber');
  scrubber.addEventListener('input', (e) => {
    renderFrame(parseInt(e.target.value));
  });

  // Layer toggles
  ['toggle-table-kps', 'toggle-ball-2d', 'toggle-reproj'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => draw2DCanvas(currentFrameIndex));
  });

  // Viewpoint buttons
  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setCameraPreset(btn.dataset.view);
    });
  });

  // Playback speed
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playbackSpeed = parseFloat(btn.dataset.speed);
      if (isPlaying) {
        togglePlayPause();
        togglePlayPause();
      }
    });
  });

  // Keyboard shortcuts (Space = Play/Pause, Left/Right = Frame Step)
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === 'ArrowLeft') {
      let prev = Math.max(0, currentFrameIndex - 1);
      renderFrame(prev);
    } else if (e.code === 'ArrowRight') {
      let next = Math.min(frameImages.length - 1, currentFrameIndex + 1);
      renderFrame(next);
    }
  });
}

// --------------------------------------------------------------------------
// 5. Telemetry & Analytics HUD Updates
// --------------------------------------------------------------------------
function updateTelemetryHUD(data) {
  const p = data.physics;
  if (!p) return;

  document.getElementById('stat-spin-class').innerText = p.spin_type || 'Topspin';
  document.getElementById('stat-topspin-hz').innerText = `${p.spin_frequencies_hz.top_back_y.toFixed(2)} Hz`;
  document.getElementById('stat-sidespin-hz').innerText = `${p.spin_frequencies_hz.sidespin_z.toFixed(2)} Hz`;
  document.getElementById('stat-peak-speed').innerText = `${p.max_speed_kmh} km/h`;
  document.getElementById('stat-reproj-error').innerText = `${p.mean_reprojection_error_pixels} px`;

  document.getElementById('text-apex-height').innerText = `${p.peak_height_meters} m`;
  document.getElementById('text-avg-speed').innerText = `${p.avg_speed_kmh} km/h`;
  document.getElementById('text-spin-badge').innerText = p.spin_type;

  // Matrices
  if (data.camera_calibration) {
    const mint = data.camera_calibration.intrinsic_matrix;
    const mext = data.camera_calibration.extrinsic_matrix;

    document.getElementById('matrix-int-pre').innerText = 
      `fx=${mint[0][0].toFixed(1)}  fy=${mint[1][1].toFixed(1)}\ncx=${mint[0][2].toFixed(1)}  cy=${mint[1][2].toFixed(1)}`;

    document.getElementById('matrix-ext-pre').innerText = 
      `R=[${mext[0][0].toFixed(2)}, ${mext[0][1].toFixed(2)}, ${mext[0][2].toFixed(2)}]\nt=[${mext[0][3].toFixed(2)}, ${mext[1][3].toFixed(2)}, ${mext[2][3].toFixed(2)}]`;
  }
}

// --------------------------------------------------------------------------
// 6. Chart.js Graphs
// --------------------------------------------------------------------------
function initCharts() {
  if (!resultsData || !resultsData.trajectory_3d) return;

  const traj3d = resultsData.trajectory_3d;
  const times = traj3d.map((_, i) => (i / 60).toFixed(2));
  const heights = traj3d.map(p => p[2]);

  // Chart 1: Height Arc
  const ctxHeight = document.getElementById('chart-height').getContext('2d');
  new Chart(ctxHeight, {
    type: 'line',
    data: {
      labels: times,
      datasets: [
        {
          label: 'Ball Height [m]',
          data: heights,
          borderColor: '#f43f5e',
          backgroundColor: 'rgba(244, 63, 94, 0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 3
        },
        {
          label: 'Table Surface (0.76m)',
          data: Array(times.length).fill(TABLE_HEIGHT),
          borderColor: '#0284c7',
          borderDash: [5, 5],
          pointRadius: 0
        },
        {
          label: 'Net Height (0.9125m)',
          data: Array(times.length).fill(TABLE_HEIGHT + NET_HEIGHT),
          borderColor: '#ffffff',
          borderDash: [2, 2],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 10 } } } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' }, title: { display: true, text: 'Time (seconds)', color: '#64748b', font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' }, min: 0.6, max: 1.2, title: { display: true, text: 'Height (meters)', color: '#64748b', font: { size: 10 } } }
      }
    }
  });

  // Chart 2: Velocity Profile
  const speeds = [];
  for (let i = 1; i < traj3d.length; i++) {
    const dx = traj3d[i][0] - traj3d[i-1][0];
    const dy = traj3d[i][1] - traj3d[i-1][1];
    const dz = traj3d[i][2] - traj3d[i-1][2];
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    speeds.push((dist * 60 * 3.6).toFixed(1)); // km/h
  }

  const ctxSpeed = document.getElementById('chart-speed').getContext('2d');
  new Chart(ctxSpeed, {
    type: 'line',
    data: {
      labels: times.slice(1),
      datasets: [{
        label: 'Speed [km/h]',
        data: speeds,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 10 } } } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' }, title: { display: true, text: 'Time (seconds)', color: '#64748b', font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' }, title: { display: true, text: 'Velocity (km/h)', color: '#64748b', font: { size: 10 } } }
      }
    }
  });

  // Chart 3: Spin Decomposition
  const p = resultsData.physics.spin_frequencies_hz;
  const ctxSpin = document.getElementById('chart-spin').getContext('2d');
  new Chart(ctxSpin, {
    type: 'bar',
    data: {
      labels: ['Corkscrew Spin (X)', 'Top/Backspin (Y)', 'Sidespin (Z)'],
      datasets: [{
        label: 'Rate [Hz]',
        data: [p.corkscrew_x, p.top_back_y, p.sidespin_z],
        backgroundColor: ['#4f46e5', '#f43f5e', '#00f2fe'],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 9 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' }, title: { display: true, text: 'Frequency (Hz / rev/s)', color: '#64748b', font: { size: 10 } } }
      }
    }
  });
}

function drawSpinCompass() {
  const canvas = document.getElementById('canvas-spin-vector');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;

  ctx.clearRect(0, 0, w, h);

  // Compass circle
  ctx.strokeStyle = 'rgba(0, 242, 254, 0.4)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, 34, 0, Math.PI * 2);
  ctx.stroke();

  // Draw axis
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.beginPath();
  ctx.moveTo(cx, 10); ctx.lineTo(cx, h - 10);
  ctx.moveTo(10, cy); ctx.lineTo(w - 10, cy);
  ctx.stroke();

  if (resultsData && resultsData.physics) {
    const p = resultsData.physics.spin_frequencies_hz;
    const topBack = p.top_back_y;
    const side = p.sidespin_z;

    const angle = Math.atan2(-topBack, side);
    const len = Math.min(30, Math.sqrt(topBack*topBack + side*side) * 8);

    const tox = cx + Math.cos(angle) * len;
    const toy = cy + Math.sin(angle) * len;

    // Arrow
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(tox, toy);
    ctx.stroke();

    ctx.fillStyle = '#00f2fe';
    ctx.beginPath();
    ctx.arc(tox, toy, 4, 0, Math.PI * 2);
    ctx.fill();

    document.getElementById('compass-hz-val').innerText = `${Math.abs(topBack).toFixed(2)} Hz ${topBack >= 0 ? 'Topspin' : 'Backspin'}`;
  }
}
