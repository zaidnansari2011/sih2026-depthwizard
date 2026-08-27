/**
 * DepthWizard viewer.
 *
 * Loads a scene bundle written by tools/export_terrain.py and renders it as a navigable
 * surface. Three things here are deliberate rather than incidental:
 *
 *  1. Heights are real metres. height.bin is raw float32, so nothing is quantised on the
 *     way into the browser. Every number the measurement tool prints is a number the
 *     model actually predicted, in the units ISRO scores.
 *
 *  2. Uncertainty is a first-class view, not a debug overlay. Differentiator 6.1 claims
 *     the model knows where it is unreliable; the only way to make a jury believe that is
 *     to let them look at it and click on it.
 *
 *  3. The mesh is decimated but the texture is not. Vertex count is what costs frames;
 *     texture resolution is nearly free. So we cap the grid and keep the imagery sharp.
 */
import * as THREE from './vendor/three.module.js';

const $ = (id) => document.getElementById(id);
const MAX_VERTS = 1_000_000;      // ~1M verts keeps a laptop GPU comfortably above 60 fps

const state = {
  manifest: null,
  height: null,      // Float32Array, metres
  sigma: null,       // Float32Array, metres
  grid: { w: 0, h: 0, stepX: 1, stepY: 1 },
  vex: 1.5,
  mode: 'texture',
  measuring: false,
  picks: [],
  touring: false,
  tourT: 0,
};

// ---------------------------------------------------------------- renderer & scene

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
scene.fog = new THREE.Fog(0x0b0f14, 1200, 6000);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.5, 20000);

scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x20262e, 1.05));
const sun = new THREE.DirectionalLight(0xfff3e0, 1.6);
sun.position.set(-1, 1.6, 0.9);
scene.add(sun);

let mesh = null;
let markers = new THREE.Group();
scene.add(markers);

// ---------------------------------------------------------------- colour ramps

// Perceptually ordered ramps. Height uses a terrain-ish ramp; uncertainty deliberately
// runs cool -> hot so "hot" reads as "do not trust this" without needing a caption.
const RAMP_HEIGHT = [[0.15,0.20,0.35],[0.12,0.42,0.45],[0.35,0.63,0.36],[0.85,0.80,0.42],[0.92,0.55,0.30],[0.98,0.95,0.92]];
const RAMP_SIGMA  = [[0.10,0.25,0.45],[0.15,0.55,0.60],[0.65,0.80,0.35],[0.95,0.70,0.20],[0.90,0.25,0.20]];

function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function rampCSS(stops) {
  return `linear-gradient(90deg, ${stops.map((c, i) =>
    `rgb(${c.map(v => Math.round(v * 255)).join(',')}) ${(100 * i / (stops.length - 1)).toFixed(0)}%`).join(',')})`;
}

// ---------------------------------------------------------------- scene loading

async function loadScenes() {
  let list = [];
  try {
    list = await (await fetch('./scenes/index.json', { cache: 'no-store' })).json();
  } catch { /* index is optional; fall through to the empty-state message */ }
  const sel = $('scene');
  sel.innerHTML = '';
  if (!list.length) {
    $('loading').innerHTML =
      `<div class="err"><b>No scenes yet.</b><br><br>Generate one:<br>
       <code>python tools/export_terrain.py --height &lt;pred.tif&gt; --texture &lt;rgb.tif&gt; --out viewer/scenes/demo</code>
       <br><br>then serve this folder:<br><code>python -m http.server -d viewer 8080</code></div>`;
    return false;
  }
  for (const s of list) {
    const o = document.createElement('option');
    o.value = s.dir;
    o.textContent = `${s.name} (${s.width}×${s.height})`;
    sel.appendChild(o);
  }
  sel.onchange = () => loadScene(sel.value);
  await loadScene(list[0].dir);
  return true;
}

async function loadScene(dir) {
  $('loading').style.display = 'grid';
  $('loading').innerHTML = `<div>Loading ${dir}…</div>`;

  const base = `./scenes/${dir}`;
  const m = await (await fetch(`${base}/manifest.json`, { cache: 'no-store' })).json();
  const hbuf = await (await fetch(`${base}/${m.files.height}`)).arrayBuffer();
  state.manifest = m;
  state.height = new Float32Array(hbuf);

  state.sigma = null;
  if (m.files.sigma) {
    try {
      state.sigma = new Float32Array(await (await fetch(`${base}/${m.files.sigma}`)).arrayBuffer());
    } catch { state.sigma = null; }
  }

  let texture = null;
  if (m.files.texture) {
    texture = await new THREE.TextureLoader().loadAsync(`${base}/${m.files.texture}`);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  }

  buildMesh(texture);
  updateStats();
  resetView();
  $('loading').style.display = 'none';
}

// ---------------------------------------------------------------- mesh

function buildMesh(texture) {
  if (mesh) {
    mesh.geometry.dispose();
    if (mesh.material.map) mesh.material.map.dispose();
    mesh.material.dispose();
    scene.remove(mesh);
  }

  const m = state.manifest;
  const W = m.width, H = m.height;

  // Decimate to stay under the vertex budget. Sampling by stride rather than averaging
  // keeps ridge lines crisp; averaging would round off exactly the rooftops we care
  // about most.
  const step = Math.max(1, Math.ceil(Math.sqrt((W * H) / MAX_VERTS)));
  const gw = Math.floor((W - 1) / step) + 1;
  const gh = Math.floor((H - 1) / step) + 1;
  state.grid = { w: gw, h: gh, stepX: step, stepY: step };

  // Ground sample distance. When the export had no metric transform we fall back to one
  // unit per pixel and say so in the HUD, rather than printing confident nonsense.
  const gsd = m.gsd_m || 1;
  state.gsd = gsd;
  state.hasMetres = !!m.gsd_m;

  const pos = new Float32Array(gw * gh * 3);
  const uv = new Float32Array(gw * gh * 2);
  const col = new Float32Array(gw * gh * 3);

  const cx = ((W - 1) * gsd) / 2, cz = ((H - 1) * gsd) / 2;
  let k = 0;
  for (let j = 0; j < gh; j++) {
    const y = Math.min(H - 1, j * step);
    for (let i = 0; i < gw; i++, k++) {
      const x = Math.min(W - 1, i * step);
      pos[k * 3 + 0] = x * gsd - cx;
      pos[k * 3 + 1] = state.height[y * W + x];   // metres; exaggeration applied on the mesh scale
      pos[k * 3 + 2] = y * gsd - cz;
      uv[k * 2 + 0] = x / (W - 1);
      uv[k * 2 + 1] = 1 - y / (H - 1);
    }
  }

  const idx = [];
  for (let j = 0; j < gh - 1; j++) {
    for (let i = 0; i < gw - 1; i++) {
      const a = j * gw + i, b = a + 1, c = a + gw, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(idx.length > 65535 ? new THREE.Uint32BufferAttribute(idx, 1)
                                  : new THREE.Uint16BufferAttribute(idx, 1));
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    map: texture || null,
    vertexColors: false,
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.DoubleSide,
    flatShading: false,
  });

  mesh = new THREE.Mesh(geo, mat);
  mesh.scale.y = state.vex;
  scene.add(mesh);

  applyMode(state.mode);
  $('s-mesh').textContent = `${(gw * gh / 1000).toFixed(0)}k verts · 1:${step}`;
}

/** Recolour vertices for the current surface mode. */
function applyMode(mode) {
  state.mode = mode;
  if (!mesh) return;
  const m = state.manifest;
  const { w: gw, h: gh, stepX: step } = state.grid;
  const W = m.width;
  const col = mesh.geometry.getAttribute('color');
  const pos = mesh.geometry.getAttribute('position');
  const legend = $('legend');

  if (mode === 'texture') {
    mesh.material.map = mesh.material.userData?.map ?? mesh.material.map;
    mesh.material.vertexColors = false;
    mesh.material.color.setHex(0xffffff);
    mesh.material.needsUpdate = true;
    legend.style.display = 'none';
    return;
  }

  let lo, hi, stops, unit;
  if (mode === 'height') {
    lo = m.height_min_m; hi = m.height_max_m; stops = RAMP_HEIGHT; unit = 'm';
  } else if (mode === 'sigma') {
    if (!state.sigma) {
      alert('This scene has no uncertainty map. Re-export with --sigma to enable it.');
      $('mode').value = state.mode = 'texture';
      return applyMode('texture');
    }
    lo = m.sigma_min_m ?? 0; hi = m.sigma_max_m ?? 1; stops = RAMP_SIGMA; unit = 'm σ';
  } else {                                   // slope, degrees
    lo = 0; hi = 60; stops = RAMP_SIGMA; unit = '°';
  }

  const span = (hi - lo) || 1;
  for (let j = 0; j < gh; j++) {
    for (let i = 0; i < gw; i++) {
      const k = j * gw + i;
      let v;
      if (mode === 'height') {
        v = pos.getY(k);
      } else if (mode === 'sigma') {
        const y = Math.min(m.height - 1, j * step), x = Math.min(W - 1, i * step);
        v = state.sigma[y * W + x];
      } else {
        // Slope from the local height gradient, in true metres per metre.
        const kx = j * gw + Math.min(gw - 1, i + 1);
        const ky = Math.min(gh - 1, j + 1) * gw + i;
        const d = state.gsd * step || 1;
        const dzdx = (pos.getY(kx) - pos.getY(k)) / d;
        const dzdy = (pos.getY(ky) - pos.getY(k)) / d;
        v = Math.atan(Math.hypot(dzdx, dzdy)) * 180 / Math.PI;
      }
      const c = ramp(stops, (v - lo) / span);
      col.setXYZ(k, c[0], c[1], c[2]);
    }
  }
  col.needsUpdate = true;

  if (!mesh.material.userData.map && mesh.material.map) mesh.material.userData.map = mesh.material.map;
  mesh.material.map = null;
  mesh.material.vertexColors = true;
  mesh.material.color.setHex(0xffffff);
  mesh.material.needsUpdate = true;

  legend.style.display = 'block';
  legend.querySelector('.bar').style.background = rampCSS(stops);
  $('legmin').textContent = `${lo.toFixed(1)} ${unit}`;
  $('legmax').textContent = `${hi.toFixed(1)} ${unit}`;
}

function updateStats() {
  const m = state.manifest;
  $('s-res').textContent = `${m.width} × ${m.height} px`;
  if (m.gsd_m) {
    $('s-gsd').textContent = `${m.gsd_m.toFixed(2)} m/px`;
    $('s-extent').textContent = `${(m.width * m.gsd_m).toFixed(0)} × ${(m.height * m.gsd_m).toFixed(0)} m`;
  } else {
    $('s-gsd').textContent = 'unknown';
    $('s-extent').textContent = `${m.width} × ${m.height} px`;
  }
  $('s-range').textContent = `${m.height_min_m.toFixed(1)} – ${m.height_max_m.toFixed(1)} m`;
  $('s-sigma').textContent = m.sigma_mean_m != null ? `± ${m.sigma_mean_m.toFixed(2)} m` : '—';
}

// ---------------------------------------------------------------- fly controls

const keys = new Set();
const vel = new THREE.Vector3();
let yaw = 0, pitch = -0.35, locked = false;

addEventListener('keydown', (e) => {
  keys.add(e.code);
  if (e.code === 'Space') e.preventDefault();
});
addEventListener('keyup', (e) => keys.delete(e.code));

renderer.domElement.addEventListener('click', (e) => {
  if (state.measuring) return pick(e);
  renderer.domElement.requestPointerLock();
});
document.addEventListener('pointerlockchange', () => {
  locked = document.pointerLockElement === renderer.domElement;
  $('crosshair').style.display = locked ? 'block' : 'none';
});
addEventListener('mousemove', (e) => {
  if (!locked) return;
  yaw -= e.movementX * 0.0022;
  pitch -= e.movementY * 0.0022;
  pitch = Math.max(-1.5, Math.min(1.5, pitch));
  state.touring = false;
  $('tour').classList.remove('on');
});

function resetView() {
  const m = state.manifest;
  const gsd = state.gsd || 1;
  const extent = Math.max(m.width, m.height) * gsd;
  camera.position.set(0, extent * 0.45, extent * 0.75);
  yaw = 0;
  pitch = -0.42;
  scene.fog.near = extent * 0.9;
  scene.fog.far = extent * 4;
  camera.far = extent * 12;
  camera.updateProjectionMatrix();
}

function updateCamera(dt) {
  const m = state.manifest;
  if (!m) return;
  const extent = Math.max(m.width, m.height) * (state.gsd || 1);

  if (state.touring) {
    // A slow orbital pass with a gentle bob. This is the shot that goes in the
    // submission video, so it is a deliberate camera move rather than a spin.
    state.tourT += dt * 0.06;
    const r = extent * 0.62;
    const h = extent * (0.28 + 0.10 * Math.sin(state.tourT * 0.7));
    camera.position.set(Math.sin(state.tourT) * r, h, Math.cos(state.tourT) * r);
    camera.lookAt(0, extent * 0.02, 0);
    return;
  }

  camera.rotation.order = 'YXZ';
  camera.rotation.set(pitch, yaw, 0);

  const sprint = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 4 : 1;
  const speed = extent * 0.18 * sprint;
  const dir = new THREE.Vector3();
  if (keys.has('KeyW')) dir.z -= 1;
  if (keys.has('KeyS')) dir.z += 1;
  if (keys.has('KeyA')) dir.x -= 1;
  if (keys.has('KeyD')) dir.x += 1;
  if (dir.lengthSq()) {
    dir.normalize().applyEuler(camera.rotation);
    vel.addScaledVector(dir, speed * dt);
  }
  if (keys.has('KeyE')) vel.y += speed * dt;
  if (keys.has('KeyQ')) vel.y -= speed * dt;

  camera.position.addScaledVector(vel, dt);
  vel.multiplyScalar(Math.pow(0.0016, dt));      // frame-rate independent damping
}

// ---------------------------------------------------------------- measurement (6.3)

const raycaster = new THREE.Raycaster();

function pick(ev) {
  if (!mesh) return;
  const r = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1,
  );
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(mesh, false)[0];
  if (!hit) return;

  // The mesh is scaled vertically for legibility; undo that so the recorded point is
  // in true metres. Forgetting this is the classic way a measurement tool ends up
  // confidently wrong.
  const p = hit.point.clone();
  p.y /= state.vex;
  state.picks.push(p);
  addMarker(hit.point);

  if (state.picks.length === 2) {
    report(state.picks[0], state.picks[1]);
    state.picks = [];
  }
}

function addMarker(worldPoint) {
  if (markers.children.length >= 2) {
    markers.clear();
  }
  const g = new THREE.Mesh(
    new THREE.SphereGeometry(Math.max(0.6, (state.gsd || 1) * 2), 16, 12),
    new THREE.MeshBasicMaterial({ color: 0x4da3ff }),
  );
  g.position.copy(worldPoint);
  markers.add(g);
}

// ------------------------------------------------- error-bar calibration (6.1, 6.3)

// hypot(sigma_a, sigma_b) is only right if the two pixels' errors are independent, and
// measured they are not: correlation is +0.94 across a metre and only reaches zero past
// about 60 m. So the raw bar is roughly 3.5x too wide on a single rooftop, honest at
// 3-15 m, and too narrow across a neighbourhood. This is the measured correction from
// tools/pair_calibration.py, refit per checkpoint. Absent the file the tool still works
// and says plainly that the bar is uncalibrated.
let calib = null;

async function loadCalibration() {
  try {
    const r = await fetch('./calibration.json');
    if (!r.ok) return;
    const j = await r.json();
    if (Array.isArray(j.curve) && j.curve.length) calib = j;
  } catch { calib = null; }
}

function barScale(sepM) {
  if (!calib) return null;
  // Interpolated in log separation. The buckets are log-spaced and the effect tracks
  // spatial error correlation, which decays over scale rather than over metres.
  const pts = calib.curve
    .map((c) => ({ x: Math.log(Math.max(0.3, (c.sep_m[0] + c.sep_m[1]) / 2)), y: c.scale }))
    .sort((p, q) => p.x - q.x);
  const x = Math.log(Math.max(0.3, sepM));
  if (x <= pts[0].x) return pts[0].y;
  if (x >= pts[pts.length - 1].x) return pts[pts.length - 1].y;
  for (let i = 1; i < pts.length; i++) {
    if (x <= pts[i].x) {
      const t = (x - pts[i - 1].x) / (pts[i].x - pts[i - 1].x);
      return pts[i - 1].y + t * (pts[i].y - pts[i - 1].y);
    }
  }
  return pts[pts.length - 1].y;
}

function sigmaAt(p) {
  if (!state.sigma) return null;
  const m = state.manifest, gsd = state.gsd || 1;
  const x = Math.round((p.x + ((m.width - 1) * gsd) / 2) / gsd);
  const y = Math.round((p.z + ((m.height - 1) * gsd) / 2) / gsd);
  if (x < 0 || y < 0 || x >= m.width || y >= m.height) return null;
  return state.sigma[y * m.width + x];
}

function report(a, b) {
  const unit = state.hasMetres ? 'm' : 'px';
  const ground = Math.hypot(b.x - a.x, b.z - a.z);
  const dh = b.y - a.y;
  const slope = Math.hypot(ground, dh);

  $('m-ground').textContent = `${ground.toFixed(2)} ${unit}`;
  $('m-dh').textContent = `${dh >= 0 ? '+' : ''}${dh.toFixed(2)} m`;
  $('m-slope').textContent = `${slope.toFixed(2)} ${unit}`;

  const sa = sigmaAt(a), sb = sigmaAt(b);
  if (sa != null && sb != null) {
    const raw = Math.hypot(sa, sb);
    const k = barScale(ground);
    if (k != null) {
      $('m-unc').textContent = `± ${(raw * k).toFixed(2)} m on Δh`;
      $('m-note').textContent =
        `calibrated ×${k.toFixed(2)} at ${ground.toFixed(0)} m separation`;
    } else {
      $('m-unc').textContent = `± ${raw.toFixed(2)} m on Δh`;
      $('m-note').textContent = 'uncalibrated — assumes independent errors';
    }
  } else {
    $('m-unc').textContent = '—';
    $('m-note').textContent = '';
  }
  $('readout').style.display = 'block';
}

// ---------------------------------------------------------------- UI wiring

$('mode').onchange = (e) => applyMode(e.target.value);

$('vex').oninput = (e) => {
  state.vex = parseFloat(e.target.value);
  $('vexv').textContent = `${state.vex.toFixed(1)}×`;
  if (mesh) mesh.scale.y = state.vex;
};

$('tour').onclick = (e) => {
  state.touring = !state.touring;
  e.target.classList.toggle('on', state.touring);
  if (state.touring && document.pointerLockElement) document.exitPointerLock();
};

$('measure').onclick = (e) => {
  state.measuring = !state.measuring;
  e.target.classList.toggle('on', state.measuring);
  state.picks = [];
  markers.clear();
  $('readout').style.display = state.measuring ? 'block' : 'none';
  if (state.measuring && document.pointerLockElement) document.exitPointerLock();
};

$('wire').onclick = (e) => {
  if (!mesh) return;
  mesh.material.wireframe = !mesh.material.wireframe;
  e.target.classList.toggle('on', mesh.material.wireframe);
};

$('reset').onclick = () => {
  state.touring = false;
  $('tour').classList.remove('on');
  resetView();
};

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- loop

let last = performance.now(), fpsAcc = 0, fpsN = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;

  updateCamera(dt);
  renderer.render(scene, camera);

  fpsAcc += 1 / Math.max(dt, 1e-4);
  if (++fpsN >= 30) {
    $('s-fps').textContent = (fpsAcc / fpsN).toFixed(0);
    fpsAcc = 0; fpsN = 0;
  }
  requestAnimationFrame(frame);
}

$('vexv').textContent = `${state.vex.toFixed(1)}×`;
loadCalibration();
loadScenes().then((ok) => { if (ok) requestAnimationFrame(frame); });
