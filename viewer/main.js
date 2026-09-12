/**
 * DepthWizard viewer.
 *
 * Loads a scene bundle written by tools/export_terrain.py and renders it as a navigable
 * surface. Several things here are deliberate rather than incidental:
 *
 *  1. Heights are real metres. height.bin is raw float32, so nothing is quantised on the
 *     way into the browser. Every number the measurement tool prints is a number the
 *     model actually predicted, in the units ISRO scores.
 *
 *  2. Uncertainty is a first-class view, not a debug overlay. Differentiator 6.1 claims
 *     the model knows where it is unreliable; the only way to make a jury believe that is
 *     to let them look at it and click on it.
 *
 *  3. The error map is shipped, not hidden. Our tall-building bias is real and a
 *     specialist jury will find it in thirty seconds. Showing it deliberately, with the
 *     reason attached, is worth more than a demo with no visible failure mode.
 *
 *  4. The mesh is decimated but the texture is not. Vertex count is what costs frames;
 *     texture resolution is nearly free. So we cap the grid and keep the imagery sharp.
 */
import * as THREE from './vendor/three.module.js';

const $ = (id) => document.getElementById(id);

/**
 * Put failures on screen instead of leaving "Loading…" forever.
 *
 * A stuck loading overlay tells an evaluator nothing and tells us nothing either -- the
 * error is sitting in a console nobody opened. Anything that escapes to the top level
 * lands here, in the overlay, with the message and the line.
 */
function fatal(what, err) {
  const box = document.getElementById('loading');
  if (!box) return;
  box.style.display = 'grid';
  const msg = (err && (err.stack || err.message)) || String(err);
  box.innerHTML = `<div class="err"><b>${what}</b><br><br>` +
    `<code>${String(msg).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]))}</code>` +
    `<br><br>Press F12 for the full trace.</div>`;
}
addEventListener('error', (e) => fatal('The viewer hit an error while starting.', e.error || e.message));
addEventListener('unhandledrejection', (e) => fatal('The viewer failed to load a scene.', e.reason));
// A 1024x1024 tile is 1,048,576 points. A budget of exactly 1M put every DFC2019 tile
// 4.8% over it, so the mesh was decimated 1:2 and rendered at 512 -- rounding off the very
// rooftops the model is judged on, to save 48k vertices. Sized to let a full tile through
// untouched. Measured 165 fps on the 671k-point Sikkim scene, so there is headroom.
const MAX_VERTS = 1_200_000;

const state = {
  manifest: null,
  height: null,      // Float32Array, metres -- our prediction
  truth: null,       // Float32Array, metres -- LiDAR reference, when the scene ships it
  tvalid: null,      // Uint8Array mask, where LiDAR actually has data
  terrain: null,     // Float32Array, metres above the scene datum -- bare earth to stand on
  sigma: null,       // Float32Array, metres
  grid: { w: 0, h: 0, stepX: 1, stepY: 1 },
  vex: 1.5,
  mode: 'texture',
  measuring: false,
  picks: [],
  touring: false,
  tourT: 0,
  comparing: false,
  split: 0.5,        // screen fraction: left of this is ours, right is LiDAR
};

// ---------------------------------------------------------------- renderer & scene

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.localClippingEnabled = true;      // required for the drag-to-compare divider
document.body.appendChild(renderer.domElement);

// Light ground, matching the CSS --bg. Deliberately not pure white: against #fff the
// surface silhouette disappears and every edge glares.
const BG = 0xeef1f4;
const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
scene.fog = new THREE.Fog(BG, 1200, 6000);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.5, 20000);

// A light background needs more ambient than a dark one or the terrain reads as a dark
// blob cut out of the page. Strong hemisphere fill, one directional for the relief.
scene.add(new THREE.HemisphereLight(0xf4f8fc, 0xc2ccd6, 1.15));
const sun = new THREE.DirectionalLight(0xffffff, 1.25);
sun.position.set(-1, 1.5, 1);
scene.add(sun);

let mesh = null;            // our prediction
let truthMesh = null;       // LiDAR reference, only while comparing
let skirt = null, truthSkirt = null;   // solid walls under each surface
let markers = new THREE.Group();
scene.add(markers);

// ---------------------------------------------------------------- colour ramps

// Every ramp is monotonic in lightness, so it stays readable in greyscale and in a
// printed submission PDF. Deliberately NOT a rainbow/jet ramp: it is the classic amateur
// signal in remote sensing and a SAC jury reads it instantly.
//
// On a light background the ramps run light -> dark rather than dark -> bright, or the
// low end dissolves into the page.
const RAMP_HEIGHT = [[0.80,0.87,0.90],[0.55,0.75,0.82],[0.30,0.58,0.71],[0.16,0.40,0.56],[0.06,0.22,0.36]];
const RAMP_SIGMA  = [[0.88,0.90,0.88],[0.95,0.85,0.55],[0.93,0.62,0.28],[0.78,0.30,0.18],[0.50,0.11,0.10]];
// Diverging, centred on zero error: blue = we said too low, red = too high.
const RAMP_ERROR  = [[0.13,0.31,0.55],[0.42,0.60,0.78],[0.92,0.92,0.90],[0.85,0.50,0.40],[0.63,0.12,0.14]];
const GREY_NODATA = [0.72, 0.74, 0.76];

const CAPTION = {
  height: 'Darker means taller. Metres above the ground.',
  sigma:  'Warmer means less certain — the model’s own estimate of how far off it may be.',
  error:  'Blue: we said too low. Red: too high. Pale: we got it right. Grey: no LiDAR here.',
  slope:  'Warmer means steeper ground.',
};

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

async function loadScenes(select) {
  let list = [];
  try {
    list = await (await fetch('./scenes/index.json', { cache: 'no-store' })).json();
  } catch { /* index is optional; fall through to the empty-state message */ }
  const sel = $('scene');
  sel.innerHTML = '';
  if (!list.length) {
    // The commonest cause by far is opening index.html straight off disk: ES modules and
    // fetch are both blocked under file://, so say that first rather than last.
    $('loading').innerHTML =
      `<div class="err"><b>No scenes loaded.</b><br><br>
       If you opened this file directly from a folder, the browser blocks it from reading
       the scene data. Run <code>run_viewer.bat</code> instead, or:<br><br>
       <code>python -m http.server -d viewer 8080</code><br><br>
       To generate a scene:<br>
       <code>python tools/export_terrain.py --height &lt;pred.tif&gt; --texture &lt;rgb.tif&gt; --out viewer/scenes/demo</code>
       </div>`;
    return false;
  }
  for (const s of list) {
    const o = document.createElement('option');
    o.value = s.dir;
    o.textContent = s.name;
    sel.appendChild(o);
  }
  sel.onchange = () => loadScene(sel.value);
  // After an upload we reload the list and jump straight to the new scene, so the user
  // sees their own image rather than having to find it in a dropdown.
  //
  // Otherwise open on the scene flagged `default` in the index, falling back to the first.
  // The list ORDER is deliberate -- it mirrors the problem statement's "urban, sparse,
  // hilly and forested" so the picker answers their criterion on sight -- but the first
  // entry is downtown Omaha, seven buildings, one of them 93 m against training data that
  // stops at 83 m. Opening there leads with our single worst case. The scene stays first
  // in the list and one click away; only the landing scene changes.
  const flagged = list.find((s) => s.default);
  const pick = (select && list.some((s) => s.dir === select)) ? select
             : (flagged ? flagged.dir : list[0].dir);
  sel.value = pick;
  await loadScene(pick);
  return true;
}

async function bin(url, Type) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url);
  return new Type(await r.arrayBuffer());
}

async function loadScene(dir) {
  $('loading').style.display = 'grid';
  $('loading').innerHTML = `<div>Loading ${dir}…</div>`;

  const base = `./scenes/${dir}`;
  const m = await (await fetch(`${base}/manifest.json`, { cache: 'no-store' })).json();
  state.manifest = m;
  // Reflect the loaded scene into the DOM. Assigning select.value does NOT add a
  // `selected` attribute, so a dumped DOM otherwise reports whichever option came first
  // no matter what is actually on screen -- which is exactly how tools/verify_viewer.py
  // got a false failure. Cheap to publish, and it makes the state observable.
  document.body.dataset.scene = dir;
  state.height = await bin(`${base}/${m.files.height}`, Float32Array);

  state.sigma = m.files.sigma ? await bin(`${base}/${m.files.sigma}`, Float32Array).catch(() => null) : null;
  state.truth = m.files.truth ? await bin(`${base}/${m.files.truth}`, Float32Array).catch(() => null) : null;
  state.tvalid = m.files.truth_valid
    ? await bin(`${base}/${m.files.truth_valid}`, Uint8Array).catch(() => null) : null;
  // Bare earth, when the scene has it. Our model outputs height ABOVE GROUND, so without
  // this a mountain renders as a flat plane with buildings standing on it.
  state.terrain = m.files.terrain
    ? await bin(`${base}/${m.files.terrain}`, Float32Array).catch(() => null) : null;

  // Per-building elevation statistics, baked by tools/bake_buildings.py. Only scenes with
  // absolute elevation have them; see the inundation section.
  state.buildings = m.files.buildings
    ? await fetch(`${base}/${m.files.buildings}`).then((r) => r.json())
      .then((d) => d.buildings).catch(() => null) : null;

  let texture = null;
  if (m.files.texture) {
    texture = await new THREE.TextureLoader().loadAsync(`${base}/${m.files.texture}`);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  }

  // A scene without LiDAR (Sikkim, and anything over India) can still be flown through;
  // it just cannot be compared or differenced. Disable rather than fail.
  const hasTruth = !!state.truth;
  $('compare').disabled = !hasTruth;
  $('compare').title = hasTruth ? '' : 'This scene has no LiDAR reference to compare against.';
  $('mode').querySelector('option[value=error]').disabled = !hasTruth;
  if (!hasTruth && (state.mode === 'error' || state.comparing)) setCompare(false), (state.mode = 'texture');
  $('mode').value = state.mode;

  buildMesh(texture);
  updateStats();
  initFlood();
  resetView();
  $('loading').style.display = 'none';
}

// ---------------------------------------------------------------- mesh

/**
 * Build a surface from a height array on the shared decimated grid.
 *
 * When the scene carries bare earth, heights are stacked on top of it: the mesh becomes a
 * true DSM (ground + everything on it) while height.bin stays the above-ground product we
 * are actually scored on. Colour still comes from the above-ground value, so the ramp
 * never quietly turns into an elevation map.
 */
function surfaceGeometry(heights, base) {
  const m = state.manifest;
  const W = m.width, H = m.height;
  const { w: gw, h: gh, stepX: step } = state.grid;
  const gsd = state.gsd;

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
      pos[k * 3 + 1] = heights[y * W + x] + (base ? base[y * W + x] : 0);
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
  return geo;
}

/**
 * Walls dropping from the surface edge to a flat base, plus a floor.
 *
 * Without this the scene is a thin sheet hanging in space with its edges curling, which
 * reads as a rendering artefact rather than a measurement. With it the scene reads as a
 * solid block of cut earth -- the same association a physical terrain model carries.
 *
 * Built as a separate object rather than woven into the grid: the surface switches
 * between a texture drape and four vertex-coloured ramps, and a skirt sharing that
 * material would get the imagery smeared vertically down its walls.
 */
function skirtGeometry(pos, gw, gh, baseY) {
  const P = (i, j) => {
    const k = j * gw + i;
    return [pos.getX(k), pos.getY(k), pos.getZ(k)];
  };
  const border = [];
  for (let i = 0; i < gw; i++) border.push(P(i, 0));
  for (let j = 1; j < gh; j++) border.push(P(gw - 1, j));
  for (let i = gw - 2; i >= 0; i--) border.push(P(i, gh - 1));
  for (let j = gh - 2; j >= 1; j--) border.push(P(0, j));
  border.push(border[0]);

  const v = [];
  for (let n = 0; n < border.length - 1; n++) {
    const a = border[n], b = border[n + 1];
    // Two triangles per edge segment. Winding is irrelevant -- the material is
    // DoubleSide, so a corner walked the "wrong" way still renders solid.
    v.push(a[0], a[1], a[2], b[0], b[1], b[2], b[0], baseY, b[2]);
    v.push(a[0], a[1], a[2], b[0], baseY, b[2], a[0], baseY, a[2]);
  }
  // Floor, so the block is closed when the camera drops below the terrain.
  const c0 = P(0, 0), c1 = P(gw - 1, 0), c2 = P(gw - 1, gh - 1), c3 = P(0, gh - 1);
  v.push(c0[0], baseY, c0[2], c1[0], baseY, c1[2], c2[0], baseY, c2[2]);
  v.push(c0[0], baseY, c0[2], c2[0], baseY, c2[2], c3[0], baseY, c3[2]);

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  g.computeVertexNormals();
  return g;
}

function makeSkirt(surface, gw, gh, extent) {
  const pos = surface.geometry.getAttribute('position');
  let minY = Infinity;
  for (let k = 0; k < pos.count; k++) minY = Math.min(minY, pos.getY(k));
  const base = minY - extent * 0.05;
  const s = new THREE.Mesh(skirtGeometry(pos, gw, gh, base), new THREE.MeshStandardMaterial({
    color: 0xb6bdc4, roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide,
  }));
  // Child of the surface, so vertical exaggeration and any transform carry over for free.
  surface.add(s);
  return s;
}

function disposeMesh(mo) {
  if (!mo) return;
  // Traverse, or the skirt attached as a child leaks its geometry on every scene switch.
  mo.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
  scene.remove(mo);
}

function buildMesh(texture) {
  // The drape is stashed in userData while a colour mode is active, so look in both
  // places or switching scenes leaks a texture per switch.
  const old = mesh && (mesh.material.map || mesh.material.userData.map);
  if (old) old.dispose();
  disposeMesh(mesh); mesh = null; skirt = null;
  disposeMesh(truthMesh); truthMesh = null; truthSkirt = null;

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
  state.gsd = m.gsd_m || 1;
  state.hasMetres = !!m.gsd_m;

  const mat = new THREE.MeshStandardMaterial({
    map: texture || null, vertexColors: false, roughness: 0.95, metalness: 0.0,
    side: THREE.DoubleSide, flatShading: false,
  });
  const extent = Math.max(W, H) * state.gsd;
  mesh = new THREE.Mesh(surfaceGeometry(state.height, state.terrain), mat);
  mesh.scale.y = state.vex;
  scene.add(mesh);
  skirt = makeSkirt(mesh, gw, gh, extent);

  if (state.truth) {
    const tmat = new THREE.MeshStandardMaterial({
      map: texture || null, vertexColors: false, roughness: 0.95, metalness: 0.0,
      side: THREE.DoubleSide, flatShading: false,
    });
    truthMesh = new THREE.Mesh(surfaceGeometry(state.truth, state.terrain), tmat);
    truthMesh.scale.y = state.vex;
    truthMesh.visible = state.comparing;
    scene.add(truthMesh);
    truthSkirt = makeSkirt(truthMesh, gw, gh, extent);
  } else {
    truthSkirt = null;
  }

  applyMode(state.mode);
  setCompare(state.comparing && !!truthMesh);
  $('s-mesh').textContent = `${(gw * gh / 1000).toFixed(0)}k points · 1:${step}`;
}

/**
 * Shade steep faces darker, and multiply that into the satellite drape.
 *
 * The model's roof edges are ramps rather than steps, and a planar-projected texture
 * stretches the roof colour down the full length of every ramp. The result reads as a
 * melted smear -- visually far worse than the height error actually is, because a viewer
 * sees a roof where there should be a wall.
 *
 * Those near-vertical faces ARE walls. Darkening them by surface slope makes them read as
 * walls, which is both what they represent and what any hillshaded product does. It adds
 * no information and hides none: the geometry is untouched and the height ramp, error map
 * and measurements are all unaffected.
 */
function slopeShade(mo) {
  const pos = mo.geometry.getAttribute('position');
  const col = mo.geometry.getAttribute('color');
  const { w: gw, h: gh, stepX: step } = state.grid;
  const d = (state.gsd * step) || 1;
  for (let j = 0; j < gh; j++) {
    for (let i = 0; i < gw; i++) {
      const k = j * gw + i;
      const kx = j * gw + Math.min(gw - 1, i + 1);
      const ky = Math.min(gh - 1, j + 1) * gw + i;
      const g = Math.hypot((pos.getY(kx) - pos.getY(k)) / d, (pos.getY(ky) - pos.getY(k)) / d);
      // cos of the slope angle: 1 on flat ground, falling toward 0 as the face turns
      // vertical. Floored at 0.42 so a wall stays legible rather than going to black.
      const s = Math.max(0.42, 1 / Math.sqrt(1 + g * g));
      col.setXYZ(k, s, s, s);
    }
  }
  col.needsUpdate = true;
  mo.material.vertexColors = true;      // multiplies the drape, rather than replacing it
}

/** Recolour vertices for the current display mode. */
function applyMode(mode) {
  const m = state.manifest;
  if (mode === 'sigma' && !state.sigma) { $('mode').value = state.mode; return; }
  if (mode === 'error' && !state.truth) { $('mode').value = state.mode; return; }
  state.mode = mode;
  if (!mesh) return;

  const { w: gw, h: gh, stepX: step } = state.grid;
  const W = m.width;
  const col = mesh.geometry.getAttribute('color');
  const pos = mesh.geometry.getAttribute('position');
  const legend = $('legend');

  if (mode === 'texture') {
    mesh.material.map = mesh.material.userData?.map ?? mesh.material.map;
    mesh.material.color.setHex(0xffffff);
    slopeShade(mesh);
    mesh.material.needsUpdate = true;
    legend.style.display = 'none';
    styleTruth('texture');
    return;
  }

  let lo, hi, stops, unit;
  if (mode === 'height') {
    lo = m.height_min_m; hi = m.height_max_m; stops = RAMP_HEIGHT; unit = 'm';
  } else if (mode === 'sigma') {
    lo = m.sigma_min_m ?? 0; hi = m.sigma_max_m ?? 1; stops = RAMP_SIGMA; unit = 'm';
  } else if (mode === 'error') {
    // Symmetric about zero, or the colour centre stops meaning "correct". Scaled by the
    // 95th percentile so a handful of extreme roof-edge pixels cannot wash the map out.
    hi = Math.max(2, m.error_px_p95_abs_m ?? 5); lo = -hi; stops = RAMP_ERROR; unit = 'm';
  } else {                                   // slope, degrees
    lo = 0; hi = 60; stops = RAMP_SIGMA; unit = '°';
  }

  const span = (hi - lo) || 1;
  for (let j = 0; j < gh; j++) {
    for (let i = 0; i < gw; i++) {
      const k = j * gw + i;
      const y = Math.min(m.height - 1, j * step), x = Math.min(W - 1, i * step);
      let v, c = null;
      if (mode === 'height') {
        // The above-ground value, NOT the mesh Y. On a terrain-based scene the mesh sits
        // on the mountain, and colouring by Y would turn the height ramp into an
        // elevation map without saying so.
        v = state.height[y * W + x];
      } else if (mode === 'sigma') {
        v = state.sigma[y * W + x];
      } else if (mode === 'error') {
        // Where LiDAR has no data there is no error to report, and inventing one would be
        // exactly the kind of quiet dishonesty this map exists to avoid.
        if (state.tvalid && !state.tvalid[y * W + x]) c = GREY_NODATA;
        else v = state.height[y * W + x] - state.truth[y * W + x];
      } else {
        // Slope from the local height gradient, in true metres per metre.
        const kx = j * gw + Math.min(gw - 1, i + 1);
        const ky = Math.min(gh - 1, j + 1) * gw + i;
        const d = state.gsd * step || 1;
        const dzdx = (pos.getY(kx) - pos.getY(k)) / d;
        const dzdy = (pos.getY(ky) - pos.getY(k)) / d;
        v = Math.atan(Math.hypot(dzdx, dzdy)) * 180 / Math.PI;
      }
      if (!c) c = ramp(stops, (v - lo) / span);
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
  $('legcap').textContent = CAPTION[mode] || '';
  styleTruth(mode);
}

/**
 * Colour the LiDAR surface for the current mode.
 *
 * Height and satellite drape apply to both surfaces, so the comparison is like-for-like.
 * Uncertainty and error are properties of OUR prediction and have no LiDAR counterpart,
 * so that side falls back to plain shaded relief rather than borrowing our colours and
 * implying the reference has an error of its own.
 */
function styleTruth(mode) {
  if (!truthMesh) return;
  const m = state.manifest;
  const { w: gw, h: gh, stepX: step } = state.grid;
  const W = m.width;
  const mat = truthMesh.material;

  if (mode === 'texture') {
    mat.map = mat.userData?.map ?? mat.map;
    mat.color.setHex(0xffffff);
    slopeShade(truthMesh);
    mat.needsUpdate = true;
    return;
  }
  if (!mat.userData.map && mat.map) mat.userData.map = mat.map;
  mat.map = null;

  if (mode === 'height') {
    const col = truthMesh.geometry.getAttribute('color');
    const lo = m.height_min_m, span = (m.height_max_m - m.height_min_m) || 1;
    for (let j = 0; j < gh; j++) {
      for (let i = 0; i < gw; i++) {
        const k = j * gw + i;
        const y = Math.min(m.height - 1, j * step), x = Math.min(W - 1, i * step);
        const c = ramp(RAMP_HEIGHT, (state.truth[y * W + x] - lo) / span);
        col.setXYZ(k, c[0], c[1], c[2]);
      }
    }
    col.needsUpdate = true;
    mat.vertexColors = true;
    mat.color.setHex(0xffffff);
  } else {
    mat.vertexColors = false;
    mat.color.setHex(0xcdd5dd);
  }
  mat.needsUpdate = true;
}

function updateStats() {
  const m = state.manifest;
  $('s-res').textContent = `${m.width} × ${m.height} px`;
  if (m.gsd_m) {
    $('s-gsd').textContent = `${(m.gsd_m * 100).toFixed(0)} cm`;
    $('s-extent').textContent = `${(m.width * m.gsd_m).toFixed(0)} × ${(m.height * m.gsd_m).toFixed(0)} m`;
  } else {
    $('s-gsd').textContent = 'unknown';
    $('s-extent').textContent = `${m.width} × ${m.height} px`;
  }
  $('s-range').textContent = `${m.height_min_m.toFixed(1)} – ${m.height_max_m.toFixed(1)} m`;
  $('s-sigma').textContent = m.sigma_mean_m != null ? `± ${m.sigma_mean_m.toFixed(2)} m` : '—';
  $('s-model').textContent = m.model ? `Heights produced by ${m.model}.` : '';

  const dash = '—';

  // Bare earth, when the scene stands on any.
  const hasTerrain = m.terrain_datum_m != null;
  $('terrain-block').style.display = hasTerrain ? 'block' : 'none';
  if (hasTerrain) {
    const lo = m.terrain_datum_m, hi = lo + m.terrain_relief_m;
    $('s-elev').textContent = `${lo.toFixed(0)} – ${hi.toFixed(0)} m`;
    $('s-relief').textContent = `${m.terrain_relief_m.toFixed(0)} m`;
    $('s-slope').textContent = m.terrain_mean_slope_deg != null
      ? `${m.terrain_mean_slope_deg.toFixed(0)}°` : dash;
    // Provenance, because the mountain is not ours and the viewer must not imply it is.
    $('s-terrnote').textContent =
      `The mountain shape comes from ${m.terrain_source || 'an external DEM'}. `
      + `Everything standing on it — buildings, trees — is our model's.`;
  }

  // Against LiDAR, when there is LiDAR.
  const hasTruth = m.truth_max_m != null;
  $('lidar-block').style.display = hasTruth ? 'block' : 'none';
  $('notruth-block').style.display = hasTruth ? 'none' : 'block';

  if (hasTruth) {
    $('s-ourmax').textContent = `${m.height_max_m.toFixed(1)} m`;
    $('s-trumax').textContent = `${m.truth_max_m.toFixed(1)} m`;
    // Per building is the headline, because it is the only figure comparable to a
    // published one. Per pixel stays on screen underneath it rather than being dropped:
    // it is the number the error map is drawn from, and hiding it would be a cherry-pick.
    const bw = m.building_wise;
    if (bw && bw.n_buildings) {
      $('s-errb').textContent = `${bw.rmse.toFixed(2)} m`;
      // The count belongs next to the number. On a downtown tile this is single digits,
      // and a per-building RMSE over 7 buildings must not be read as a stable result.
      $('s-errbnote').textContent =
        `Across ${bw.n_buildings} building${bw.n_buildings === 1 ? '' : 's'} in this scene, `
        + `one height each. ${bw.n_buildings < 25 ? 'Too few to be a stable figure — the '
          + 'benchmark pools 3,090 buildings over 80 tiles.' : 'The published peer reports '
          + '5.9 m over Asia.'}`;
    } else {
      $('s-errb').textContent = dash;
      $('s-errbnote').textContent = '';
    }
    $('s-err').textContent = m.error_px_rmse_m != null ? `${m.error_px_rmse_m.toFixed(2)} m` : dash;
    $('s-errg').textContent = m.error_ground_px_rmse_m != null
      ? `${m.error_ground_px_rmse_m.toFixed(2)} m` : dash;

    // Say plainly when the reference contains something taller than anything we can
    // produce. This is our known failure and the viewer should name it, not bury it.
    let note = '';
    const gap = m.truth_max_m - m.height_max_m;
    if (gap > 8) {
      note = `The tallest building here is ${gap.toFixed(0)} m higher than anything our `
           + `model produced. Our training data stops at 83 m.`;
    } else if (m.error_px_rmse_m != null) {
      note = 'Measured over every pixel, including roof edges, where error is largest.';
    }
    $('s-errnote').textContent = note;
  } else {
    $('s-notruth').textContent = m.reference_note
      || 'There is no laser survey of this place to check against, so nothing here is '
       + 'scored. It shows the model running on imagery it has never seen.';
  }
}

// ---------------------------------------------------------------- drag to compare

// The divider is a screen-space line but clipping happens in world space, so the plane is
// rebuilt from the camera every frame. Deriving it from the camera (rather than fixing a
// world plane once) is what lets the divider keep meaning the same thing while the camera
// orbits -- a fixed plane would drift off the line the moment you turned.
const clipOurs = new THREE.Plane();
const clipTruth = new THREE.Plane();
const _p0 = new THREE.Vector3(), _a = new THREE.Vector3(), _b = new THREE.Vector3(),
      _t = new THREE.Vector3();

function updateClipPlanes() {
  if (!state.comparing || !truthMesh) return;
  const c = state.split * 2 - 1;                       // screen fraction -> NDC x

  // The set of world points landing on NDC x = c is a plane through the eye. Build it
  // from two rays on that screen line rather than assuming it is perpendicular to the
  // camera's right vector -- under perspective that assumption is only exact at c = 0.
  _p0.copy(camera.position);
  _a.set(c, -1, 0.5).unproject(camera).sub(_p0);
  _b.set(c, 1, 0.5).unproject(camera).sub(_p0);
  const n = _a.cross(_b).normalize();
  clipOurs.setFromNormalAndCoplanarPoint(n, _p0);

  // Orient it so our surface survives on the left of the divider. three.js clips
  // fragments at negative distance.
  _t.set(Math.max(-1, c - 0.2), 0, 0.5).unproject(camera);
  if (clipOurs.distanceToPoint(_t) < 0) clipOurs.negate();
  clipTruth.copy(clipOurs).negate();
}

function setCompare(on) {
  state.comparing = on && !!truthMesh;
  $('compare').classList.toggle('on', state.comparing);
  $('cmp').style.display = state.comparing ? 'block' : 'none';
  // The skirt has to be clipped with its surface or the walls of the hidden half stay on
  // screen, standing in mid-air with nothing on top of them.
  const clip = (o, plane) => {
    if (!o) return;
    o.material.clippingPlanes = state.comparing ? [plane] : null;
    o.material.needsUpdate = true;
  };
  if (truthMesh) truthMesh.visible = state.comparing;
  clip(truthMesh, clipTruth);
  clip(truthSkirt, clipTruth);
  clip(mesh, clipOurs);
  clip(skirt, clipOurs);
  if (state.comparing) {
    if (document.pointerLockElement) document.exitPointerLock();
    layoutDivider();
    updateClipPlanes();
  }
}

function layoutDivider() {
  const x = state.split * innerWidth;
  $('cmp').querySelector('.line').style.left = `${x}px`;
  $('cmp').querySelector('.grip').style.left = `${x}px`;
  $('cmp').querySelector('.tag.l').style.left = `${x}px`;
  $('cmp').querySelector('.tag.r').style.left = `${x}px`;
}

(function wireDivider() {
  const cmp = $('cmp');
  let dragging = false;
  const move = (e) => {
    if (!dragging) return;
    state.split = Math.max(0.04, Math.min(0.96, e.clientX / innerWidth));
    layoutDivider();
  };
  cmp.querySelector('.grip').addEventListener('pointerdown', (e) => {
    dragging = true; e.preventDefault();
    cmp.querySelector('.grip').setPointerCapture(e.pointerId);
  });
  addEventListener('pointermove', move);
  addEventListener('pointerup', () => { dragging = false; });
})();

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
  if (state.comparing) return;            // dragging the divider must not grab the mouse
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

// ---------------------------------------------------------------- ray vs heightfield
//
// The surface is a regular grid, so a general triangle raycast is the wrong instrument.
// three.js ships no BVH: intersectObject() tests every triangle, and at MAX_VERTS that is
// ~2.1M of them. Hover ran one of those casts per frame and the scale bar another every
// tenth frame, so moving the pointer cost more than drawing the scene did.
//
// Marching the grid is O(cells crossed) -- a few hundred -- and returns the same answer,
// because it tests the same two triangles per cell that surfaceGeometry() builds. Cell
// traversal is Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing", 1987.
//
// Everything below works in grid units and TRUE metres. Dividing the ray's y by the
// vertical exaggeration scales origin and direction together, which leaves the ray
// parameter t untouched -- so the point and distance handed back are still measured along
// the original ray, exactly as intersectObject() reported them.

// Surface height at a grid vertex, in true metres. Mirrors surfaceGeometry(): above-ground
// height stacked on bare earth when the scene carries it, so the mesh is a true DSM.
function gridVertexHeight(i, j) {
  const W = state.manifest.width;
  const k = (j * state.grid.stepY) * W + i * state.grid.stepX;
  const h = state.height[k];
  return state.terrain ? h + state.terrain[k] : h;
}

/**
 * Intersect a world-space ray with the rendered surface.
 *
 * Returns { point, distance } in the same scaled world space intersectObject() used, or
 * null on a miss. `point` is rebuilt from the original origin and direction rather than
 * from grid coordinates, so it carries no round-trip error.
 */
function terrainRay(origin, dir) {
  if (!mesh || !state.height) return null;

  const m = state.manifest;
  const W = m.width, H = m.height;
  const { w: gw, h: gh, stepX: step } = state.grid;
  const gsd = state.gsd, vex = state.vex || 1;
  const cell = step * gsd;                       // world metres per grid cell
  const cx = ((W - 1) * gsd) / 2, cz = ((H - 1) * gsd) / 2;

  const ox = (origin.x + cx) / cell, oz = (origin.z + cz) / cell, oy = origin.y / vex;
  const dx = dir.x / cell, dz = dir.z / cell, dy = dir.y / vex;

  // Clip to the grid's footprint first: a ray aimed at the sky should cost two divides,
  // not a walk to the horizon.
  const EPS = 1e-12;
  let t0 = 0, t1 = Infinity;
  const slab = (o, d, hi) => {
    if (Math.abs(d) < EPS) return o >= 0 && o <= hi;
    let ta = (0 - o) / d, tb = (hi - o) / d;
    if (ta > tb) { const s = ta; ta = tb; tb = s; }
    if (ta > t0) t0 = ta;
    if (tb < t1) t1 = tb;
    return t0 <= t1;
  };
  if (!slab(ox, dx, gw - 1) || !slab(oz, dz, gh - 1)) return null;

  // Nudge inside the entry face so the starting cell is never the one behind us.
  const tStart = t0 + 1e-6;
  let i = Math.floor(ox + dx * tStart);
  let j = Math.floor(oz + dz * tStart);
  i = Math.min(gw - 2, Math.max(0, i));
  j = Math.min(gh - 2, Math.max(0, j));

  const si = dx >= 0 ? 1 : -1, sj = dz >= 0 ? 1 : -1;
  const tDeltaX = Math.abs(dx) > EPS ? Math.abs(1 / dx) : Infinity;
  const tDeltaZ = Math.abs(dz) > EPS ? Math.abs(1 / dz) : Infinity;
  let tMaxX = Math.abs(dx) > EPS ? ((dx > 0 ? i + 1 : i) - ox) / dx : Infinity;
  let tMaxZ = Math.abs(dz) > EPS ? ((dz > 0 ? j + 1 : j) - oz) / dz : Infinity;

  // The two triangles of a cell, as the affine height plane h = A + B*u + C*v over the
  // cell's local (u, v). Split and winding match surfaceGeometry()'s (a, c, b), (b, c, d):
  // the first covers u + v <= 1, the second the other half.
  const INSET = 1e-6;
  const hitCell = () => {
    const h00 = gridVertexHeight(i, j), h10 = gridVertexHeight(i + 1, j);
    const h01 = gridVertexHeight(i, j + 1), h11 = gridVertexHeight(i + 1, j + 1);
    const U0 = ox - i, V0 = oz - j;
    let best = null;

    for (let tri = 0; tri < 2; tri++) {
      const A = tri === 0 ? h00 : h01 + h10 - h11;
      const B = tri === 0 ? h10 - h00 : h11 - h01;
      const C = tri === 0 ? h01 - h00 : h11 - h10;
      if (!Number.isFinite(A) || !Number.isFinite(B) || !Number.isFinite(C)) continue;

      const den = dy - B * dx - C * dz;
      if (Math.abs(den) < EPS) continue;               // ray runs parallel to the facet
      const t = (A + B * U0 + C * V0 - oy) / den;
      if (!(t >= t0 - INSET) || t > t1 + INSET) continue;
      if (best !== null && t >= best) continue;

      const u = U0 + dx * t, v = V0 + dz * t;
      if (u < -INSET || u > 1 + INSET || v < -INSET || v > 1 + INSET) continue;
      if (tri === 0 ? u + v > 1 + INSET : u + v < 1 - INSET) continue;
      best = t;
    }
    return best;
  };

  // Bounded by the grid diagonal: a ray cannot cross more cells than that and still be
  // inside. The guard is belt-and-braces against a degenerate direction.
  const maxSteps = gw + gh + 4;
  for (let n = 0; n < maxSteps; n++) {
    const t = hitCell();
    if (t !== null) {
      return {
        point: origin.clone().addScaledVector(dir, t),
        distance: t * dir.length(),
      };
    }
    if (tMaxX < tMaxZ) { i += si; tMaxX += tDeltaX; } else { j += sj; tMaxZ += tDeltaZ; }
    // Stepping out of the grid is the only exit. Bailing early on t1 instead would drop
    // the boundary cell the ray is still inside when it leaves the footprint, which is
    // exactly where a grazing look at the horizon finds its hit.
    if (i < 0 || i > gw - 2 || j < 0 || j > gh - 2) return null;
  }
  return null;
}

// Build the camera ray with three.js so the unprojection stays its business, then march.
function terrainAtNdc(ndc) {
  raycaster.setFromCamera(ndc, camera);
  return terrainRay(raycaster.ray.origin, raycaster.ray.direction);
}

function pick(ev) {
  if (!mesh) return;
  const r = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1,
  );
  const hit = terrainAtNdc(ndc);
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
    new THREE.MeshBasicMaterial({ color: 0x0b5cab }),
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

// ---------------------------------------------------------------- hover readout
//
// Height, confidence and steepness under the cursor, updated as it moves.
//
// The brief asks for "analysis of structural heights and slopes from arbitrary aerial
// perspectives". The two-click tool answers that, but only after a juror has found it and
// worked out that it wants two clicks. Hover answers it on the first mouse movement,
// which is the difference between a capability we have and one they see.
//
// Suppressed entirely while the pointer is locked: in first-person mode the cursor is a
// crosshair steering the camera, and a panel chasing it would be noise.

/** Grid node under a world point, or null if the point falls outside the raster. */
function gridAt(p) {
  const m = state.manifest;
  if (!m) return null;
  const gsd = state.gsd || 1;
  const x = Math.round((p.x + ((m.width - 1) * gsd) / 2) / gsd);
  const y = Math.round((p.z + ((m.height - 1) * gsd) / 2) / gsd);
  if (x < 0 || y < 0 || x >= m.width || y >= m.height) return null;
  return { x, y, W: m.width, H: m.height, gsd };
}

/**
 * The surface the mesh actually shows, terrain included.
 *
 * Slope has to be measured on ground-plus-buildings, not on height above ground. On
 * Sikkim the mountain *is* the slope, and an above-ground-only reading would call a 24°
 * hillside flat.
 */
function surfaceAt(g, x, y) {
  const k = y * g.W + x;
  const h = state.height[k], t = state.terrain ? state.terrain[k] : 0;
  return (Number.isFinite(h) ? h : 0) + (Number.isFinite(t) ? t : 0);
}

/** Steepness in degrees, central differences over the neighbouring nodes. */
function slopeDegAt(g) {
  const xm = Math.max(0, g.x - 1), xp = Math.min(g.W - 1, g.x + 1);
  const ym = Math.max(0, g.y - 1), yp = Math.min(g.H - 1, g.y + 1);
  const dx = (xp - xm) * g.gsd, dy = (yp - ym) * g.gsd;
  const gx = dx > 0 ? (surfaceAt(g, xp, g.y) - surfaceAt(g, xm, g.y)) / dx : 0;
  const gy = dy > 0 ? (surfaceAt(g, g.x, yp) - surfaceAt(g, g.x, ym)) / dy : 0;
  return (Math.atan(Math.hypot(gx, gy)) * 180) / Math.PI;
}

function updateHover(ev) {
  const el = $('hover');
  if (!mesh) { el.style.display = 'none'; return; }
  const r = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1,
  );
  const hit = terrainAtNdc(ndc);
  if (!hit) { el.style.display = 'none'; return; }

  // Undo the vertical exaggeration before reading anything metric -- the same trap
  // pick() documents, and the classic way a readout ends up confidently wrong.
  const p = hit.point.clone();
  p.y /= state.vex;
  const g = gridAt(p);
  if (!g) { el.style.display = 'none'; return; }

  const unit = state.hasMetres ? 'm' : 'px';
  const h = state.height[g.y * g.W + g.x];
  $('h-height').textContent = Number.isFinite(h) ? `${h.toFixed(1)} ${unit}` : '—';
  const s = sigmaAt(p);
  $('h-sigma').textContent =
    s != null && Number.isFinite(s) ? `± ${s.toFixed(1)} ${unit}` : '—';
  $('h-slope').textContent = state.hasMetres ? `${slopeDegAt(g).toFixed(0)}°` : '—';

  // Keep the panel on screen: flip it to the other side of the cursor near an edge.
  const pad = 14, w = el.offsetWidth || 188, hh = el.offsetHeight || 74;
  let left = ev.clientX + pad, top = ev.clientY + pad;
  if (left + w > innerWidth - 4) left = ev.clientX - pad - w;
  if (top + hh > innerHeight - 4) top = ev.clientY - pad - hh;
  el.style.left = `${Math.max(4, left)}px`;
  el.style.top = `${Math.max(4, top)}px`;
  el.style.display = 'block';
}

// ---------------------------------------------------------------- inundation
//
// The problem statement's theme is Disaster Management, and a metric surface model with
// calibrated uncertainty is exactly the input a flood question wants. Three decisions make
// this defensible rather than a gimmick:
//
//   1. It runs on ABSOLUTE elevation (datum + terrain), never on height above ground.
//      Thresholding above-ground height would flood a 3 m hut on a ridge before a 30 m
//      block in a valley, which is backwards.
//   2. Water is filled from the edge of the scene by connectivity, so enclosed hollows
//      stay dry. A bare threshold fills pits no water can reach.
//   3. Buildings within one sigma of the waterline are reported SEPARATELY. Anyone can
//      threshold a raster; only a model with calibrated per-pixel uncertainty can say
//      which answers it does not trust, and ours is calibrated.
//
// The fill is planar: no flow, no volume, no time, no infiltration. The UI says so. A
// crude model labelled honestly beats a crude model dressed up as a good one.
//
// Only scenes with a CRS and a terrain band can support this. The DFC2019 scenes have
// crs null and no terrain, so the tool is absent there and says why -- the same pattern
// the compare and error tools already use on Sikkim.

const WATER_NODES = 256;        // coarse grid for the fill; the water edge is not a claim
let waterMesh = null;

/** True where terrain is at or below L and reachable from the edge of the scene. */
function floodMask(L) {
  const m = state.manifest, W = m.width, H = m.height;
  const datum = m.terrain_datum_m || 0;
  const sx = Math.max(1, Math.floor(W / WATER_NODES));
  const sy = Math.max(1, Math.floor(H / WATER_NODES));
  const gw = Math.floor((W - 1) / sx) + 1, gh = Math.floor((H - 1) / sy) + 1;

  // Terrain only, deliberately. Water is shaped by the ground; buildings stand in it and
  // do not dam it. Including building height here would carve dry channels along rooftops.
  const below = new Uint8Array(gw * gh);
  for (let j = 0; j < gh; j++) {
    const y = Math.min(H - 1, j * sy);
    for (let i = 0; i < gw; i++) {
      const e = datum + state.terrain[y * W + Math.min(W - 1, i * sx)];
      below[j * gw + i] = Number.isFinite(e) && e <= L ? 1 : 0;
    }
  }

  const seen = new Uint8Array(gw * gh);
  const stack = [];
  const seed = (k) => { if (below[k] && !seen[k]) { seen[k] = 1; stack.push(k); } };
  for (let i = 0; i < gw; i++) { seed(i); seed((gh - 1) * gw + i); }
  for (let j = 0; j < gh; j++) { seed(j * gw); seed(j * gw + gw - 1); }
  while (stack.length) {
    const k = stack.pop(), i = k % gw, j = (k - i) / gw;
    if (i > 0) seed(k - 1);
    if (i < gw - 1) seed(k + 1);
    if (j > 0) seed(k - gw);
    if (j < gh - 1) seed(k + gw);
  }
  return { gw, gh, sx, sy, seen };
}

/** Rebuild the water surface and the readout for level L (absolute metres). */
function applyFlood(L) {
  const m = state.manifest, W = m.width, H = m.height;
  const datum = m.terrain_datum_m || 0, gsd = state.gsd;
  const { gw, gh, sx, sy, seen } = floodMask(L);

  if (waterMesh) { scene.remove(waterMesh); waterMesh.geometry.dispose(); waterMesh = null; }

  const cx = ((W - 1) * gsd) / 2, cz = ((H - 1) * gsd) / 2;
  const yLocal = L - datum;               // mesh space; mesh.scale.y applies the exaggeration
  const vi = new Int32Array(gw * gh).fill(-1);
  const pos = [];
  let n = 0;
  for (let j = 0; j < gh; j++) {
    for (let i = 0; i < gw; i++) {
      if (!seen[j * gw + i]) continue;
      vi[j * gw + i] = n++;
      pos.push(Math.min(W - 1, i * sx) * gsd - cx, yLocal, Math.min(H - 1, j * sy) * gsd - cz);
    }
  }
  const idx = [];
  for (let j = 0; j < gh - 1; j++) {
    for (let i = 0; i < gw - 1; i++) {
      // Only where all four corners are under water, so the shoreline stays clean rather
      // than fringed with half-triangles.
      const a = vi[j * gw + i], b = vi[j * gw + i + 1];
      const c = vi[(j + 1) * gw + i], d = vi[(j + 1) * gw + i + 1];
      if (a < 0 || b < 0 || c < 0 || d < 0) continue;
      idx.push(a, c, b, b, c, d);
    }
  }

  if (n && idx.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    g.setIndex(idx);
    waterMesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
      color: 0x2e6f9e, transparent: true, opacity: 0.55,
      side: THREE.DoubleSide, depthWrite: false,
    }));
    waterMesh.scale.y = state.vex;
    scene.add(waterMesh);
  }

  // ---- readout
  let wet = 0;
  for (let k = 0; k < seen.length; k++) wet += seen[k];
  $('f-area').textContent = `${((100 * wet) / seen.length).toFixed(1)} %`;

  if (state.buildings && state.buildings.length) {
    let inund = 0, unc = 0;
    for (const b of state.buildings) {
      const i = Math.min(gw - 1, Math.round(b.x * (gw - 1)));
      const j = Math.min(gh - 1, Math.round(b.y * (gh - 1)));
      if (!seen[j * gw + i]) continue;         // not reachable by water: dry, whatever its height
      const s = b.s || 0;
      if (b.e + s < L) inund += 1;
      else if (Math.abs(b.e - L) <= s) unc += 1;
    }
    $('f-in').textContent = `${inund}`;
    $('f-unc').textContent = `${unc}`;
  } else {
    $('f-in').textContent = '—';
    $('f-unc').textContent = '—';
  }
  $('waterv').textContent = `${L.toFixed(0)} m`;
}

/** Show or hide the tool for the current scene, and set the slider's range. */
function initFlood() {
  const m = state.manifest;
  const ok = state.terrain && m.terrain_datum_m != null && (m.geo || {}).crs;
  const btn = $('flood'), why = $('flood-why'), ctl = $('flood-ctl');

  state.flooding = false;
  btn.classList.remove('on');
  ctl.style.display = 'none';
  if (waterMesh) { scene.remove(waterMesh); waterMesh.geometry.dispose(); waterMesh = null; }

  if (!ok) {
    // Absent, not broken. This scene has no absolute elevation, so any water level we drew
    // would be a number we made up.
    btn.style.display = 'none';
    why.style.display = 'block';
    why.textContent = 'Flooding needs real ground elevation. This scene is not '
      + 'georeferenced, so there is no sea level to measure against.';
    return;
  }
  btn.style.display = '';
  why.style.display = 'none';

  const lo = m.terrain_elev_min_m, hi = m.terrain_elev_max_m;
  const sl = $('water');
  sl.min = Math.floor(lo);
  sl.max = Math.ceil(hi);
  sl.step = Math.max(1, Math.round((hi - lo) / 400));
  sl.value = Math.round(lo + (hi - lo) * 0.15);
  $('f-src').textContent = state.buildings
    ? 'Building outlines: Google Open Buildings; the heights are ours.' : '';
}

$('flood').onclick = () => {
  state.flooding = !state.flooding;
  $('flood').classList.toggle('on', state.flooding);
  $('flood-ctl').style.display = state.flooding ? 'block' : 'none';
  if (state.flooding) applyFlood(parseFloat($('water').value));
  else if (waterMesh) {
    scene.remove(waterMesh); waterMesh.geometry.dispose(); waterMesh = null;
  }
};

// Debounced: the fill plus a geometry rebuild is far too much to run on every input event
// while a slider is being dragged.
let floodTimer = 0;
$('water').addEventListener('input', (e) => {
  $('waterv').textContent = `${parseFloat(e.target.value).toFixed(0)} m`;
  if (!state.flooding) return;
  clearTimeout(floodTimer);
  floodTimer = setTimeout(() => applyFlood(parseFloat(e.target.value)), 90);
});

// ---------------------------------------------------------------- scale bar
//
// A perspective camera has no single scale: metres per pixel changes with depth across the
// frame, so a scale bar in a 3D view is only ever true somewhere. This one is calibrated
// where the centre of the view meets the surface, which is what the reader is looking at,
// and it is redrawn as the camera moves.
//
// It is an orientation aid, not a measurement. The two-click tool is the measurement, and
// it carries a calibrated error bar; this carries none and must never look like it does.

const SCREEN_CENTRE = new THREE.Vector2(0, 0);

/** Round to the nearest 1, 2 or 5 times a power of ten -- standard scale-bar steps. */
function niceLength(target) {
  const p = 10 ** Math.floor(Math.log10(target));
  let best = p;
  for (const k of [1, 2, 5, 10]) {
    if (Math.abs(k * p - target) < Math.abs(best - target)) best = k * p;
  }
  return best;
}

function updateScaleBar() {
  const el = $('scalebar');
  // Without a ground sample distance the scene is in pixels, and a bar in metres would be
  // a fabrication. Say nothing rather than something untrue.
  if (!mesh || !state.hasMetres) { el.style.display = 'none'; return; }

  const hit = terrainAtNdc(SCREEN_CENTRE);
  if (!hit) { el.style.display = 'none'; return; }

  const vh = renderer.domElement.clientHeight || 1;
  const mPerPx = (2 * hit.distance * Math.tan(((camera.fov * Math.PI) / 180) / 2)) / vh;
  if (!Number.isFinite(mPerPx) || mPerPx <= 0) { el.style.display = 'none'; return; }

  const metres = niceLength(120 * mPerPx);        // aim for a bar about 120 px wide
  $('sb-bar').style.width = `${Math.round(metres / mPerPx)}px`;
  $('sb-label').textContent =
    metres >= 1000 ? `${+(metres / 1000).toFixed(2)} km`
      : metres >= 1 ? `${+metres.toFixed(0)} m`
        : `${+(metres * 100).toFixed(0)} cm`;
  el.style.display = 'block';
}

let hoverPending = 0;
renderer.domElement.addEventListener('pointermove', (ev) => {
  if (locked) { $('hover').style.display = 'none'; return; }
  // One raycast per frame at most. A cast per pointermove event is wasted work against a
  // mesh this size and shows up as stutter while orbiting.
  if (hoverPending) return;
  hoverPending = requestAnimationFrame(() => { hoverPending = 0; updateHover(ev); });
});
renderer.domElement.addEventListener('pointerleave', () => {
  $('hover').style.display = 'none';
});

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
      $('m-unc').textContent = `± ${(raw * k).toFixed(2)} m`;
      $('m-note').textContent =
        `on the height difference, calibrated for ${ground.toFixed(0)} m apart`;
    } else {
      $('m-unc').textContent = `± ${raw.toFixed(2)} m`;
      $('m-note').textContent = 'uncalibrated — assumes the two errors are independent';
    }
  } else {
    $('m-unc').textContent = '—';
    $('m-note').textContent = '';
  }
  $('readout').style.display = 'block';
}

// ---------------------------------------------------------------- upload (deliverable)

/**
 * Let the user run the model on their own image.
 *
 * The brief asks for a platform that lets users "upload imagery, visualize reconstructed
 * terrain, and validate estimated height values". The model runs in Python, so this only
 * works behind tools/serve_viewer.py. We ask the server whether it exists rather than
 * assuming: in the single-file build there is no server, and an upload button that cannot
 * work is worse than no button.
 */
async function initUpload() {
  try {
    const r = await fetch('./api/capabilities', { cache: 'no-store' });
    if (!r.ok) return;
    if (!(await r.json()).upload) return;
  } catch { return; }                    // static hosting or the baked build: stay hidden
  $('upload-block').style.display = 'block';

  const send = async (file) => {
    if (!file) return;
    $('upprog').style.display = 'block';
    $('pick').disabled = true;
    const setP = (step, pct) => {
      $('up-step').textContent = step;
      $('up-pct').textContent = `${pct}%`;
      $('up-bar').style.width = `${pct}%`;
    };
    setP('uploading', 4);
    let job;
    try {
      const res = await fetch(`./api/upload?name=${encodeURIComponent(file.name)}`,
                              { method: 'POST', body: file });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || 'upload refused');
      job = j.job;
    } catch (e) {
      setP(String(e.message || e), 100);
      $('pick').disabled = false;
      return;
    }
    // Poll rather than stream: a progress socket is more code and more to go wrong for
    // a job that takes tens of seconds.
    for (;;) {
      await new Promise((r2) => setTimeout(r2, 900));
      let s;
      try { s = await (await fetch(`./api/job/${job}`, { cache: 'no-store' })).json(); }
      catch { continue; }
      setP(s.step || s.state, s.pct ?? 50);
      if (s.state === 'done') {
        await loadScenes(s.scene);
        setP(s.georeferenced ? 'done — heights are above sea level'
                             : 'done — heights are relative (no coordinates in that file)', 100);
        break;
      }
      if (s.state === 'error') { setP(s.step, 100); break; }
    }
    $('pick').disabled = false;
  };

  $('pick').onclick = () => $('file').click();
  $('file').onchange = (e) => send(e.target.files[0]);

  // Drag and drop over the whole window. dragleave fires constantly as the pointer
  // crosses child elements, so track depth rather than trusting a single leave.
  let depth = 0;
  addEventListener('dragover', (e) => { e.preventDefault(); });
  addEventListener('dragenter', (e) => {
    e.preventDefault(); depth++; document.body.classList.add('dragging');
  });
  addEventListener('dragleave', () => {
    if (--depth <= 0) { depth = 0; document.body.classList.remove('dragging'); }
  });
  addEventListener('drop', (e) => {
    e.preventDefault(); depth = 0; document.body.classList.remove('dragging');
    send(e.dataTransfer?.files?.[0]);
  });
}

// ---------------------------------------------------------------- UI wiring

$('mode').onchange = (e) => applyMode(e.target.value);

$('vex').oninput = (e) => {
  state.vex = parseFloat(e.target.value);
  $('vexv').textContent = `${state.vex.toFixed(1)}×`;
  if (mesh) mesh.scale.y = state.vex;
  if (truthMesh) truthMesh.scale.y = state.vex;
  // The water plane is in the same exaggerated space, or it would sit at the wrong height
  // against the terrain the moment the slider moves.
  if (waterMesh) waterMesh.scale.y = state.vex;
};

// Sun azimuth, in the cartographic sense: degrees clockwise from north. 315 (north-west)
// is the standard hillshade default and the one every GIS user's eye expects, because
// lighting from the lower right inverts relief perception.
$('sun').oninput = (e) => {
  const az = parseFloat(e.target.value);
  $('sunv').textContent = `${az.toFixed(0)}°`;
  const r = az * Math.PI / 180;
  sun.position.set(Math.sin(r), 1.5, -Math.cos(r)).multiplyScalar(100);
};

$('compare').onclick = () => setCompare(!state.comparing);

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
  if (truthMesh) truthMesh.material.wireframe = mesh.material.wireframe;
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
  if (state.comparing) layoutDivider();
});

// ---------------------------------------------------------------- loop

let last = performance.now(), fpsAcc = 0, fpsN = 0, sbN = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;

  updateCamera(dt);
  updateClipPlanes();
  renderer.render(scene, camera);

  fpsAcc += 1 / Math.max(dt, 1e-4);
  if (++fpsN >= 30) {
    $('s-fps').textContent = (fpsAcc / fpsN).toFixed(0);
    fpsAcc = 0; fpsN = 0;
  }
  // Every tenth frame is about six updates a second -- past the point anyone notices on a
  // bar that only changes when the camera moves, and it keeps the raycast off the hot path.
  if (++sbN >= 10) { sbN = 0; updateScaleBar(); }
  requestAnimationFrame(frame);
}

$('vexv').textContent = `${state.vex.toFixed(1)}×`;
$('sun').dispatchEvent(new Event('input'));
loadCalibration();
initUpload();
loadScenes().then((ok) => { if (ok) requestAnimationFrame(frame); });
