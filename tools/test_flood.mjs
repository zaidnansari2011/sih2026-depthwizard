/**
 * Acceptance tests for the inundation fill, run against the REAL floodMask() source
 * extracted from viewer/main.js -- not a reimplementation, which would only prove that
 * two copies of my own reasoning agree with each other.
 *
 * Covers criteria 1-3 of docs/feature-plan.md section A.
 */
import { readFileSync } from 'fs';

const src = readFileSync('D:/sih2026/depthwizard/viewer/main.js', 'utf8');

// Pull the function out by brace matching from its declaration.
const start = src.indexOf('function floodMask(');
if (start < 0) throw new Error('floodMask not found');
let depth = 0, end = -1;
for (let i = src.indexOf('{', start); i < src.length; i++) {
  if (src[i] === '{') depth++;
  else if (src[i] === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
}
const fnSrc = src.slice(start, end);

// Take the shipped grid constant from source too, so the test cannot pass at a resolution
// we do not actually ship.
const wn = src.match(/const\s+WATER_NODES\s*=\s*(\d+)/);
if (!wn) throw new Error('WATER_NODES not found');
console.log(`floodMask extracted, WATER_NODES = ${wn[1]}\n`);

// floodMask reads only state.manifest and state.terrain.
let state = {};
const floodMask = new Function('state', 'WATER_NODES', `${fnSrc}; return floodMask;`)(
  new Proxy({}, { get: (_, k) => state[k] }), Number(wn[1]));

function makeScene(W, H, fill) {
  const terrain = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) terrain[y * W + x] = fill(x, y);
  state = { manifest: { width: W, height: H, terrain_datum_m: 0 }, terrain };
}

let pass = 0, fail = 0;
const check = (name, cond, detail = '') => {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}  ${detail}`); }
};
const wet = (m) => m.seen.reduce((a, b) => a + b, 0);

// ---- 1. below the minimum floods nothing; above the maximum floods everything
makeScene(512, 512, (x, y) => 100 + 50 * Math.sin(x / 40) * Math.cos(y / 40));
let m = floodMask(40);
check('level below terrain minimum floods nothing', wet(m) === 0, `wet=${wet(m)}`);
m = floodMask(1000);
check('level above terrain maximum floods all', wet(m) === m.seen.length,
  `wet=${wet(m)} of ${m.seen.length}`);

// ---- 2. monotonic: raising the level never un-floods anything
makeScene(512, 512, (x, y) => 100 + 50 * Math.sin(x / 40) * Math.cos(y / 40));
let prev = -1, mono = true;
for (const L of [60, 80, 100, 120, 140]) {
  const w = wet(floodMask(L));
  if (w < prev) mono = false;
  prev = w;
}
check('flooded area is monotonic in water level', mono);

// ---- 3. THE ONE THAT MATTERS: an enclosed basin below the level must stay dry.
// A high rim everywhere, a deep pit in the middle, and a low border so the edge is wet.
makeScene(256, 256, (x, y) => {
  const inPit = x > 100 && x < 150 && y > 100 && y < 150;
  if (inPit) return 10;                       // deep basin, far below the test level
  const nearEdge = x < 8 || y < 8 || x > 247 || y > 247;
  return nearEdge ? 10 : 900;                 // wall between the edge and the pit
});
m = floodMask(500);
const gw = m.gw, gh = m.gh;
const at = (fx, fy) => m.seen[Math.round(fy * (gh - 1)) * gw + Math.round(fx * (gw - 1))];
check('scene edge floods', at(0.01, 0.5) === 1);
check('ENCLOSED BASIN STAYS DRY (no bathtub fill)', at(0.49, 0.49) === 0,
  'a pit unreachable from the edge was flooded');
check('the wall between them stays dry', at(0.3, 0.5) === 0);

// ---- 4. a channel connecting the basin to the edge must let water in
makeScene(256, 256, (x, y) => {
  const inPit = x > 100 && x < 150 && y > 100 && y < 150;
  if (inPit) return 10;
  const channel = y > 120 && y < 130 && x <= 100;   // carve a path to the west edge
  if (channel) return 10;
  const nearEdge = x < 8 || y < 8 || x > 247 || y > 247;
  return nearEdge ? 10 : 900;
});
m = floodMask(500);
check('the same basin DOES flood once connected', at(0.49, 0.49) === 1);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
