/*
 * ORBWATCH tracker front end.
 *
 * This file draws and interpolates. It computes no orbital mechanics. Every
 * position, angle, lighting flag and pass time comes from the Python API, which
 * uses the tested ORBWATCH modules. Between API samples the browser linearly
 * interpolates, which at the sample spacing used here is well under a
 * kilometre of error for a low orbit, far below one screen pixel.
 */

import { createBehaviourView } from "./behaviour.js";
import { createMissionView } from "./mission.js";

const d3 = window.d3;
const topojson = window.topojson;

// ---------------------------------------------------------------------------
// Presets. Object IDs were checked against Celestrak on 2026-09-16. Station and
// city coordinates are approximate and for orientation only.
// ---------------------------------------------------------------------------

const OBJECT_PRESETS = [
  { norad: 25544, label: "ISS (ZARYA)" },
  { norad: 48274, label: "CSS (TIANHE)" },
  { norad: 20580, label: "HUBBLE SPACE TELESCOPE" },
  { norad: 27386, label: "ENVISAT (DEFUNCT)" },
  { norad: 40697, label: "SENTINEL-2A" },
  { norad: 33591, label: "NOAA 19" },
  { norad: 41866, label: "GOES 16 (GEO)" },
  { norad: 40882, label: "INMARSAT 5-F3 (GEO, HELD)" },
  { norad: 27438, label: "INTELSAT 905 (GEO, INCLINED OPS)" },
];

const STATION_PRESETS = [
  { name: "Zurich", lat: 47.3769, lon: 8.5417, alt: 0.45 },
  { name: "Pasadena", lat: 34.2013, lon: -118.1714, alt: 0.35 },
  { name: "Svalbard", lat: 78.2297, lon: 15.3975, alt: 0.5 },
  { name: "Kiruna", lat: 67.8571, lon: 20.9644, alt: 0.4 },
  { name: "Kourou", lat: 5.2514, lon: -52.8047, alt: 0.05 },
  { name: "Madrid DSN", lat: 40.4314, lon: -4.2481, alt: 0.83 },
  { name: "Canberra DSN", lat: -35.4014, lon: 148.9817, alt: 0.69 },
  { name: "Goldstone DSN", lat: 35.4267, lon: -116.89, alt: 1.0 },
];

const CITIES = [
  ["ZURICH", 47.37, 8.54], ["PARIS", 48.86, 2.35], ["LONDON", 51.51, -0.13],
  ["BERLIN", 52.52, 13.4], ["MADRID", 40.42, -3.7], ["ROME", 41.9, 12.5],
  ["MOSCOW", 55.76, 37.62], ["CAIRO", 30.04, 31.24], ["NAIROBI", -1.29, 36.82],
  ["CAPE TOWN", -33.92, 18.42], ["DUBAI", 25.2, 55.27], ["NEW DELHI", 28.61, 77.21],
  ["BEIJING", 39.9, 116.4], ["TOKYO", 35.68, 139.69], ["SINGAPORE", 1.35, 103.82],
  ["SYDNEY", -33.87, 151.21], ["HONOLULU", 21.31, -157.86], ["ANCHORAGE", 61.22, -149.9],
  ["LOS ANGELES", 34.05, -118.24], ["HOUSTON", 29.76, -95.37], ["NEW YORK", 40.71, -74.01],
  ["MEXICO CITY", 19.43, -99.13], ["LIMA", -12.05, -77.04], ["SAO PAULO", -23.55, -46.63],
  ["BUENOS AIRES", -34.6, -58.38], ["REYKJAVIK", 64.15, -21.94],
  ["KOUROU", 5.24, -52.77], ["BAIKONUR", 45.96, 63.31], ["CAPE CANAVERAL", 28.39, -80.6],
];

const LAYERS = [
  ["grid", "GRATICULE", true],
  ["borders", "BORDERS", true],
  ["cities", "CITIES", true],
  ["night", "NIGHT SIDE", true],
  ["track", "GROUND TRACK", true],
  ["footprint", "FOOTPRINT", true],
  ["station", "STATION", true],
  ["stars", "STARFIELD", true],
  ["detailed", "DETAILED COASTS", false],
];

// Detailed coastlines have about ten times the vertices of the coarse ones and
// are redrawn every frame, so they are only used when asked for and when the
// globe is still enough to show them: not while dragging or zooming, and not at
// time warps where the globe turns every frame.
const DETAILED_MAX_RATE = 10;

const RATES = [-3600, -600, -60, -10, 1, 10, 60, 600, 3600];
const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

const COLORS = {
  ocean0: "#061a3a", ocean1: "#020714",
  land: "rgba(22, 70, 104, 0.42)", coast: "#45cdf5", coastGlow: "rgba(69, 205, 245, 0.16)",
  border: "rgba(79, 209, 255, 0.2)", grid: "rgba(48, 92, 210, 0.72)", equator: "rgba(90, 140, 255, 0.9)",
  day: "rgba(70, 150, 235, 0.17)", night: "rgba(0, 2, 10, 0.66)", twilight: "rgba(0, 2, 10, 0.16)",
  terminator: "rgba(255, 181, 71, 0.5)",
  track: "#ff4d5e", trackPast: "rgba(255, 77, 94, 0.38)",
  foot: "#ffe45c", station: "#ff5fd2", inview: "#3dff8a", sun: "#ffe45c",
  city: "rgba(200, 232, 255, 0.78)", cityDot: "rgba(255, 90, 110, 0.9)",
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const stored = JSON.parse(localStorage.getItem("orbwatch.settings") || "{}");

const state = {
  mainView: "globe",
  norad: null,
  object: null,
  track: null,
  passes: null,
  skyTracks: new Map(),
  site: stored.site || { ...STATION_PRESETS[0] },
  mask: stored.mask ?? 0,
  layers: Object.fromEntries(LAYERS.map(([key, , on]) => [key, stored.layers?.[key] ?? on])),
  view: { rotate: [-8.5, -30, 0], zoom: 1, follow: true, interacting: false },
  inflight: { track: false, passes: false, sky: new Set() },
  failures: { track: 0, passes: 0 },
  lastPanelUpdate: 0,
};

const clock = { baseSim: Date.now(), baseWall: performance.now(), rate: 1, paused: false };

function saveSettings() {
  localStorage.setItem(
    "orbwatch.settings",
    JSON.stringify({ site: state.site, mask: state.mask, layers: state.layers }),
  );
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const pad2 = (n) => String(Math.floor(n)).padStart(2, "0");
const wrap180 = (deg) => ((((deg + 180) % 360) + 360) % 360) - 180;

function fmtHMS(totalSeconds, { signed = false } = {}) {
  if (!Number.isFinite(totalSeconds)) return "--:--:--";
  const sign = totalSeconds < 0 ? "-" : signed ? "+" : "";
  const s = Math.abs(totalSeconds);
  const days = Math.floor(s / 86400);
  const hms = `${pad2((s % 86400) / 3600)}:${pad2((s % 3600) / 60)}:${pad2(s % 60)}`;
  return days > 0 ? `${sign}${days}d ${hms}` : `${sign}${hms}`;
}

const fmtUTC = (ms) => new Date(ms).toISOString().slice(11, 19);

function fmtDate(ms) {
  const d = new Date(ms);
  return `${pad2(d.getUTCDate())} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

function fmtLocal(ms) {
  const d = new Date(ms);
  const time = d.toLocaleTimeString([], { hour12: false });
  const zone = new Intl.DateTimeFormat([], { timeZoneName: "short" })
    .formatToParts(d)
    .find((part) => part.type === "timeZoneName")?.value ?? "";
  return `${time} ${zone}`;
}

const fmtLat = (v) => `${Math.abs(v).toFixed(2)}° ${v >= 0 ? "N" : "S"}`;
const fmtLon = (v) => `${Math.abs(v).toFixed(2)}° ${v >= 0 ? "E" : "W"}`;

function compass(azDeg) {
  const points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  return points[Math.round((((azDeg % 360) + 360) % 360) / 22.5) % 16];
}

function setText(id, text, cls) {
  const el = $(id);
  if (el.textContent !== text) el.textContent = text;
  if (cls !== undefined) el.className = `v ${cls}`.trim();
}

let toastTimer = null;
function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 5000);
}

// ---------------------------------------------------------------------------
// Simulation clock
// ---------------------------------------------------------------------------

function simNow() {
  if (clock.paused) return clock.baseSim;
  return clock.baseSim + (performance.now() - clock.baseWall) * clock.rate;
}

function rebase() {
  clock.baseSim = simNow();
  clock.baseWall = performance.now();
}

function setRate(rate) { rebase(); clock.rate = rate; }
function togglePause() { rebase(); clock.paused = !clock.paused; }
function jumpBy(ms) { rebase(); clock.baseSim += ms; }
function jumpTo(ms) { rebase(); clock.baseSim = ms; }
function goLive() { clock.baseSim = Date.now(); clock.baseWall = performance.now(); clock.rate = 1; clock.paused = false; }

function isLive() {
  return !clock.paused && clock.rate === 1 && Math.abs(simNow() - Date.now()) < 2000;
}

function stepRate(direction) {
  const index = RATES.indexOf(clock.rate);
  const next = clamp((index === -1 ? RATES.indexOf(1) : index) + direction, 0, RATES.length - 1);
  if (clock.paused) togglePause();
  setRate(RATES[next]);
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function getJSON(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
  const response = await fetch(url);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

const siteKey = () => `${state.site.lat}|${state.site.lon}|${state.site.alt}|${state.mask}`;

function siteParams() {
  return { lat: state.site.lat, lon: state.site.lon, alt: state.site.alt, site: state.site.name, mask: state.mask };
}

function periodSeconds() {
  return state.object ? state.object.nominal_period_min * 60 : 5400;
}

async function loadObject(norad) {
  $("loading").classList.remove("hidden");
  try {
    const object = await getJSON("/api/object", { norad });
    state.norad = norad;
    state.object = object;
    state.track = null;
    state.passes = null;
    state.skyTracks.clear();
    state.view.follow = true;
    updateHash();
    document.title = `${object.name} · ORBWATCH`;
    $("norad-input").value = norad;
    renderElements();
    behaviour.setObject(norad, object.name);
    mission.setObject(norad, object.name);
  } catch (error) {
    toast(`Could not load NORAD ${norad}: ${error.message}`);
    if (!state.object) $("loading").querySelector("span").textContent = "NO ELEMENT SET LOADED";
    return;
  }
  await ensureTrack(simNow(), true);
  $("loading").classList.add("hidden");
}

async function ensureTrack(now, force = false) {
  if (!state.object || state.inflight.track) return;
  const period = periodSeconds();
  const track = state.track;
  const stale =
    force ||
    !track ||
    track.norad !== state.norad ||
    track.siteKey !== siteKey() ||
    Math.abs(now - track.centerMs) > 0.35 * period * 1000;
  if (!stale) return;
  if (state.failures.track > 0 && performance.now() < state.failures.retryAt) return;

  state.inflight.track = true;
  const requestNorad = state.norad;
  const requestSite = siteKey();
  try {
    const before = Math.min(period, 259200);
    const after = Math.min(1.5 * period, 259200);
    const step = clamp(Math.round(period / 240), 10, 600);
    const payload = await getJSON("/api/track", {
      norad: requestNorad, t: new Date(now).toISOString(), before, after, step, ...siteParams(),
    });
    // The object or station may have changed while this request was in flight.
    if (requestNorad !== state.norad || requestSite !== siteKey()) return;
    payload.centerMs = now;
    payload.siteKey = requestSite;
    payload.norad = requestNorad;
    state.track = payload;
    state.failures.track = 0;
  } catch (error) {
    state.failures.track += 1;
    state.failures.retryAt = performance.now() + Math.min(30000, 2000 * state.failures.track);
    toast(`Track request failed: ${error.message}`);
  } finally {
    state.inflight.track = false;
  }
}

async function ensurePasses(now) {
  if (!state.object || state.inflight.passes) return;
  const passes = state.passes;
  const lastLos = passes?.passes.length ? passes.passes[passes.passes.length - 1].los_unix_ms : null;
  const stale =
    !passes ||
    passes.norad !== state.norad ||
    passes.siteKey !== siteKey() ||
    now < passes.window_start_unix_ms ||
    now > passes.window_start_unix_ms + 12 * 3600 * 1000 ||
    (lastLos !== null && now > lastLos);
  if (!stale) return;
  if (state.failures.passes > 0 && performance.now() < state.failures.passesRetryAt) return;

  state.inflight.passes = true;
  const requestNorad = state.norad;
  const requestSite = siteKey();
  try {
    const hours = periodSeconds() > 36000 ? 48 : 24;
    const payload = await getJSON("/api/passes", {
      norad: requestNorad, t: new Date(now).toISOString(), hours, ...siteParams(),
    });
    if (requestNorad !== state.norad || requestSite !== siteKey()) return;
    payload.norad = requestNorad;
    payload.siteKey = requestSite;
    state.passes = payload;
    state.failures.passes = 0;
    renderPassTable(now);
  } catch (error) {
    state.failures.passes += 1;
    state.failures.passesRetryAt = performance.now() + Math.min(60000, 4000 * state.failures.passes);
    toast(`Pass prediction failed: ${error.message}`);
  } finally {
    state.inflight.passes = false;
  }
}

async function ensureSkyTrack(pass) {
  const key = `${state.norad}|${pass.aos_unix_ms}|${siteKey()}`;
  if (state.skyTracks.has(key) || state.inflight.sky.has(key)) return state.skyTracks.get(key);
  state.inflight.sky.add(key);
  try {
    const halfSpan = pass.duration_s / 2;
    const center = new Date(pass.aos_unix_ms + halfSpan * 1000).toISOString();
    const step = clamp(Math.round(pass.duration_s / 120), 5, 600);
    const payload = await getJSON("/api/track", {
      norad: state.norad, t: center, before: halfSpan + step, after: halfSpan + step, step, ...siteParams(),
    });
    state.skyTracks.set(key, payload);
    return payload;
  } catch {
    return undefined;
  } finally {
    state.inflight.sky.delete(key);
  }
}

// ---------------------------------------------------------------------------
// Interpolation between API samples
// ---------------------------------------------------------------------------

const LINEAR_FIELDS = ["lat_deg", "alt_km", "speed_km_s", "footprint_deg", "elevation_deg", "range_km",
  "subsolar_lat_deg", "site_sun_elevation_deg", "age_days", "orbit_number"];
const ANGLE_FIELDS = ["lon_deg", "azimuth_deg", "subsolar_lon_deg"];
const FLAG_FIELDS = ["valid", "sunlit", "in_view", "optically_visible"];

function sampleAt(track, ms) {
  const s = track.samples;
  const t = s.t_unix_ms;
  if (!t.length || ms < t[0] || ms > t[t.length - 1]) return null;

  let lo = 0;
  let hi = t.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= ms) lo = mid; else hi = mid;
  }
  if (!s.valid[lo] || !s.valid[hi]) {
    const nearest = ms - t[lo] < t[hi] - ms ? lo : hi;
    if (!s.valid[nearest]) return null;
    lo = hi = nearest;
  }

  const f = hi === lo ? 0 : (ms - t[lo]) / (t[hi] - t[lo]);
  const out = { t: ms };
  for (const key of LINEAR_FIELDS) out[key] = s[key][lo] + (s[key][hi] - s[key][lo]) * f;
  for (const key of ANGLE_FIELDS) {
    const span = key === "azimuth_deg" ? 360 : 360;
    let delta = s[key][hi] - s[key][lo];
    delta = ((((delta + span / 2) % span) + span) % span) - span / 2;
    out[key] = s[key][lo] + delta * f;
  }
  out.lon_deg = wrap180(out.lon_deg);
  out.subsolar_lon_deg = wrap180(out.subsolar_lon_deg);
  out.azimuth_deg = ((out.azimuth_deg % 360) + 360) % 360;
  for (const key of FLAG_FIELDS) out[key] = s[key][f < 0.5 ? lo : hi];
  return out;
}

// ---------------------------------------------------------------------------
// Globe rendering
// ---------------------------------------------------------------------------

const canvas = $("globe");
const ctx = canvas.getContext("2d");
const projection = d3.geoOrthographic().clipAngle(90).precision(0.35);
const geoPath = d3.geoPath(projection, ctx);
const graticule = d3.geoGraticule10();
const equator = { type: "LineString", coordinates: d3.range(-180, 181, 2).map((lon) => [lon, 0]) };

const geo = { land50: null, land110: null, borders: null };
let width = 0;
let height = 0;
let baseRadius = 0;
let starCanvas = null;

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  width = rect.width;
  height = rect.height;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  baseRadius = Math.min(width, height) * 0.43;
  projection.translate([width / 2, height / 2 + 6]);
  buildStars(dpr);
}

function buildStars(dpr) {
  starCanvas = document.createElement("canvas");
  starCanvas.width = canvas.width;
  starCanvas.height = canvas.height;
  const sctx = starCanvas.getContext("2d");
  sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  let seed = 7;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let k = 0; k < Math.round((width * height) / 2600); k += 1) {
    const r = random() < 0.92 ? random() * 0.8 + 0.2 : random() * 1.2 + 0.8;
    sctx.fillStyle = `rgba(${200 + random() * 55}, ${210 + random() * 45}, 255, ${0.25 + random() * 0.6})`;
    sctx.beginPath();
    sctx.arc(random() * width, random() * height, r, 0, 2 * Math.PI);
    sctx.fill();
  }
}

const viewCenter = () => [-state.view.rotate[0], -state.view.rotate[1]];
const onNearSide = (lonLat) => d3.geoDistance(lonLat, viewCenter()) < Math.PI / 2 - 0.01;

function strokeGeo(geometry, style, lineWidth, dash = []) {
  ctx.beginPath();
  geoPath(geometry);
  ctx.setLineDash(dash);
  ctx.strokeStyle = style;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  ctx.setLineDash([]);
}

function fillGeo(geometry, style) {
  ctx.beginPath();
  geoPath(geometry);
  ctx.fillStyle = style;
  ctx.fill();
}

function drawBackdrop(cx, cy, r) {
  ctx.clearRect(0, 0, width, height);
  if (state.layers.stars && starCanvas) {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(starCanvas, 0, 0);
    ctx.restore();
  }
  const halo = ctx.createRadialGradient(cx, cy, r * 0.96, cx, cy, r * 1.16);
  halo.addColorStop(0, "rgba(79, 209, 255, 0.34)");
  halo.addColorStop(0.35, "rgba(40, 120, 255, 0.12)");
  halo.addColorStop(1, "rgba(40, 120, 255, 0)");
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(cx, cy, r * 1.16, 0, 2 * Math.PI);
  ctx.fill();

  const ocean = ctx.createRadialGradient(cx - r * 0.35, cy - r * 0.4, r * 0.1, cx, cy, r);
  ocean.addColorStop(0, COLORS.ocean0);
  ocean.addColorStop(1, COLORS.ocean1);
  ctx.fillStyle = ocean;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.fill();
}

function drawDay(cur) {
  if (!state.layers.night || !cur) return;
  const subsolar = [cur.subsolar_lon_deg, cur.subsolar_lat_deg];
  fillGeo(d3.geoCircle().center(subsolar).radius(90)(), COLORS.day);
}

function drawNight(cur) {
  if (!state.layers.night || !cur) return;
  const antisolar = [wrap180(cur.subsolar_lon_deg + 180), -cur.subsolar_lat_deg];
  for (const radius of [102, 96]) {
    fillGeo(d3.geoCircle().center(antisolar).radius(radius)(), COLORS.twilight);
  }
  const night = d3.geoCircle().center(antisolar).radius(90)();
  fillGeo(night, COLORS.night);
  strokeGeo(night, COLORS.terminator, 1, [3, 5]);
}

function drawEarth(cur) {
  const detailed =
    state.layers.detailed && geo.land50 && !state.view.interacting && Math.abs(clock.rate) <= DETAILED_MAX_RATE;
  const land = detailed ? geo.land50 : geo.land110;
  if (!land) return;

  drawDay(cur);
  fillGeo(land, COLORS.land);
  drawNight(cur);

  if (state.layers.grid) {
    strokeGeo(graticule, COLORS.grid, 0.6);
    strokeGeo(equator, COLORS.equator, 0.9);
  }
  strokeGeo(land, COLORS.coastGlow, 3.2);
  strokeGeo(land, COLORS.coast, 0.95);
  if (state.layers.borders && geo.borders) strokeGeo(geo.borders, COLORS.border, 0.55);

  if (state.layers.cities) {
    ctx.font = "10px 'Share Tech Mono', monospace";
    ctx.textBaseline = "middle";
    const placed = [];
    const site = [state.site.lon, state.site.lat];
    if (state.layers.station && onNearSide(site)) {
      const [sx, sy] = projection(site);
      placed.push({ x0: sx - 8, y0: sy - 9, x1: sx + 12 + state.site.name.length * 8, y1: sy + 9 });
    }
    const overlaps = (box) => placed.some((p) => box.x0 < p.x1 && box.x1 > p.x0 && box.y0 < p.y1 && box.y1 > p.y0);

    for (const [name, lat, lon] of CITIES) {
      if (!onNearSide([lon, lat])) continue;
      const [x, y] = projection([lon, lat]);
      const box = { x0: x - 3, y0: y - 12, x1: x + 8 + ctx.measureText(name).width, y1: y + 3 };
      if (overlaps(box)) continue;
      placed.push(box);
      ctx.fillStyle = COLORS.cityDot;
      ctx.beginPath();
      ctx.arc(x, y, 1.8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = COLORS.city;
      ctx.fillText(name, x + 5, y - 6);
    }
  }
}

function drawTrack(now) {
  if (!state.layers.track || !state.track) return;
  const s = state.track.samples;
  const past = [];
  const future = [];
  let run = [];

  const flush = (into) => { if (run.length > 1) into.push(run); run = []; };
  let side = null;
  for (let k = 0; k < s.t_unix_ms.length; k += 1) {
    const isFuture = s.t_unix_ms[k] > now;
    if (side !== null && isFuture !== side) {
      const cur = sampleAt(state.track, now);
      if (cur) run.push([cur.lon_deg, cur.lat_deg]);
      flush(side ? future : past);
      if (cur) run.push([cur.lon_deg, cur.lat_deg]);
    }
    side = isFuture;
    if (!s.valid[k]) { flush(isFuture ? future : past); continue; }
    run.push([s.lon_deg[k], s.lat_deg[k]]);
  }
  flush(side ? future : past);

  for (const segment of past) {
    strokeGeo({ type: "LineString", coordinates: segment }, COLORS.trackPast, 1.3, [4, 4]);
  }
  for (const segment of future) {
    const chunk = 18;
    for (let k = 0; k < segment.length - 1; k += chunk) {
      const piece = segment.slice(k, Math.min(segment.length, k + chunk + 1));
      const fade = 1 - (0.7 * k) / Math.max(1, segment.length);
      ctx.globalAlpha = fade * 0.35;
      strokeGeo({ type: "LineString", coordinates: piece }, COLORS.track, 5);
      ctx.globalAlpha = fade;
      strokeGeo({ type: "LineString", coordinates: piece }, COLORS.track, 1.8);
    }
    ctx.globalAlpha = 1;
  }
}

function drawFootprint(cur) {
  if (!state.layers.footprint || !cur) return;
  const circle = d3.geoCircle().center([cur.lon_deg, cur.lat_deg]).radius(cur.footprint_deg)();
  fillGeo(circle, "rgba(255, 228, 92, 0.045)");
  strokeGeo(circle, COLORS.foot, 1.2, [2, 4]);
}

function drawStation(cur, t) {
  if (!state.layers.station) return;
  const site = [state.site.lon, state.site.lat];
  const inView = cur?.in_view;
  const ringColor = inView ? COLORS.inview : COLORS.station;

  if (cur) {
    const ring = d3.geoCircle().center(site).radius(cur.footprint_deg)();
    strokeGeo(ring, ringColor, inView ? 1.6 : 1.1);
    if (inView) {
      strokeGeo(
        { type: "LineString", coordinates: [site, [cur.lon_deg, cur.lat_deg]] },
        `rgba(61, 255, 138, ${0.55 + 0.35 * Math.sin(t * 6)})`, 1.4, [6, 4],
      );
    }
  }

  if (!onNearSide(site)) return;
  const [x, y] = projection(site);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = ringColor;
  ctx.shadowColor = ringColor;
  ctx.shadowBlur = 10;
  ctx.fillRect(-4, -4, 8, 8);
  ctx.restore();
  ctx.font = "600 12px 'Rajdhani', sans-serif";
  ctx.fillStyle = ringColor;
  ctx.textBaseline = "middle";
  ctx.fillText(state.site.name.toUpperCase(), x + 10, y + 1);
}

function drawSun(cur) {
  if (!cur) return;
  const sun = [cur.subsolar_lon_deg, cur.subsolar_lat_deg];
  if (!onNearSide(sun)) return;
  const [x, y] = projection(sun);
  const glow = ctx.createRadialGradient(x, y, 0, x, y, 22);
  glow.addColorStop(0, "rgba(255, 240, 150, 0.95)");
  glow.addColorStop(0.3, "rgba(255, 228, 92, 0.55)");
  glow.addColorStop(1, "rgba(255, 228, 92, 0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, 22, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = COLORS.sun;
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, 2 * Math.PI);
  ctx.fill();
}

function drawSatellite(cur, t) {
  if (!cur) return;
  const point = [cur.lon_deg, cur.lat_deg];
  if (!onNearSide(point)) return;
  const [x, y] = projection(point);
  const color = cur.sunlit ? "255, 240, 170" : "140, 170, 220";

  const pulse = (t * 0.9) % 1;
  ctx.strokeStyle = `rgba(${color}, ${0.7 * (1 - pulse)})`;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(x, y, 6 + pulse * 22, 0, 2 * Math.PI);
  ctx.stroke();

  const glow = ctx.createRadialGradient(x, y, 0, x, y, 16);
  glow.addColorStop(0, `rgba(${color}, 0.9)`);
  glow.addColorStop(1, `rgba(${color}, 0)`);
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, 16, 0, 2 * Math.PI);
  ctx.fill();

  ctx.strokeStyle = `rgba(${color}, 0.95)`;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    ctx.moveTo(x + dx * 7, y + dy * 7);
    ctx.lineTo(x + dx * 13, y + dy * 13);
  }
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(x, y, 3.2, 0, 2 * Math.PI);
  ctx.fill();

  ctx.font = "700 14px 'Rajdhani', sans-serif";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(234, 248, 255, 0.95)";
  const label = state.object?.name ?? "";
  ctx.fillText(label, x + 16, y - 14);
  ctx.font = "11px 'Share Tech Mono', monospace";
  ctx.fillStyle = "rgba(255, 228, 92, 0.9)";
  ctx.fillText(`${cur.alt_km.toFixed(0)} KM`, x + 16, y + 1);
}

function updateFollow(cur) {
  if (!state.view.follow || !cur) return;
  const target = [-cur.lon_deg, -clamp(cur.lat_deg, -60, 60) * 0.7];
  const rotate = state.view.rotate;
  rotate[0] += wrap180(target[0] - rotate[0]) * 0.07;
  rotate[1] += (target[1] - rotate[1]) * 0.07;
  rotate[0] = wrap180(rotate[0]);
}

function frame() {
  const now = simNow();
  const t = performance.now() / 1000;
  ensureTrack(now);
  ensurePasses(now);

  const cur = state.track ? sampleAt(state.track, now) : null;
  updateFollow(cur);

  if (state.mainView === "globe" && width > 0 && height > 0) {
    const r = baseRadius * state.view.zoom;
    projection.scale(r).rotate(state.view.rotate);
    const [cx, cy] = projection.translate();

    drawBackdrop(cx, cy, r);
    drawEarth(cur);
    drawTrack(now);
    drawFootprint(cur);
    drawStation(cur, t);
    drawSun(cur);
    drawSatellite(cur, t);
  }

  if (t * 1000 - state.lastPanelUpdate > 100) {
    state.lastPanelUpdate = t * 1000;
    updatePanels(now, cur);
  }
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------

function currentAndNextPass(now) {
  const list = state.passes?.passes ?? [];
  const current = list.find((p) => p.aos_unix_ms <= now && now <= p.los_unix_ms) ?? null;
  const next = list.find((p) => p.aos_unix_ms > now) ?? null;
  return { current, next };
}

function updatePanels(now, cur) {
  const object = state.object;
  const live = isLive();

  const mode = $("status-mode");
  if (clock.paused) { mode.textContent = "PAUSED"; mode.className = "pill pill-sim"; }
  else if (live) { mode.textContent = "LIVE"; mode.className = "pill pill-live"; }
  else { mode.textContent = `SIM ×${clock.rate}`; mode.className = "pill pill-sim"; }
  $("rate").textContent = clock.paused ? "PAUSED" : `×${clock.rate}`;
  $("btn-play").textContent = clock.paused ? "▶" : "❚❚";
  document.querySelector(".btn-live").classList.toggle("on", live);
  $("btn-follow").classList.toggle("on", state.view.follow);
  $("hud-view").textContent = state.view.follow ? "FOLLOW" : "FREE";

  $("clock-time").textContent = fmtUTC(now);
  $("clock-date").textContent = `${fmtDate(now)} · UTC${live ? "" : " · SIMULATED"}`;

  if (!object) return;

  const ageDays = (now - object.epoch_unix_ms) / 86400000;
  const agePill = $("status-age");
  agePill.textContent = `TLE ${ageDays >= 0 ? "+" : ""}${ageDays.toFixed(1)} D`;
  agePill.className = `pill ${Math.abs(ageDays) < 3 ? "pill-good" : Math.abs(ageDays) < 14 ? "pill-warn" : "pill-bad"}`;

  $("hud-object").textContent = object.name;
  setText("t-name", object.name);
  setText("t-norad", String(object.norad_id));
  setText("t-cospar", object.international_designator || "--");
  setText("t-utc", `${fmtDate(now)} ${fmtUTC(now)}`);
  setText("t-local", fmtLocal(now));
  setText("t-age", fmtHMS(ageDays * 86400, { signed: true }),
    Math.abs(ageDays) < 3 ? "good" : Math.abs(ageDays) < 14 ? "warn" : "bad");
  setText("t-period", fmtHMS(object.nominal_period_min * 60));
  setText("t-incl", `${object.mean_elements.inclination_deg.toFixed(4)}°`);

  if (cur) {
    setText("t-orbit", cur.orbit_number.toFixed(2));
    setText("t-alt", `${cur.alt_km.toFixed(2)} km`);
    setText("t-lat", fmtLat(cur.lat_deg));
    setText("t-lon", fmtLon(cur.lon_deg));
    setText("t-speed", `${cur.speed_km_s.toFixed(3)} km/s`);
    setText("t-light", cur.sunlit ? "SUNLIT" : "ECLIPSE", cur.sunlit ? "" : "dim");
    setText("t-range", `${cur.range_km.toFixed(1)} km`, cur.in_view ? "" : "bad");
    setText("t-elev", `${cur.elevation_deg.toFixed(2)}°`, cur.in_view ? "good" : "bad");
    setText("t-azim", `${cur.azimuth_deg.toFixed(2)}° ${compass(cur.azimuth_deg)}`);
    const status = cur.optically_visible ? ["VISIBLE", "good"]
      : cur.in_view ? ["IN VIEW", "good"] : ["BELOW HORIZON", "dim"];
    setText("t-status", status[0], status[1]);
  } else {
    for (const id of ["t-orbit", "t-alt", "t-lat", "t-lon", "t-speed", "t-light", "t-range", "t-elev", "t-azim", "t-status"]) {
      setText(id, "--", "dim");
    }
  }
  setText("t-site", `${state.site.name.toUpperCase()} · MASK ${state.mask}°`);

  const { current, next } = currentAndNextPass(now);
  const focus = current ?? next;
  const countdownValue = $("countdown-value");
  if (current && current.starts_before_window && current.ends_after_window) {
    $("countdown-label").textContent = "CONTINUOUS VIEW";
    countdownValue.textContent = "IN VIEW";
    countdownValue.className = "countdown-value inview";
    $("countdown-sub").textContent = `${state.site.name.toUpperCase()} · WHOLE PREDICTION WINDOW`;
  } else if (current) {
    $("countdown-label").textContent = "IN VIEW · LOS IN";
    countdownValue.textContent = fmtHMS((current.los_unix_ms - now) / 1000);
    countdownValue.className = "countdown-value inview";
    $("countdown-sub").textContent = `${state.site.name.toUpperCase()} · MAX EL ${current.max_elevation_deg.toFixed(1)}°`;
  } else if (next) {
    $("countdown-label").textContent = "NEXT AOS IN";
    countdownValue.textContent = fmtHMS((next.aos_unix_ms - now) / 1000);
    countdownValue.className = "countdown-value";
    $("countdown-sub").textContent = `${state.site.name.toUpperCase()} · ${fmtUTC(next.aos_unix_ms)} UTC · MAX EL ${next.max_elevation_deg.toFixed(1)}°`;
  } else {
    $("countdown-label").textContent = "NEXT AOS";
    countdownValue.textContent = state.passes ? "NONE" : "--:--:--";
    countdownValue.className = "countdown-value";
    $("countdown-sub").textContent = state.passes ? `NO PASS OVER ${state.site.name.toUpperCase()} IN WINDOW` : "";
  }

  if (focus) {
    setText("t-aos", focus.starts_before_window ? "IN PROGRESS" : `${fmtUTC(focus.aos_unix_ms)} · ${focus.aos_azimuth_deg.toFixed(0)}° ${compass(focus.aos_azimuth_deg)}`, "");
    setText("t-maxel", `${focus.max_elevation_deg.toFixed(1)}° · ${fmtUTC(focus.max_elevation_unix_ms)}`, focus.max_elevation_deg > 30 ? "good" : "");
    setText("t-los", focus.ends_after_window ? "BEYOND WINDOW" : `${fmtUTC(focus.los_unix_ms)} · ${focus.los_azimuth_deg.toFixed(0)}° ${compass(focus.los_azimuth_deg)}`, "");
  } else {
    for (const id of ["t-aos", "t-maxel", "t-los"]) setText(id, "--", "dim");
  }

  drawSky(now, cur, focus, Boolean(current));
  highlightPassRows(now);
}

function drawSky(now, cur, pass, inPass) {
  const sky = $("skyplot");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = 150;
  if (sky.width !== size * dpr) { sky.width = size * dpr; sky.height = size * dpr; }
  const g = sky.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, size, size);

  const c = size / 2;
  const R = c - 14;
  const toXY = (az, el) => {
    const rr = ((90 - clamp(el, 0, 90)) / 90) * R;
    const a = (az * Math.PI) / 180;
    return [c + rr * Math.sin(a), c - rr * Math.cos(a)];
  };

  const bg = g.createRadialGradient(c, c, 0, c, c, R);
  bg.addColorStop(0, "rgba(10, 30, 60, 0.9)");
  bg.addColorStop(1, "rgba(3, 8, 18, 0.9)");
  g.fillStyle = bg;
  g.beginPath();
  g.arc(c, c, R, 0, 2 * Math.PI);
  g.fill();

  g.strokeStyle = "rgba(79, 209, 255, 0.25)";
  g.lineWidth = 1;
  for (const el of [0, 30, 60]) {
    g.beginPath();
    g.arc(c, c, ((90 - el) / 90) * R, 0, 2 * Math.PI);
    g.stroke();
  }
  g.beginPath();
  g.moveTo(c - R, c); g.lineTo(c + R, c);
  g.moveTo(c, c - R); g.lineTo(c, c + R);
  g.stroke();

  g.fillStyle = "rgba(79, 209, 255, 0.8)";
  g.font = "10px 'Share Tech Mono', monospace";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText("N", c, 6); g.fillText("S", c, size - 6);
  g.fillText("E", size - 6, c); g.fillText("W", 6, c);

  if (state.mask > 0) {
    g.strokeStyle = "rgba(255, 95, 210, 0.45)";
    g.setLineDash([2, 3]);
    g.beginPath();
    g.arc(c, c, ((90 - state.mask) / 90) * R, 0, 2 * Math.PI);
    g.stroke();
    g.setLineDash([]);
  }

  $("sky-caption").textContent = pass ? (inPass ? "SKY · CURRENT PASS" : "SKY · NEXT PASS") : "SKY · NO PASS";
  if (!pass) return;

  const key = `${state.norad}|${pass.aos_unix_ms}|${siteKey()}`;
  const track = state.skyTracks.get(key);
  if (!track) { ensureSkyTrack(pass); return; }

  const s = track.samples;
  g.strokeStyle = "rgba(255, 77, 94, 0.9)";
  g.lineWidth = 1.8;
  g.beginPath();
  let started = false;
  for (let k = 0; k < s.t_unix_ms.length; k += 1) {
    if (!s.valid[k] || s.elevation_deg[k] < 0) { started = false; continue; }
    const [x, y] = toXY(s.azimuth_deg[k], s.elevation_deg[k]);
    if (!started) { g.moveTo(x, y); started = true; } else { g.lineTo(x, y); }
  }
  g.stroke();

  if (!pass.starts_before_window) {
    const [x, y] = toXY(pass.aos_azimuth_deg, 0);
    g.fillStyle = "#3dff8a";
    g.beginPath(); g.arc(x, y, 3, 0, 2 * Math.PI); g.fill();
  }

  if (inPass && cur && cur.in_view) {
    const [x, y] = toXY(cur.azimuth_deg, cur.elevation_deg);
    g.fillStyle = "#ffffff";
    g.shadowColor = "#ffe45c";
    g.shadowBlur = 10;
    g.beginPath(); g.arc(x, y, 4, 0, 2 * Math.PI); g.fill();
    g.shadowBlur = 0;
  }
}

function renderPassTable(now) {
  const body = $("p-body");
  const payload = state.passes;
  $("p-site").textContent = state.site.name.toUpperCase();
  $("p-mask").textContent = `MASK ${state.mask}°`;
  body.replaceChildren();

  if (!payload || payload.passes.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "empty";
    cell.textContent = payload ? "NO PASSES IN WINDOW" : "--";
    row.append(cell);
    body.append(row);
    return;
  }

  for (const pass of payload.passes) {
    const row = document.createElement("tr");
    row.dataset.aos = pass.aos_unix_ms;
    row.dataset.los = pass.los_unix_ms;
    const cells = [
      fmtDate(pass.aos_unix_ms).slice(0, 6),
      pass.starts_before_window ? "<WIN" : fmtUTC(pass.aos_unix_ms),
      pass.ends_after_window ? "WIN>" : fmtUTC(pass.los_unix_ms),
      fmtHMS(pass.duration_s).replace(/^00:/, ""),
      null,
      `${compass(pass.aos_azimuth_deg)}→${compass(pass.los_azimuth_deg)}`,
    ];
    cells.forEach((text, index) => {
      const cell = document.createElement("td");
      if (index === 4) {
        cell.textContent = `${pass.max_elevation_deg.toFixed(0)}°`;
        const bar = document.createElement("span");
        bar.className = "maxel-bar";
        bar.style.width = `${Math.round((pass.max_elevation_deg / 90) * 40)}px`;
        cell.append(bar);
      } else {
        cell.textContent = text;
      }
      row.append(cell);
    });
    row.addEventListener("click", () => {
      jumpTo(pass.aos_unix_ms - 120000);
      state.view.follow = true;
      selectTab("telemetry");
    });
    body.append(row);
  }
  highlightPassRows(now);
}

function highlightPassRows(now) {
  for (const row of $("p-body").querySelectorAll("tr[data-aos]")) {
    const aos = Number(row.dataset.aos);
    const los = Number(row.dataset.los);
    row.classList.toggle("current", aos <= now && now <= los);
    row.classList.toggle("past", los < now);
  }
}

function renderElements() {
  const object = state.object;
  $("e-tle").textContent = `${object.name}\n${object.line1}\n${object.line2}`;
  const m = object.mean_elements;
  const rows = [
    ["EPOCH", `${object.epoch_utc.slice(0, 19).replace("T", " ")} Z`],
    ["INCLINATION", `${m.inclination_deg.toFixed(4)}°`],
    ["RAAN", `${m.raan_deg.toFixed(4)}°`],
    ["ECCENTRICITY", m.eccentricity.toFixed(7)],
    ["ARG PERIGEE", `${m.argument_of_perigee_deg.toFixed(4)}°`],
    ["MEAN ANOMALY", `${m.mean_anomaly_deg.toFixed(4)}°`],
    ["MEAN MOTION", `${m.mean_motion_rev_per_day.toFixed(8)} rev/d`],
    ["NOMINAL PERIOD", `${object.nominal_period_min.toFixed(2)} min`],
    ["B*", `${m.bstar_per_earth_radius.toExponential(4)} /ER`],
    ["REV AT EPOCH", String(m.revolution_number_at_epoch)],
    ["ELEMENT SET", String(m.element_set_number)],
  ];
  const container = $("e-rows");
  container.replaceChildren();
  for (const [key, value] of rows) {
    const row = document.createElement("div");
    row.className = "row";
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = key;
    const v = document.createElement("span");
    v.className = "v";
    v.textContent = value;
    row.append(k, v);
    container.append(row);
  }
}

// ---------------------------------------------------------------------------
// Setup and interaction
// ---------------------------------------------------------------------------

const TABS = ["telemetry", "passes", "elements", "setup", "events", "mission"];
const VIEW_TAB = { behaviour: "events", mission: "mission" };

function selectTab(name) {
  for (const tab of document.querySelectorAll(".tab")) tab.classList.toggle("active", tab.dataset.tab === name);
  for (const body of document.querySelectorAll(".tab-body")) body.classList.toggle("active", body.id === `tab-${name}`);
  if (name === "events" && state.mainView !== "behaviour") setMainView("behaviour");
  if (name === "mission" && state.mainView !== "mission") setMainView("mission");
}

function updateHash() {
  if (state.norad === null) return;
  const view = state.mainView === "globe" ? "" : `&view=${state.mainView}`;
  history.replaceState(null, "", `#norad=${state.norad}${view}`);
}

function setMainView(name) {
  state.mainView = name;
  $("globe-wrap").hidden = name !== "globe";
  $("behaviour-wrap").hidden = name !== "behaviour";
  $("mission-wrap").hidden = name !== "mission";
  for (const button of document.querySelectorAll(".view-switch button")) {
    button.classList.toggle("on", button.dataset.view === name);
  }
  if (name === "behaviour") behaviour.activate();
  else behaviour.deactivate();
  if (name === "mission") mission.activate();
  else mission.deactivate();
  if (name === "globe") resize();
  else selectTab(VIEW_TAB[name]);
  updateHash();
}

const behaviour = createBehaviourView({
  getJSON,
  onShowRequested: () => {
    if (state.mainView !== "behaviour") setMainView("behaviour");
  },
});

const mission = createMissionView({
  getJSON,
  onShowRequested: () => {
    if (state.mainView !== "mission") setMainView("mission");
  },
});

function fillSiteForm() {
  $("site-name").value = state.site.name;
  $("site-lat").value = state.site.lat;
  $("site-lon").value = state.site.lon;
  $("site-alt").value = state.site.alt;
  $("site-mask").value = state.mask;
  const preset = STATION_PRESETS.findIndex((p) => p.name === state.site.name && p.lat === state.site.lat && p.lon === state.site.lon);
  $("site-preset").value = preset >= 0 ? String(preset) : "";
}

function applySite(site, mask) {
  state.site = site;
  state.mask = mask;
  state.track = null;
  state.passes = null;
  state.skyTracks.clear();
  saveSettings();
  fillSiteForm();
  renderPassTable(simNow());
}

function setupControls() {
  for (const preset of OBJECT_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.norad;
    option.textContent = `${preset.norad} · ${preset.label}`;
    $("object-preset").append(option);
  }
  $("object-preset").addEventListener("change", (event) => {
    if (event.target.value) loadObject(Number(event.target.value));
    event.target.value = "";
  });
  $("object-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const value = $("norad-input").value.trim();
    if (/^\d+$/.test(value)) loadObject(Number(value));
    else toast("Enter a numeric NORAD catalogue number.");
  });

  const presetSelect = $("site-preset");
  presetSelect.append(new Option("CUSTOM", ""));
  STATION_PRESETS.forEach((preset, index) => presetSelect.append(new Option(preset.name.toUpperCase(), String(index))));
  presetSelect.addEventListener("change", () => {
    const preset = STATION_PRESETS[Number(presetSelect.value)];
    if (!preset) return;
    $("site-name").value = preset.name;
    $("site-lat").value = preset.lat;
    $("site-lon").value = preset.lon;
    $("site-alt").value = preset.alt;
  });
  $("site-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const lat = Number($("site-lat").value);
    const lon = Number($("site-lon").value);
    const alt = Number($("site-alt").value);
    const mask = Number($("site-mask").value);
    const name = $("site-name").value.trim() || "Site";
    if (![lat, lon, alt, mask].every(Number.isFinite) || Math.abs(lat) > 90 || Math.abs(lon) > 180 || alt < -0.5 || alt > 10 || mask < 0 || mask > 45) {
      toast("Station values out of range: lat ±90, lon ±180, alt −0.5 to 10 km, mask 0 to 45°.");
      return;
    }
    applySite({ name, lat, lon, alt }, mask);
  });

  const toggles = $("layer-toggles");
  for (const [key, label] of LAYERS) {
    const wrapper = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.layers[key];
    input.addEventListener("change", () => { state.layers[key] = input.checked; saveSettings(); });
    wrapper.append(input, document.createTextNode(label));
    toggles.append(wrapper);
  }

  for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", () => selectTab(tab.dataset.tab));
  for (const button of document.querySelectorAll(".view-switch button")) {
    button.addEventListener("click", () => setMainView(button.dataset.view));
  }

  $("e-copy").addEventListener("click", async () => {
    if (!state.object) return;
    try {
      await navigator.clipboard.writeText(`${state.object.name}\n${state.object.line1}\n${state.object.line2}`);
      $("e-copy").textContent = "COPIED";
      setTimeout(() => ($("e-copy").textContent = "COPY TLE"), 1500);
    } catch {
      toast("Clipboard unavailable.");
    }
  });

  const actions = {
    back: () => jumpBy(-600000),
    fwd: () => jumpBy(600000),
    slower: () => stepRate(-1),
    faster: () => stepRate(1),
    toggle: () => togglePause(),
    live: () => { goLive(); state.view.follow = true; },
  };
  for (const button of document.querySelectorAll(".controls button")) {
    button.addEventListener("click", () => actions[button.dataset.act]?.());
  }
  $("btn-follow").addEventListener("click", () => { state.view.follow = !state.view.follow; });

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    const key = event.key.toLowerCase();
    if (key === " ") { event.preventDefault(); actions.toggle(); }
    else if (key === "arrowright") actions.faster();
    else if (key === "arrowleft") actions.slower();
    else if (key === "l") actions.live();
    else if (key === "f") state.view.follow = !state.view.follow;
    else if (key === "b") setMainView(state.mainView === "behaviour" ? "globe" : "behaviour");
    else if (key === "m") setMainView(state.mainView === "mission" ? "globe" : "mission");
    else if (key === "p") {
      if (state.mainView !== "mission") setMainView("mission");
      mission.toggleMode();
    }
    else if (["1", "2", "3", "4", "5", "6"].includes(key)) selectTab(TABS[Number(key) - 1]);
  });

  let last = null;
  d3.select(canvas)
    .call(
      d3.drag()
        .on("start", (event) => { state.view.follow = false; state.view.interacting = true; last = [event.x, event.y]; })
        .on("drag", (event) => {
          const k = 90 / (baseRadius * state.view.zoom);
          state.view.rotate[0] = wrap180(state.view.rotate[0] + (event.x - last[0]) * k);
          state.view.rotate[1] = clamp(state.view.rotate[1] - (event.y - last[1]) * k, -90, 90);
          last = [event.x, event.y];
        })
        .on("end", () => { state.view.interacting = false; }),
    )
    .on("dblclick", () => { state.view.follow = true; });

  let wheelTimer = null;
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.view.interacting = true;
    state.view.zoom = clamp(state.view.zoom * Math.exp(-event.deltaY * 0.0012), 0.55, 8);
    clearTimeout(wheelTimer);
    wheelTimer = setTimeout(() => { state.view.interacting = false; }, 180);
  }, { passive: false });

  new ResizeObserver(resize).observe(canvas.parentElement);
}

async function loadGeography() {
  const [land50, world110] = await Promise.all([
    fetch("data/land-50m.json").then((r) => r.json()),
    fetch("data/countries-110m.json").then((r) => r.json()),
  ]);
  geo.land50 = topojson.feature(land50, land50.objects.land);
  geo.land110 = topojson.feature(world110, world110.objects.land);
  geo.borders = topojson.mesh(world110, world110.objects.countries, (a, b) => a !== b);
}

function followHash() {
  const norad = /norad=(\d+)/.exec(window.location.hash);
  const found = /view=(behaviour|mission)/.exec(window.location.hash);
  const view = found ? found[1] : "globe";
  if (view !== state.mainView) setMainView(view);
  if (norad && Number(norad[1]) !== state.norad) loadObject(Number(norad[1]));
}

async function main() {
  setupControls();
  fillSiteForm();
  resize();
  requestAnimationFrame(frame);

  try {
    await loadGeography();
  } catch (error) {
    toast(`Map data failed to load: ${error.message}`);
  }

  window.addEventListener("hashchange", followHash);
  const fromHash = /norad=(\d+)/.exec(window.location.hash);
  const startView = /view=(behaviour|mission)/.exec(window.location.hash);
  if (startView) setMainView(startView[1]);
  await loadObject(fromHash ? Number(fromHash[1]) : OBJECT_PRESETS[0].norad);
}

main();
