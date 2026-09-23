/*
 * ORBWATCH behaviour view.
 *
 * One object's element set history over months: the orbit it actually flew,
 * the manoeuvres the events layer found in it, what the detector set aside and
 * why, and the delta-v budgets that check what it could not see.
 *
 * Like the globe, this file draws and computes no physics. Every number comes
 * from /api/behaviour, which runs orbwatch.events.
 */

const d3 = window.d3;

const C = {
  cyan: "#4fd1ff",
  amber: "#ffb547",
  yellow: "#ffe45c",
  green: "#3dff8a",
  red: "#ff4d5e",
  dim: "#6f86a6",
};

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
const MARGIN = { left: 88, right: 22, top: 4, bottom: 26 };
const HEADER = 24;
const GAP = 10;
const SHARES = [0.37, 0.37, 0.26];
const DAY_MS = 86400000;

const $ = (id) => document.getElementById(id);
const pad2 = (n) => String(Math.floor(n)).padStart(2, "0");

function fmtDay(ms) {
  const d = new Date(ms);
  return `${pad2(d.getUTCDate())} ${MONTHS[d.getUTCMonth()]}`;
}
const fmtDate = (ms) => `${fmtDay(ms)} ${new Date(ms).getUTCFullYear()}`;
const fmtClock = (ms) => new Date(ms).toISOString().slice(11, 16);
const fmtStamp = (ms) => `${fmtDate(ms)} ${fmtClock(ms)}`;

function tickFormat(date) {
  if (d3.utcDay(date) < date) return fmtClock(+date);
  if (d3.utcMonth(date) < date) return fmtDay(+date);
  if (d3.utcYear(date) < date) return MONTHS[date.getUTCMonth()];
  return String(date.getUTCFullYear());
}

function fmtMagnitude(value) {
  if (value >= 1000) return `${d3.format("~g")(value / 1000)}K`;
  return d3.format("~g")(value);
}

function fmtSpeed(value) {
  if (value === null || value === undefined) return "--";
  return value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function row(key, value, cls = "") {
  const node = el("div", "row");
  node.append(el("span", "k", key), el("span", `v ${cls}`.trim(), value));
  return node;
}

/** Operational name of each axis. At GEO the in-plane signal is east-west
 * station-keeping and the out-of-plane one north-south; elsewhere they are
 * changes of altitude and of inclination. */
function axisName(kind, geo, short = false) {
  if (kind === "in_plane") return short ? (geo ? "E-W" : "ALT") : geo ? "EAST-WEST" : "ALTITUDE";
  return short ? (geo ? "N-S" : "INCL") : geo ? "NORTH-SOUTH" : "INCLINATION";
}

function changeText(event) {
  if (event.kind === "in_plane") {
    const km = event.delta_a_km;
    const sign = km >= 0 ? "+" : "−";
    return Math.abs(km) >= 1 ? `${sign}${Math.abs(km).toFixed(2)} km` : `${sign}${Math.round(Math.abs(km) * 1000)} m`;
  }
  const arcsec = event.change_arcsec;
  return `${arcsec < 0 ? "−" : ""}${Math.abs(arcsec).toFixed(1)}″`;
}

function statusText(event) {
  if (event.status === "manoeuvre") return `${event.profile.toUpperCase()} MANOEUVRE`;
  if (event.status === "drag_surge") return "DRAG SURGE · NATURAL";
  return {
    below_floor: "SET ASIDE · BELOW FLOOR",
    incoherent: "SET ASIDE · INCOHERENT",
    excursion: "SET ASIDE · RETURNED EXCURSION",
  }[event.status];
}

function explain(event, data) {
  const geo = data.regime === "geosynchronous";
  if (event.status === "manoeuvre" && event.profile === "impulsive") {
    return "One gap carries most of the change: a burn.";
  }
  if (event.status === "manoeuvre") {
    return geo
      ? "Spread over several gaps: low thrust or a sequence of small burns."
      : "A rise spread over several gaps: a long low-thrust burn, or several small burns close together.";
  }
  if (event.status === "drag_surge") {
    return "Decay faster than the days before, spread over several updates: extra drag, usually a geomagnetic storm heating the upper atmosphere. Natural, not a burn, and counted as drag in the budget.";
  }
  if (event.status === "below_floor") {
    return `Under the ${data.floors_m_s[event.kind]} m/s floor and under ${data.floor_override_sigmas[event.kind]}σ, inside what fit noise produces on satellites that cannot manoeuvre.`;
  }
  if (event.status === "incoherent") {
    return "Its per-gap steps cancel: an element set off the path and back, not a burn.";
  }
  return "Undone by another event within 10 days: a run of bad element sets, not a burn.";
}

const PURPOSE = {
  in_plane: "replacing altitude lost to drag",
  out_of_plane: "holding the orbit plane in place",
};

const fmtSig = (value) => d3.format(".2~r")(value);

function noRequirement(kind, geo, budget) {
  if (kind === "in_plane" && geo) {
    return "No physical requirement: east-west keeping corrects in both directions, so the net drift bounds nothing.";
  }
  if (kind === "in_plane" && budget.required_m_s === 0) {
    return "No raises needed: the altitude history is explained by drag alone.";
  }
  if (kind === "in_plane") return "No physical requirement over this span.";
  return "No physical requirement: there is no natural out-of-plane model outside GEO.";
}

function bandClass(event) {
  if (event.status === "manoeuvre") return event.profile;
  return event.status === "drag_surge" ? "drag" : "aside";
}

const isShown = (event) => event.status === "manoeuvre" || event.status === "drag_surge";
const eventKey = (event) => `${event.kind}|${event.start_unix_ms}|${event.status}`;

// ---------------------------------------------------------------------------

export function createBehaviourView({ getJSON, onShowRequested }) {
  const charts = $("bh-charts");
  const tooltip = $("bh-tooltip");
  const svg = d3.select(charts).append("svg").attr("class", "bh-svg");

  const view = {
    norad: null,
    name: "",
    active: false,
    days: 365,
    sigmas: 5,
    detector: null,
    showSetAside: false,
    showRejected: true,
    planeFit: "nature",
    data: null,
    key: null,
    requestKey: null,
    error: null,
    selected: null,
    domain: null,
  };
  let chart = null;

  // -- loading --------------------------------------------------------------

  const requestKey = () => `${view.norad}|${view.days}|${view.sigmas}`;

  async function load() {
    if (!view.norad) return;
    const key = requestKey();
    if (view.key === key && view.data) {
      render();
      return;
    }
    if (view.requestKey === key) return;
    view.requestKey = key;
    view.error = null;
    $("bh-loading").classList.remove("hidden");
    $("bh-message").hidden = true;
    renderPanel();
    try {
      const data = await getJSON("/api/behaviour", { norad: view.norad, days: view.days, sigmas: view.sigmas });
      if (view.requestKey !== key) return;
      if (view.key?.split("|")[0] !== String(view.norad)) view.selected = null;
      view.data = data;
      view.key = key;
      if (view.selected) {
        view.selected = data.events.find((e) => eventKey(e) === eventKey(view.selected)) ?? null;
      }
    } catch (error) {
      if (view.requestKey !== key) return;
      view.data = null;
      view.key = key;
      view.error = error.message;
    } finally {
      if (view.requestKey === key) {
        view.requestKey = null;
        $("bh-loading").classList.add("hidden");
      }
    }
    render();
  }

  // -- rendering ------------------------------------------------------------

  function render() {
    renderToolbar();
    renderMessage();
    renderPanel();
    build();
  }

  function renderToolbar() {
    const d = view.data;
    $("bh-name").textContent = view.name || (view.norad ? `NORAD ${view.norad}` : "--");
    if (!d) {
      $("bh-sub").textContent = view.norad ? `NORAD ${view.norad}` : "";
      return;
    }
    const regime = d.regime === "geosynchronous" ? "GEOSYNCHRONOUS" : "NOT GEOSYNCHRONOUS";
    $("bh-sub").textContent =
      `NORAD ${d.norad_id} · ${regime} · ${fmtDate(d.start_unix_ms)} → ${fmtDate(d.stop_unix_ms)} · ${d.quality.used} ELEMENT SETS`;
    const geo = d.regime === "geosynchronous";
    const detector = view.detector ?? (geo ? "out_of_plane" : "in_plane");
    for (const button of document.querySelectorAll("#bh-detector button")) {
      button.textContent = axisName(button.dataset.kind, geo);
      button.classList.toggle("on", button.dataset.kind === detector);
    }
  }

  function renderMessage() {
    const box = $("bh-message");
    box.hidden = !view.error;
    if (!view.error) return;
    box.replaceChildren(el("div", "bh-message-title", "NO HISTORY FOR THIS OBJECT"), el("p", "", view.error));
    if (/credentials/i.test(view.error)) {
      box.append(
        el(
          "p",
          "bh-message-hint",
          "The behaviour view reads element set histories from Space-Track. Start the tracker from the orbwatch folder that holds your .env file, or pass --env-file.",
        ),
      );
    }
  }

  // -- charts ---------------------------------------------------------------

  function panelSpecs(d) {
    const geo = d.regime === "geosynchronous";
    const s = d.series;
    const detector = view.detector ?? (geo ? "out_of_plane" : "in_plane");
    const det = d.detector[detector];
    return [
      {
        id: "a",
        kind: "in_plane",
        title: "SEMI-MAJOR AXIS",
        unit: "KM",
        lines: [{ values: s.a_km, color: C.cyan, label: "a", format: (v) => `${d3.format(",.3f")(v)} km` }],
      },
      geo
        ? {
            id: "o",
            kind: "out_of_plane",
            title: "INCLINATION VECTOR",
            unit: "DEG",
            lines: [
              { values: s.ivec_x_deg, color: C.cyan, label: "i sin Ω", format: (v) => `${v.toFixed(4)}°` },
              { values: s.ivec_y_deg, color: C.amber, label: "i cos Ω", format: (v) => `${v.toFixed(4)}°` },
            ],
          }
        : {
            id: "o",
            kind: "out_of_plane",
            title: "INCLINATION",
            unit: "DEG",
            lines: [{ values: s.inclination_deg, color: C.cyan, label: "i", format: (v) => `${v.toFixed(4)}°` }],
          },
      {
        id: "d",
        kind: detector,
        detector: det,
        title: `DETECTOR · ${axisName(detector, geo)}`,
        unit: `UNEXPLAINED CHANGE PER GAP · ${det.unit === "m" ? "M" : "ARCSEC"} · LOG`,
      },
    ];
  }

  function build() {
    svg.selectAll("*").remove();
    chart = null;
    hideHover();
    const d = view.data;
    if (!d || !view.active) return;
    const width = charts.clientWidth;
    const height = charts.clientHeight;
    if (width < 260 || height < 260) return;
    svg.attr("width", width).attr("height", height);

    const plotW = width - MARGIN.left - MARGIN.right;
    const avail = height - MARGIN.top - MARGIN.bottom - 3 * HEADER - 2 * GAP;
    const pad = (d.stop_unix_ms - d.start_unix_ms) * 0.01;
    const x0 = d3.scaleUtc().domain([d.start_unix_ms - pad, d.stop_unix_ms + pad]).range([0, plotW]);

    const defs = svg.append("defs");
    const root = svg.append("g").attr("transform", `translate(${MARGIN.left},0)`);

    let y = MARGIN.top;
    const panels = panelSpecs(d).map((spec, index) => {
      const top = y + HEADER;
      const h = avail * SHARES[index];
      y = top + h + GAP;

      const g = root.append("g").attr("transform", `translate(0,${top})`);
      const clipId = `bh-clip-${spec.id}`;
      defs.append("clipPath").attr("id", clipId).append("rect").attr("width", plotW).attr("height", h);
      g.append("rect").attr("class", "bh-frame").attr("width", plotW).attr("height", h);

      const title = g.append("text").attr("class", "bh-title").attr("y", -8);
      title.append("tspan").text(spec.title);
      title.append("tspan").attr("class", "bh-unit").attr("dx", 12).text(spec.unit);
      if (spec.lines && spec.lines.length > 1) {
        const legend = g.append("g").attr("class", "bh-legend-inline").attr("transform", `translate(${plotW},-8)`);
        let offset = 0;
        for (const line of [...spec.lines].reverse()) {
          const text = legend.append("text").attr("x", -offset).attr("text-anchor", "end").text(line.label);
          const w = text.node().getComputedTextLength();
          legend.append("line").attr("x1", -offset - w - 20).attr("x2", -offset - w - 6).attr("y1", -4).attr("y2", -4).attr("stroke", line.color);
          offset += w + 34;
        }
      }

      const panel = {
        ...spec,
        top,
        h,
        vgrid: g.append("g").attr("class", "bh-grid"),
        hgrid: g.append("g").attr("class", "bh-grid"),
        yAxis: g.append("g").attr("class", "bh-axis"),
      };
      const body = g.append("g").attr("clip-path", `url(#${clipId})`);
      panel.bands = body.append("g");
      panel.data = body.append("g");
      panel.rug = body.append("g");
      panel.cross = body.append("g").attr("class", "bh-cross").style("display", "none");
      panel.cross.append("line").attr("y1", 0).attr("y2", h);

      if (spec.detector) {
        const Td = spec.detector.t_unix_ms;
        const shown = d.events.filter((e) => isShown(e) && e.kind === spec.kind);
        panel.counted = spec.detector.flagged.map((flagged, j) => {
          if (!flagged) return null;
          const hit = shown.find((e) => e.start_unix_ms <= Td[j] && Td[j] <= e.end_unix_ms);
          return hit ? hit.status : null;
        });
        panel.y = d3.scaleLog().range([h - 5, 5]).clamp(true);
        panel.threshold = panel.data.append("line").attr("class", "bh-threshold").attr("x1", 0).attr("x2", plotW);
        panel.edgeTicks = panel.data.append("g");
        panel.points = panel.data.append("g");
        panel.thresholdLabel = g.append("text").attr("class", "bh-threshold-label").attr("x", plotW - 6).attr("text-anchor", "end");
        panel.marker = panel.cross.append("circle").attr("r", 4).attr("class", "bh-marker");
      } else {
        panel.y = d3.scaleLinear().range([h - 8, 8]);
        panel.paths = spec.lines.map((line) => panel.data.append("path").attr("class", "bh-line").attr("stroke", line.color));
        panel.samples = spec.lines.map((line) => panel.data.append("g").attr("fill", line.color));
        panel.dots = spec.lines.map((line) => panel.cross.append("circle").attr("r", 3.2).attr("fill", line.color));
      }
      return panel;
    });

    const bottom = y - GAP;
    const xAxis = root.append("g").attr("class", "bh-axis bh-xaxis").attr("transform", `translate(0,${bottom})`);
    const overlay = root
      .append("rect")
      .attr("class", "bh-overlay")
      .attr("y", MARGIN.top)
      .attr("width", plotW)
      .attr("height", bottom - MARGIN.top);

    const extent = [[0, MARGIN.top], [plotW, bottom]];
    const zoom = d3
      .zoom()
      .scaleExtent([1, 2000])
      .extent(extent)
      .translateExtent(extent)
      .on("zoom", (event) => {
        chart.xz = event.transform.rescaleX(x0);
        view.domain = event.transform.k > 1.0001 ? chart.xz.domain().map(Number) : null;
        redraw();
        if (event.sourceEvent) hover(event.sourceEvent);
      });
    overlay
      .call(zoom)
      .on("dblclick.zoom", null)
      .on("dblclick", resetZoom)
      .on("mousemove", hover)
      .on("mouseleave", hideHover)
      .on("click", () => {
        if (chart?.hovered) select(chart.hovered, { zoom: true });
      });

    chart = { plotW, panels, x0, xz: x0, xAxis, overlay, zoom, bottom, hovered: null };
    if (view.domain) overlay.call(zoom.transform, transformFor(view.domain));
    else redraw();
  }

  function transformFor([t0, t1]) {
    const a = chart.x0(t0);
    const b = chart.x0(t1);
    const k = Math.max(1, Math.min(2000, chart.plotW / Math.max(1e-6, b - a)));
    return d3.zoomIdentity.scale(k).translate(-a, 0);
  }

  function redraw() {
    const d = view.data;
    const { xz, panels, plotW } = chart;
    const [t0, t1] = xz.domain().map(Number);
    const T = d.series.t_unix_ms;
    const i0 = Math.max(0, d3.bisectLeft(T, t0) - 1);
    const i1 = Math.min(T.length, d3.bisectRight(T, t1) + 1);
    const xTicks = xz.ticks(Math.max(3, Math.floor(plotW / 110)));
    chart.xAxis.call(d3.axisBottom(xz).tickValues(xTicks).tickFormat(tickFormat).tickSizeOuter(0));

    for (const p of panels) {
      p.vgrid
        .selectAll("line")
        .data(xTicks)
        .join("line")
        .attr("x1", (t) => xz(t))
        .attr("x2", (t) => xz(t))
        .attr("y1", 0)
        .attr("y2", p.h);
      if (p.detector) drawDetector(p, t0, t1);
      else drawLines(p, T, i0, i1);
      drawBands(p);
    }
  }

  function drawLines(p, T, i0, i1) {
    const { xz, plotW } = chart;
    const idx = d3.range(i0, i1);
    let lo = Infinity;
    let hi = -Infinity;
    for (const line of p.lines) {
      for (const i of idx) {
        const v = line.values[i];
        if (v === null) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    const span = hi - lo || Math.abs(hi) * 1e-6 || 1e-6;
    p.y.domain([lo - span * 0.08, hi + span * 0.08]);

    const count = Math.max(2, Math.floor(p.h / 40));
    const ticks = p.y.ticks(count);
    p.yAxis.call(d3.axisLeft(p.y).tickValues(ticks).tickFormat(p.y.tickFormat(count)).tickSizeOuter(0));
    p.hgrid
      .selectAll("line")
      .data(ticks)
      .join("line")
      .attr("x1", 0)
      .attr("x2", plotW)
      .attr("y1", (v) => p.y(v))
      .attr("y2", (v) => p.y(v));

    const showSamples = idx.length < plotW / 7;
    p.lines.forEach((line, k) => {
      const path = d3
        .line()
        .defined((i) => line.values[i] !== null)
        .x((i) => xz(T[i]))
        .y((i) => p.y(line.values[i]));
      p.paths[k].attr("d", path(idx));
      p.samples[k]
        .selectAll("circle")
        .data(showSamples ? idx.filter((i) => line.values[i] !== null) : [])
        .join("circle")
        .attr("r", 1.9)
        .attr("cx", (i) => xz(T[i]))
        .attr("cy", (i) => p.y(line.values[i]));
    });

    p.rug
      .selectAll("line")
      .data(view.showRejected ? view.data.rejected_unix_ms : [])
      .join("line")
      .attr("class", "bh-rug")
      .attr("x1", (t) => xz(t))
      .attr("x2", (t) => xz(t))
      .attr("y1", 0)
      .attr("y2", 8);
  }

  function drawDetector(p, t0, t1) {
    const { xz, plotW } = chart;
    const det = p.detector;
    const Td = det.t_unix_ms;
    const j0 = Math.max(0, d3.bisectLeft(Td, t0) - 1);
    const j1 = Math.min(Td.length, d3.bisectRight(Td, t1) + 1);
    const idx = d3.range(j0, j1).filter((j) => det.magnitude[j] !== null);

    const positives = idx.map((j) => det.magnitude[j]).filter((v) => v > 0).sort(d3.ascending);
    const hi = Math.max(det.threshold, positives.length ? positives[positives.length - 1] : det.threshold);
    const lo = positives.length
      ? Math.max(d3.quantile(positives, 0.02), det.threshold / 1000)
      : det.threshold / 100;
    p.y.domain([lo / 1.5, hi * 1.8]);
    p.floor = lo / 1.5;

    const [a, b] = p.y.domain();
    let ticks = [];
    for (let e = Math.ceil(Math.log10(a)); e <= Math.floor(Math.log10(b)); e += 1) ticks.push(10 ** e);
    if (ticks.length < 2) ticks = p.y.ticks(3);
    p.yAxis.call(d3.axisLeft(p.y).tickValues(ticks).tickFormat(fmtMagnitude).tickSizeOuter(0));
    p.hgrid
      .selectAll("line")
      .data(ticks)
      .join("line")
      .attr("x1", 0)
      .attr("x2", plotW)
      .attr("y1", (v) => p.y(v))
      .attr("y2", (v) => p.y(v));

    p.points
      .selectAll("circle")
      .data(idx, (j) => j)
      .join("circle")
      .attr("class", (j) => {
        if (det.flagged[j]) {
          if (p.counted[j] === "drag_surge") return "bh-pt flagged drag";
          return p.counted[j] ? "bh-pt flagged" : "bh-pt flagged aside";
        }
        return det.edge_threshold[j] !== null ? "bh-pt edge" : "bh-pt";
      })
      .attr("r", (j) => (det.flagged[j] || det.edge_threshold[j] !== null ? 2.8 : 1.6))
      .attr("cx", (j) => xz(Td[j]))
      .attr("cy", (j) => p.y(Math.max(det.magnitude[j], p.floor)));

    p.edgeTicks
      .selectAll("line")
      .data(idx.filter((j) => det.edge_threshold[j] !== null), (j) => j)
      .join("line")
      .attr("class", "bh-edge-threshold")
      .attr("x1", (j) => xz(Td[j]) - 7)
      .attr("x2", (j) => xz(Td[j]) + 7)
      .attr("y1", (j) => p.y(det.edge_threshold[j]))
      .attr("y2", (j) => p.y(det.edge_threshold[j]));

    const yT = p.y(det.threshold);
    p.threshold.attr("y1", yT).attr("y2", yT);
    const unit = det.unit === "m" ? " m" : "″";
    p.thresholdLabel
      .attr("y", yT - 5)
      .text(`${view.data.sigmas}σ THRESHOLD ${d3.format(",.3~r")(det.threshold)}${unit}`);
  }

  function visibleEvents(kind) {
    return view.data.events.filter((e) => e.kind === kind && (isShown(e) || view.showSetAside));
  }

  function bandX(event) {
    const { xz } = chart;
    const a = xz(event.start_unix_ms);
    const b = xz(event.end_unix_ms);
    return b - a >= 2 ? [a, b - a] : [(a + b) / 2 - 1, 2];
  }

  function drawBands(p) {
    const selectedKey = view.selected ? eventKey(view.selected) : null;
    p.bands
      .selectAll("rect")
      .data(visibleEvents(p.kind), eventKey)
      .join("rect")
      .attr("class", (e) => `bh-band ${bandClass(e)}${eventKey(e) === selectedKey ? " selected" : ""}`)
      .attr("x", (e) => bandX(e)[0])
      .attr("width", (e) => bandX(e)[1])
      .attr("y", 0)
      .attr("height", p.h);
  }

  // -- hover, selection and zoom -------------------------------------------

  function hover(event) {
    if (!chart || !view.data) return;
    const [mx, my] = d3.pointer(event, chart.overlay.node());
    const { xz, panels } = chart;
    const d = view.data;
    const T = d.series.t_unix_ms;
    const t = +xz.invert(mx);
    let i = d3.bisectLeft(T, t);
    if (i >= T.length || (i > 0 && t - T[i - 1] < T[i] - t)) i -= 1;
    const panel = panels.find((p) => my >= p.top - HEADER && my <= p.top + p.h) ?? null;

    let gap = null;
    for (const p of panels) {
      p.cross.style("display", null);
      p.cross.select("line").attr("x1", mx).attr("x2", mx);
      if (p.detector) {
        const Td = p.detector.t_unix_ms;
        let j = d3.bisectLeft(Td, t);
        if (j >= Td.length || (j > 0 && t - Td[j - 1] < Td[j] - t)) j -= 1;
        const v = p.detector.magnitude[j];
        if (j >= 0 && v !== null) {
          p.marker.attr("cx", xz(Td[j])).attr("cy", p.y(Math.max(v, p.floor)));
          if (panel === p) {
            const edge = p.detector.edge_threshold[j];
            gap = {
              value: v,
              flagged: p.detector.flagged[j],
              counted: p.counted[j],
              det: p.detector,
              threshold: edge ?? p.detector.threshold,
              edge: edge !== null,
            };
          }
        }
      } else {
        p.lines.forEach((line, k) => {
          const v = line.values[i];
          p.dots[k].attr("cx", xz(T[i])).attr("cy", v === null ? -20 : p.y(v));
        });
      }
    }

    const hits = [];
    for (const kind of panel ? [panel.kind] : ["in_plane", "out_of_plane"]) {
      for (const e of visibleEvents(kind)) {
        const [x, w] = bandX(e);
        if (mx >= x - 4 && mx <= x + w + 4) hits.push(e);
      }
    }
    hits.sort((a, b) => isShown(a) - isShown(b) || a.delta_v_m_s - b.delta_v_m_s).reverse();
    chart.hovered = hits[0] ?? null;
    chart.overlay.classed("pointing", Boolean(chart.hovered));
    renderTooltip(mx, my, i, gap, hits.slice(0, 2));
  }

  function renderTooltip(mx, my, i, gap, hits) {
    const d = view.data;
    const T = d.series.t_unix_ms;
    tooltip.replaceChildren();
    tooltip.append(el("div", "tt-time", `ELEMENT SET ${fmtStamp(T[i])} UTC`));
    for (const p of chart.panels) {
      if (p.detector) continue;
      for (const line of p.lines) {
        const v = line.values[i];
        tooltip.append(row(line.label, v === null ? "--" : line.format(v)));
      }
    }
    if (gap) {
      const unit = gap.det.unit === "m" ? " m" : "″";
      const fmt = d3.format(",.3~r");
      tooltip.append(
        row("GAP", `${fmt(gap.value)}${unit} vs ${fmt(gap.threshold)}${unit}`, gap.flagged ? "flag" : "dim"),
      );
      if (gap.flagged) {
        tooltip.append(
          el(
            "div",
            "tt-note",
            gap.counted === "manoeuvre"
              ? "Flagged, and part of a counted manoeuvre."
              : gap.counted === "drag_surge"
                ? "Flagged, and part of a drag surge: natural, not a burn."
                : "Flagged, then set aside as below floor, incoherent or an excursion. Tick SET ASIDE to see why.",
          ),
        );
      }
      if (gap.edge) {
        tooltip.append(
          el("div", "tt-note", "Edge gap: no past to estimate the natural rate from, so the threshold is raised by what not knowing it could produce."),
        );
      }
    }
    const geo = d.regime === "geosynchronous";
    for (const e of hits) {
      const box = el("div", `tt-event ${bandClass(e)}`);
      box.append(el("div", "tt-head", `${axisName(e.kind, geo)} · ${statusText(e)}`));
      box.append(row("CHANGE", changeText(e)));
      box.append(row("DELTA-V", `${fmtSpeed(e.delta_v_m_s)} m/s`));
      box.append(row("PEAK", `${Math.round(e.significance)}σ · ${e.gaps} GAP${e.gaps > 1 ? "S" : ""}`));
      box.append(row("WINDOW", `${fmtDay(e.start_unix_ms)} ${fmtClock(e.start_unix_ms)} → ${fmtDay(e.end_unix_ms)} ${fmtClock(e.end_unix_ms)}`));
      box.append(el("div", "tt-note", explain(e, d)));
      tooltip.append(box);
    }
    if (hits.length) tooltip.append(el("div", "tt-hint", "CLICK TO ZOOM"));

    tooltip.hidden = false;
    const width = charts.clientWidth;
    const height = charts.clientHeight;
    const tw = tooltip.offsetWidth;
    const th = tooltip.offsetHeight;
    let left = MARGIN.left + mx + 18;
    if (left + tw > width - 8) left = MARGIN.left + mx - tw - 18;
    const top = Math.max(6, Math.min(height - th - 6, my - th / 3));
    tooltip.style.left = `${Math.max(6, left)}px`;
    tooltip.style.top = `${top}px`;
  }

  function hideHover() {
    tooltip.hidden = true;
    if (!chart) return;
    chart.hovered = null;
    chart.overlay.classed("pointing", false);
    for (const p of chart.panels) p.cross.style("display", "none");
  }

  function applyDomain(domain) {
    if (!chart) {
      view.domain = domain;
      return;
    }
    chart.overlay.transition().duration(650).call(chart.zoom.transform, transformFor(domain));
  }

  function zoomToEvent(event) {
    const width = event.end_unix_ms - event.start_unix_ms;
    const pad = Math.max(5 * DAY_MS, 3 * width);
    applyDomain([event.start_unix_ms - pad, event.end_unix_ms + pad]);
  }

  function resetZoom() {
    view.domain = null;
    if (chart) chart.overlay.transition().duration(500).call(chart.zoom.transform, d3.zoomIdentity);
  }

  function select(event, { zoom = false } = {}) {
    view.selected = event;
    if (!isShown(event) && !view.showSetAside) {
      view.showSetAside = true;
      $("bh-setaside").checked = true;
    }
    if (chart) for (const p of chart.panels) drawBands(p);
    highlightRows();
    if (zoom) zoomToEvent(event);
  }

  // -- side panel -----------------------------------------------------------

  function renderPanel() {
    const d = view.data;
    const status = $("ev-status");
    const content = $("ev-content");
    content.hidden = !d;
    status.hidden = Boolean(d);
    if (!d) {
      if (view.requestKey) status.textContent = "Analysing the element set history. The first load of an object downloads it from Space-Track, which takes a few seconds.";
      else if (view.error) status.textContent = view.error;
      else status.textContent = "Open the BEHAVIOUR view (B) to analyse this object's element set history.";
      return;
    }
    const geo = d.regime === "geosynchronous";
    $("ev-span").textContent = `${fmtDate(d.start_unix_ms)} → ${fmtDate(d.stop_unix_ms)}`;

    const q = d.quality;
    const inPlane = d.detector.in_plane;
    const outOfPlane = d.detector.out_of_plane;
    $("ev-quality").replaceChildren(
      row("ELEMENT SETS", String(q.records_in)),
      row("RE-FITS MERGED", String(q.refits_merged), "dim"),
      row("REJECTED", `${q.transients_rejected} TRANSIENT`, q.transients_rejected ? "warn" : "dim"),
      row("USED", String(q.used), "good"),
      row("SCATTER PER GAP", `${d3.format(",.3~r")(inPlane.sigma)} m · ${d3.format(".3~r")(outOfPlane.sigma)}″`),
    );

    renderBudgets(d, geo);
    renderPlane(d);
    renderTable(d, geo);
    renderSurges(d);

    const count = (status) => d.events.filter((e) => e.status === status).length;
    const floors = d.floors_m_s;
    $("ev-aside").replaceChildren(
      row("BELOW FLOOR", `${count("below_floor")}  (< ${floors.in_plane} / ${floors.out_of_plane} m/s)`, "dim"),
      row("INCOHERENT", String(count("incoherent")), "dim"),
      row("EXCURSIONS", String(count("excursion")), "dim"),
    );
  }

  function renderBudgets(d, geo) {
    const box = $("ev-budgets");
    box.replaceChildren();
    const span = Math.round(d.span_days);
    for (const kind of ["in_plane", "out_of_plane"]) {
      const b = d.budgets[kind];
      const free = kind === "out_of_plane" && d.natural?.control === "free";
      const required = !free && b.required_m_s !== null && b.required_m_s > 0;
      const scatter = b.scatter_per_gap_m_s;

      const card = el("div", "budget");
      const head = el("div", "budget-head");
      head.append(el("span", "", axisName(kind, geo)));
      let total = `${fmtSpeed(b.detected_m_s)} m/s SEEN`;
      if (free) total = "NOT HELD";
      else if (required) total = `≥ ${fmtSpeed(b.required_m_s)} m/s`;
      head.append(el("span", `budget-total${required ? "" : " dim"}`, total));
      card.append(head);

      if (free) {
        const n = d.natural;
        card.append(
          el(
            "div",
            "budget-line",
            `The orbit plane moved ${n.observed_displacement_deg.toFixed(3)}° where uncontrolled drift predicts ${n.natural_displacement_deg.toFixed(3)}°: nobody is holding its inclination.`,
          ),
          el(
            "div",
            "budget-method",
            `The ${fmtSpeed(b.required_m_s)} m/s left over is within the drift model's 10% accuracy, so no north-south propellant is implied.`,
          ),
        );
        box.append(card);
        continue;
      }

      if (required) {
        const seen = Math.min(1, b.closure);
        const bar = el("div", "budget-bar");
        const fill = el("i", seen >= 0.8 ? "good" : "warn");
        fill.style.width = `${100 * seen}%`;
        bar.append(fill);
        card.append(
          bar,
          el("div", "budget-line", `Spent over ${span} days ${PURPOSE[kind]}. A minimum set by physics, whether or not the burns are visible.`),
          el("div", "budget-seen", `SEEN AS INDIVIDUAL BURNS ${fmtSpeed(b.detected_m_s)} m/s · ${Math.round(100 * b.closure)}%`),
        );
      } else {
        card.append(
          el("div", "budget-seen", `SEEN AS INDIVIDUAL BURNS ${fmtSpeed(b.detected_m_s)} m/s`),
          el("div", "budget-line", noRequirement(kind, geo, b)),
        );
      }

      if (b.typical_burn_m_s !== null && scatter > 0) {
        const plural = b.burns === 1 ? "burn" : "burns";
        card.append(
          el(
            "div",
            "budget-line",
            `${b.burns} ${plural} resolved, typically ${fmtSig(b.typical_burn_m_s)} m/s: ${fmtSig(b.typical_burn_m_s / scatter)}× the ${fmtSig(scatter)} m/s scatter between catalogue updates.`,
          ),
        );
      }
      if (required && b.closure < 0.8 && b.required_per_gap_m_s !== null) {
        card.append(
          el(
            "div",
            "budget-line budget-warn",
            `The rest came in corrections the size of that scatter, ${fmtSig(b.required_per_gap_m_s)} m/s per update on average against ${fmtSig(scatter)} m/s. The total is measured; the individual corrections cannot be resolved from public element sets.`,
          ),
        );
      }
      if (required) card.append(el("div", "budget-method", `Method: ${b.method}.`));
      box.append(card);
    }
  }

  function renderPlane(d) {
    const section = $("ev-plane-section");
    section.hidden = !d.natural;
    if (!d.natural) return;
    for (const button of document.querySelectorAll("#ev-plane-fit button")) {
      button.classList.toggle("on", button.dataset.fit === view.planeFit);
    }

    const W = 400;
    const H = 300;
    const m = { l: 52, r: 14, t: 12, b: 34 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;
    const s = d.series;
    const n = d.natural;
    const observed = s.ivec_x_deg.map((x, k) => [x, s.ivec_y_deg[k]]).filter(([x, y]) => x !== null && y !== null);
    const natural = n.ivec_x_deg.map((x, k) => [x, n.ivec_y_deg[k]]);
    const fitted = view.planeFit === "nature" ? observed.concat(natural) : observed;

    const [xa, xb] = d3.extent(fitted, (p) => p[0]);
    const [ya, yb] = d3.extent(fitted, (p) => p[1]);
    const perPx = Math.max((xb - xa) / pw, (yb - ya) / ph, 1e-6) * 1.15;
    const cx = (xa + xb) / 2;
    const cy = (ya + yb) / 2;
    const x = d3.scaleLinear().domain([cx - (perPx * pw) / 2, cx + (perPx * pw) / 2]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([cy - (perPx * ph) / 2, cy + (perPx * ph) / 2]).range([H - m.b, m.t]);

    const plot = d3.select("#ev-plane").selectAll("svg").data([0]).join("svg").attr("viewBox", `0 0 ${W} ${H}`);
    plot.selectAll("*").remove();
    const defs = plot.append("defs");
    defs
      .append("marker")
      .attr("id", "ev-arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 8)
      .attr("refY", 5)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M0,0 L10,5 L0,10 z")
      .attr("fill", C.amber);

    plot.append("rect").attr("class", "bh-frame").attr("x", m.l).attr("y", m.t).attr("width", pw).attr("height", ph);
    const xt = x.ticks(4);
    const yt = y.ticks(4);
    plot.append("g").attr("class", "bh-grid").selectAll("line").data(xt).join("line")
      .attr("x1", (v) => x(v)).attr("x2", (v) => x(v)).attr("y1", m.t).attr("y2", H - m.b);
    plot.append("g").attr("class", "bh-grid").selectAll("line").data(yt).join("line")
      .attr("x1", m.l).attr("x2", W - m.r).attr("y1", (v) => y(v)).attr("y2", (v) => y(v));
    plot.append("g").attr("class", "bh-axis").attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickValues(xt).tickFormat(x.tickFormat(4)).tickSizeOuter(0));
    plot.append("g").attr("class", "bh-axis").attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(y).tickValues(yt).tickFormat(y.tickFormat(4)).tickSizeOuter(0));
    plot.append("text").attr("class", "bh-axis-label").attr("x", W - m.r).attr("y", H - 4).attr("text-anchor", "end").text("i sin Ω, deg");
    plot.append("text").attr("class", "bh-axis-label").attr("transform", `translate(12,${m.t}) rotate(-90)`).attr("text-anchor", "end").text("i cos Ω, deg");

    defs.append("clipPath").attr("id", "ev-plane-clip").append("rect").attr("x", m.l).attr("y", m.t).attr("width", pw).attr("height", ph);
    const body = plot.append("g").attr("clip-path", "url(#ev-plane-clip)");
    const line = d3.line().x((p) => x(p[0])).y((p) => y(p[1]));
    body.append("path").attr("class", "ev-natural").attr("d", line(natural)).attr("marker-end", "url(#ev-arrow)");
    body.append("path").attr("class", "ev-observed").attr("d", line(observed));
    const start = observed[0];
    const end = observed[observed.length - 1];
    body.append("circle").attr("class", "ev-start").attr("r", 4).attr("cx", x(start[0])).attr("cy", y(start[1]));
    body.append("circle").attr("class", "ev-end").attr("r", 3.5).attr("cx", x(end[0])).attr("cy", y(end[1]));
    const tip = natural[natural.length - 1];
    if (x(tip[0]) > m.l && x(tip[0]) < W - m.r && y(tip[1]) > m.t && y(tip[1]) < H - m.b) {
      body.append("text").attr("class", "ev-natural-label").attr("x", x(tip[0])).attr("y", y(tip[1]) - 10)
        .attr("text-anchor", x(tip[0]) > W - 90 ? "end" : "middle").text("UNCONTROLLED");
    }

    const nat = n.natural_displacement_deg;
    const obs = n.observed_displacement_deg;
    const required = d.budgets.out_of_plane.required_m_s;
    let caption;
    if (n.control === "held") {
      caption = `Left alone, this orbit plane would have drifted ${nat.toFixed(3)}° along the dashed path. It moved ${obs.toFixed(3)}°, so someone is holding it, and holding it costs at least ${fmtSpeed(required)} m/s whether or not the individual corrections are visible.`;
    } else if (n.control === "free") {
      caption = `Left alone, this orbit plane would have drifted ${nat.toFixed(3)}°. It moved ${obs.toFixed(3)}°: it follows nature, and nobody is holding its inclination.`;
    } else {
      caption = `Left alone, this orbit plane would have drifted ${nat.toFixed(3)}°. It moved ${obs.toFixed(3)}°: partly held, or held for only part of the span.`;
    }
    $("ev-plane-caption").textContent = caption;
  }

  function renderTable(d, geo) {
    const body = $("ev-body");
    body.replaceChildren();
    const list = d.events.filter((e) => e.status === "manoeuvre");
    $("ev-count").textContent = list.length ? `${list.length} · CLICK TO ZOOM` : "";
    if (!list.length) {
      const tr = el("tr");
      const td = el("td", "empty", "No manoeuvre above the floor.");
      td.colSpan = 5;
      tr.append(td);
      body.append(tr);
      return;
    }
    for (const e of list) {
      const tr = el("tr");
      tr.dataset.key = eventKey(e);
      const date = el("td");
      date.append(el("i", `ev-dot ${e.profile}`), document.createTextNode(`${fmtDay(e.start_unix_ms)} ${String(new Date(e.start_unix_ms).getUTCFullYear()).slice(2)}`));
      tr.append(date, el("td", "", axisName(e.kind, geo, true)), el("td", "", changeText(e)), el("td", "", fmtSpeed(e.delta_v_m_s)), el("td", "", String(Math.round(e.significance))));
      tr.title = `${statusText(e)}. ${explain(e, d)}`;
      tr.addEventListener("click", () => {
        onShowRequested();
        select(e, { zoom: true });
      });
      body.append(tr);
    }
    highlightRows();
  }

  function renderSurges(d) {
    const list = d.events.filter((e) => e.status === "drag_surge");
    $("ev-surge-section").hidden = !list.length;
    const body = $("ev-surges");
    body.replaceChildren();
    for (const e of list) {
      const tr = el("tr");
      tr.dataset.key = eventKey(e);
      const date = el("td");
      date.append(el("i", "ev-dot drag"), document.createTextNode(`${fmtDay(e.start_unix_ms)} ${String(new Date(e.start_unix_ms).getUTCFullYear()).slice(2)}`));
      tr.append(date, el("td", "", (e.window_hours / 24).toFixed(1)), el("td", "", changeText(e)), el("td", "", String(Math.round(e.significance))));
      tr.title = explain(e, d);
      tr.addEventListener("click", () => {
        onShowRequested();
        select(e, { zoom: true });
      });
      body.append(tr);
    }
  }

  function highlightRows() {
    const key = view.selected ? eventKey(view.selected) : null;
    for (const tr of document.querySelectorAll("#ev-body tr, #ev-surges tr")) {
      tr.classList.toggle("current", tr.dataset.key === key);
    }
  }

  // -- controls -------------------------------------------------------------

  $("bh-days").addEventListener("change", (event) => {
    view.days = Number(event.target.value);
    view.domain = null;
    load();
  });
  $("bh-sigmas").addEventListener("change", (event) => {
    view.sigmas = Number(event.target.value);
    load();
  });
  $("bh-setaside").addEventListener("change", (event) => {
    view.showSetAside = event.target.checked;
    if (chart) redraw();
  });
  $("bh-rejected").addEventListener("change", (event) => {
    view.showRejected = event.target.checked;
    if (chart) redraw();
  });
  $("bh-reset").addEventListener("click", resetZoom);
  for (const button of document.querySelectorAll("#bh-detector button")) {
    button.addEventListener("click", () => {
      view.detector = button.dataset.kind;
      renderToolbar();
      build();
    });
  }
  for (const button of document.querySelectorAll("#ev-plane-fit button")) {
    button.addEventListener("click", () => {
      view.planeFit = button.dataset.fit;
      if (view.data) renderPlane(view.data);
    });
  }

  let resizeFrame = null;
  new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      if (view.active) build();
    });
  }).observe(charts);

  // -- public ---------------------------------------------------------------

  return {
    setObject(norad, name) {
      if (norad !== view.norad) {
        view.norad = norad;
        view.data = null;
        view.key = null;
        view.error = null;
        view.selected = null;
        view.domain = null;
        view.detector = null;
      }
      view.name = name;
      if (view.active) load();
      else {
        renderToolbar();
        renderPanel();
      }
    },
    activate() {
      view.active = true;
      load();
      build();
    },
    deactivate() {
      view.active = false;
      hideHover();
    },
  };
}
