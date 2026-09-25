/*
 * ORBWATCH mission view.
 *
 * Plans a rendezvous with the tracked object from a rideshare and shows it in
 * two modes. TRANSFER is the one trade that matters to get there: how fast
 * against how much delta-v. PROXIMITY is what happens once there: the approach,
 * the safety ellipse and the departure in the target's frame, what each burn
 * failing would do, and the Monte Carlo that sizes the proximity budget line.
 * The Python API computes the whole front, a full budget for every option on
 * it and the proximity plan, so moving the time budget, clicking a dot or
 * hovering a burn is instant; only a change of setting asks again.
 */

const d3 = window.d3;

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
const DAY_MS = 86400000;
const $ = (id) => document.getElementById(id);
const pad2 = (n) => String(Math.floor(n)).padStart(2, "0");
const fmtDate = (ms) => {
  const d = new Date(ms);
  return `${pad2(d.getUTCDate())} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
};
const fmtClock = (hours) => {
  const minutes = Math.round(hours * 60) % 1440;
  return `${pad2(minutes / 60)}:${pad2(minutes % 60)}`;
};
function fmtOffset(deg, sunSynchronous) {
  const sign = deg > 0 ? "+" : deg < 0 ? "−" : "";
  if (!sunSynchronous || deg === 0) return `${sign}${Math.abs(deg)}°`;
  const hours = Math.abs(deg) / 15;
  return `${sign}${Math.abs(deg)}° · ${Number.isInteger(hours) ? hours : hours.toFixed(2)} h ${deg < 0 ? "earlier" : "later"}`;
}
const fmtV = (v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

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

const DEFAULTS = {
  altitude: 525, node_offset: -15, dry: 150, isp: 220, inj_alt: 10, inj_inc: 0.1, ops: 90, disposal: 300,
  ellipse: 250, keep_out: 200, orbits: 5, errors: 1,
};
const KEEP_OUT_GAP_M = 25;
const fmtKm = (m) => (m >= 1000 ? `${(m / 1000).toLocaleString("en", { maximumFractionDigits: 1 })} km` : `${Math.round(m)} m`);
const fmtErrors = (v) => `×${v} ${v === 0 ? "none" : v < 1 ? "better" : v === 1 ? "typical" : "worse"}`;
const fmtPercent = (f) => (f === 0 ? "0%" : f < 0.001 ? "<0.1%" : `${(100 * f).toFixed(1)}%`);

export function createMissionView({ getJSON, onShowRequested }) {
  const charts = $("ms-charts");
  const tooltip = $("ms-tooltip");
  const svg = d3.select(charts).append("svg").attr("class", "bh-svg");

  const view = {
    norad: null,
    name: "",
    active: false,
    params: { ...DEFAULTS },
    budgetDays: null,
    data: null,
    key: null,
    requestKey: null,
    error: null,
    selected: 0,
    overBudget: false,
    mode: "transfer",
    zoom: d3.zoomIdentity,
    hoverBurn: null,
    pinnedBurn: null,
  };
  let proximityChart = null;

  const requestKey = () => `${view.norad}|${Object.values(view.params).join("|")}`;

  async function load() {
    if (!view.norad || !view.active) return;
    const key = requestKey();
    if (view.key === key && view.data) {
      render();
      return;
    }
    if (view.requestKey === key) return;
    view.requestKey = key;
    view.error = null;
    $("ms-loading").classList.remove("hidden");
    try {
      const data = await getJSON("/api/mission", { norad: view.norad, ...view.params });
      if (view.requestKey !== key) return;
      view.data = data;
      view.key = key;
      view.pinnedBurns = null;
      view.hoverBurns = null;
      if (view.budgetDays === null) view.budgetDays = data.default_budget_days;
    } catch (error) {
      if (view.requestKey !== key) return;
      view.data = null;
      view.key = key;
      view.error = error.message;
    } finally {
      if (view.requestKey === key) {
        view.requestKey = null;
        $("ms-loading").classList.add("hidden");
      }
    }
    chooseByBudget();
    render();
  }

  function chooseByBudget() {
    const options = view.data?.options ?? [];
    let best = -1;
    options.forEach((o, i) => {
      if (o.total_days <= view.budgetDays && (best < 0 || o.transfer_m_s < options[best].transfer_m_s)) best = i;
    });
    view.overBudget = best < 0;
    view.selected = best < 0 ? 0 : best;
  }

  // -- rendering ------------------------------------------------------------

  function render() {
    renderToolbar();
    renderCharts();
    renderPanel();
  }

  function renderToolbar() {
    $("ms-name").textContent = view.name || (view.norad ? `NORAD ${view.norad}` : "--");
    const d = view.data;
    const proximity = view.mode === "proximity";
    for (const button of $("ms-mode").querySelectorAll("button")) button.classList.toggle("on", button.dataset.mode === view.mode);
    for (const [id, show] of [
      ["ms-controls-transfer", !proximity], ["ms-legend-transfer", !proximity], ["ms-hint-transfer", !proximity],
      ["ms-controls-proximity", proximity], ["ms-legend-proximity", proximity], ["ms-hint-proximity", proximity],
    ]) $(id).hidden = !show;
    const sunSynchronous = d?.dropoff.kind === "sun-synchronous";
    const rideshare = !d ? "A RIDESHARE" : sunSynchronous ? "A SUN-SYNCHRONOUS RIDESHARE" : "A RIDESHARE INTO ITS INCLINATION";
    if (proximity) {
      const p = d?.proximity;
      $("ms-sub").textContent = p
        ? `IN THE TARGET'S FRAME · ONE ORBIT ${p.period_min.toFixed(0)} MIN · CLOHESSY-WILTSHIRE, ${p.monte_carlo.runs.toLocaleString("en")} MONTE CARLO RUNS`
        : "IN THE TARGET'S FRAME";
    } else {
      $("ms-sub").textContent = d
        ? `RENDEZVOUS FROM ${rideshare} · TARGET ${d.target.altitude_km.toFixed(0)} KM, i ${d.target.inclination_deg.toFixed(2)}°`
        : "RENDEZVOUS FROM A RIDESHARE";
    }
    $("ms-ellipse").value = view.params.ellipse;
    $("ms-ellipse-v").textContent = `${view.params.ellipse} m`;
    $("ms-keepout").value = view.params.keep_out;
    $("ms-keepout-v").textContent = `${view.params.keep_out} m`;
    $("ms-orbits").value = view.params.orbits;
    $("ms-orbits-v").textContent = `${view.params.orbits} orbit${view.params.orbits === 1 ? "" : "s"}`;
    $("ms-errors").value = view.params.errors;
    $("ms-errors-v").textContent = fmtErrors(view.params.errors);
    $("ms-alt").value = view.params.altitude;
    $("ms-alt-v").textContent = `${view.params.altitude} km`;
    $("ms-offset").value = view.params.node_offset;
    $("ms-offset-v").textContent = fmtOffset(view.params.node_offset, sunSynchronous);
    if (view.budgetDays !== null) {
      $("ms-days").value = Math.round(view.budgetDays);
      $("ms-days-v").textContent = `${Math.round(view.budgetDays)} d`;
    }
    const message = $("ms-message");
    message.hidden = !view.error;
    if (view.error) {
      message.replaceChildren(el("div", "bh-message-title", "NO MISSION PLAN"), el("p", "", view.error));
    }
  }

  function renderCharts() {
    svg.selectAll("*").remove();
    tooltip.hidden = true;
    proximityChart = null;
    const d = view.data;
    if (!d || !view.active) return;
    const W = charts.clientWidth;
    const H = charts.clientHeight;
    if (W < 300 || H < 260) return;
    svg.attr("width", W).attr("height", H);
    if (view.mode === "proximity") renderProximity(W, H);
    else renderTransfer(W, H);
  }

  function renderTransfer(W, H) {
    const d = view.data;
    const m = { l: 70, r: 26, t: 30, b: 34 };
    const gapH = Math.max(110, (H - m.t - m.b) * 0.3);
    const tradeH = H - m.t - m.b - gapH - 48;
    const plotW = W - m.l - m.r;
    const options = d.options;
    const chosen = options[view.selected];

    // Trade: time against transfer delta-v.
    const maxDays = Math.max(d3.max(options, (o) => o.total_days), view.budgetDays ?? 0) * 1.04;
    const x = d3.scaleLinear().domain([0, maxDays]).range([m.l, m.l + plotW]);
    const vMax = d3.max(options, (o) => o.transfer_m_s);
    const span = Math.max(vMax - d.floor_m_s, 1);
    const y = d3.scaleLinear().domain([d.floor_m_s - 0.1 * span, vMax + 0.06 * span]).range([m.t + tradeH, m.t]);
    const g = svg.append("g");
    g.append("text").attr("class", "bh-title").attr("x", m.l).attr("y", m.t - 10).text("TIME AGAINST DELTA-V");
    g.append("text").attr("class", "bh-unit").attr("x", m.l + 190).attr("y", m.t - 10).text("TRANSFER, M/S · DAYS TO THE HOLD POINT");
    g.append("rect").attr("class", "bh-frame").attr("x", m.l).attr("y", m.t).attr("width", plotW).attr("height", tradeH);
    g.append("g").attr("class", "bh-grid").selectAll("line").data(y.ticks(5)).join("line")
      .attr("x1", m.l).attr("x2", m.l + plotW).attr("y1", (v) => y(v)).attr("y2", (v) => y(v));
    g.append("g").attr("class", "bh-grid").selectAll("line").data(x.ticks(8)).join("line")
      .attr("y1", m.t).attr("y2", m.t + tradeH).attr("x1", (v) => x(v)).attr("x2", (v) => x(v));
    g.append("g").attr("class", "bh-axis").attr("transform", `translate(0,${m.t + tradeH})`)
      .call(d3.axisBottom(x).ticks(8).tickSizeOuter(0));
    g.append("g").attr("class", "bh-axis").attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));

    const tilt = Math.abs(d.gap.inclination_deg);
    g.append("line").attr("class", "ms-floor").attr("x1", m.l).attr("x2", m.l + plotW)
      .attr("y1", y(d.floor_m_s)).attr("y2", y(d.floor_m_s));
    g.append("text").attr("class", "ms-label ms-label-floor").attr("x", m.l + plotW - 6).attr("y", y(d.floor_m_s) + 13)
      .attr("text-anchor", "end").text(`FLOOR ${d.floor_m_s.toFixed(0)} M/S · THE ALTITUDE CHANGE${tilt >= 0.005 ? ` AND THE ${tilt.toFixed(2)}° TILT` : ""}, WHATEVER THE WAIT`);
    g.append("text").attr("class", "ms-label ms-label-direct").attr("x", m.l + plotW - 6).attr("y", m.t + 16)
      .attr("text-anchor", "end").text(`↑ TURNING THE PLANE DIRECTLY: ${Math.round(d.direct_m_s).toLocaleString("en")} M/S`);

    if (view.budgetDays !== null) {
      const bx = x(view.budgetDays);
      g.append("line").attr("class", "ms-budget").attr("x1", bx).attr("x2", bx).attr("y1", m.t).attr("y2", m.t + tradeH);
      const right = bx > m.l + plotW - 110;
      g.append("text").attr("class", "ms-label ms-label-budget").attr("x", right ? bx - 5 : bx + 5).attr("y", m.t + 32)
        .attr("text-anchor", right ? "end" : "start").text(`BUDGET ${Math.round(view.budgetDays)} D`);
    }

    const line = d3.line().x((o) => x(o.total_days)).y((o) => y(o.transfer_m_s)).curve(d3.curveStepAfter);
    g.append("path").attr("class", "ms-front").attr("d", line(options));
    g.selectAll(".ms-dot").data(options).join("circle")
      .attr("class", (o, i) => `ms-dot${i === view.selected ? " chosen" : ""}${o.total_days > view.budgetDays ? " late" : ""}`)
      .attr("r", (o, i) => (i === view.selected ? 7 : 3.4))
      .attr("cx", (o) => x(o.total_days))
      .attr("cy", (o) => y(o.transfer_m_s))
      .on("mousemove", (event, o) => showTip(event, o))
      .on("mouseleave", () => (tooltip.hidden = true))
      .on("click", (event, o) => {
        view.selected = options.indexOf(o);
        view.budgetDays = Math.ceil(o.total_days);
        view.overBudget = false;
        render();
      });

    // Planes lining up: the node gap closing during the drift.
    const top = m.t + tradeH + 48;
    const xg = d3.scaleLinear().domain([0, chosen.total_days * 1.02]).range([m.l, m.l + plotW]);
    const end = d.gap.raan_deg + chosen.gap_rate_deg_per_day * chosen.wait_days;
    const [ga, gb] = d3.extent([d.gap.raan_deg, end, 0]);
    const pad = Math.max(1, (gb - ga) * 0.15);
    const yg = d3.scaleLinear().domain([ga - pad, gb + pad]).range([top + gapH, top]);
    const h = svg.append("g");
    h.append("text").attr("class", "bh-title").attr("x", m.l).attr("y", top - 10).text("PLANES LINING UP");
    h.append("text").attr("class", "bh-unit").attr("x", m.l + 152).attr("y", top - 10)
      .text(`NODE GAP, DEG · J2 CLOSES IT AT ${Math.abs(chosen.gap_rate_deg_per_day).toFixed(3)}°/DAY`);
    h.append("rect").attr("class", "bh-frame").attr("x", m.l).attr("y", top).attr("width", plotW).attr("height", gapH);
    h.append("g").attr("class", "bh-axis").attr("transform", `translate(0,${top + gapH})`)
      .call(d3.axisBottom(xg).ticks(8).tickSizeOuter(0));
    h.append("g").attr("class", "bh-axis").attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(yg).ticks(4).tickSizeOuter(0));
    h.append("line").attr("class", "ms-zero").attr("x1", m.l).attr("x2", m.l + plotW).attr("y1", yg(end)).attr("y2", yg(end));
    h.append("path").attr("class", "ms-gap").attr("d", d3.line()([
      [xg(0), yg(d.gap.raan_deg)], [xg(chosen.wait_days), yg(end)], [xg(chosen.total_days), yg(end)],
    ]));
    h.append("circle").attr("class", "ms-aligned").attr("r", 4).attr("cx", xg(chosen.wait_days)).attr("cy", yg(end));
    const nearRight = xg(chosen.wait_days) > m.l + plotW - 150;
    h.append("text").attr("class", "ms-label").attr("x", xg(chosen.wait_days) + (nearRight ? -8 : 8))
      .attr("y", yg(end) - 8).attr("text-anchor", nearRight ? "end" : "start")
      .text(`ALIGNED AFTER ${chosen.wait_days.toFixed(0)} D`);
  }

  function showTip(event, o) {
    const rect = charts.getBoundingClientRect();
    tooltip.replaceChildren(
      el("div", "tt-time", `${o.total_days.toFixed(0)} DAYS · ARRIVE ${fmtDate(o.arrival_unix_ms)}`),
      row("TRANSFER", `${o.transfer_m_s.toFixed(1)} m/s`),
      row("DRIFT ORBIT", `${o.altitude_km.toFixed(0)} km, i ${o.inclination_deg.toFixed(2)}°`),
      row("WAIT", `${o.wait_days.toFixed(0)} d`),
      row("PROPELLANT", `${o.propellant_kg.toFixed(1)} kg`),
      el("div", "tt-hint", "CLICK TO CHOOSE"),
    );
    tooltip.hidden = false;
    let left = event.clientX - rect.left + 16;
    if (left + tooltip.offsetWidth > rect.width - 8) left = event.clientX - rect.left - tooltip.offsetWidth - 16;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${Math.max(6, event.clientY - rect.top - 20)}px`;
  }

  // -- proximity ------------------------------------------------------------

  /** Planned burns numbered in order. Course corrections and trims on the
   *  ellipse are zero in the plan, so they get no number and no failure case. */
  function burnNumbers(p) {
    const numbers = new Map();
    p.burns.forEach((b, k) => {
      if (b.missed) numbers.set(k, numbers.size + 1);
    });
    return numbers;
  }

  function inspectionWindow(p) {
    const insert = p.burns.find((b) => b.kind === "insert");
    const leave = p.burns.find((b) => b.kind === "transfer" && b.dv_m_s > 0 && b.time_h > insert.time_h);
    return [insert.time_h, leave.time_h];
  }

  const shownBurns = () => new Set(view.hoverBurns ?? view.pinnedBurns ?? []);

  function renderProximity(W, H) {
    const p = view.data.proximity;
    const mc = p.monte_carlo;
    const m = { l: 70, r: 26, t: 30, b: 30 };
    const plotW = W - m.l - m.r;
    const lowH = Math.max(124, Math.min(250, (H - m.t - m.b) * 0.36));
    const topH = H - m.t - m.b - lowH - 56;
    const numbers = burnNumbers(p);
    const defs = svg.append("defs");

    // The whole sequence in the target's frame: in-track across, radial up, to scale.
    const g = svg.append("g");
    g.append("text").attr("class", "bh-title").attr("x", m.l).attr("y", m.t - 10).text("APPROACH, INSPECTION, DEPARTURE");
    g.append("text").attr("class", "bh-unit").attr("x", m.l + 300).attr("y", m.t - 10).text("IN-TRACK ACROSS, RADIAL UP, M · TO SCALE");
    g.append("rect").attr("class", "bh-frame").attr("x", m.l).attr("y", m.t).attr("width", plotW).attr("height", topH);
    defs.append("clipPath").attr("id", "ms-clip-top")
      .append("rect").attr("x", m.l).attr("y", m.t).attr("width", plotW).attr("height", topH);
    // Fit the plan and every failure case, so a missed brake's loop stays in view.
    const fitted = p.nominal.concat(...p.burns.filter((b) => b.missed).map((b) => b.missed.path));
    const [ya, yb] = d3.extent(fitted.map((q) => q[1]).concat([0]));
    const [xa, xb] = d3.extent(fitted.map((q) => q[0]).concat([0]));
    const mpp = Math.max((yb - ya) / plotW, (xb - xa) / topH) * 1.12;
    const yMid = (ya + yb) / 2;
    const xMid = (xa + xb) / 2;
    const X = d3.scaleLinear().domain([yMid - (mpp * plotW) / 2, yMid + (mpp * plotW) / 2]).range([m.l, m.l + plotW]);
    const Y = d3.scaleLinear().domain([xMid - (mpp * topH) / 2, xMid + (mpp * topH) / 2]).range([m.t + topH, m.t]);

    const gridX = g.append("g").attr("class", "bh-grid");
    const gridY = g.append("g").attr("class", "bh-grid");
    const xAxis = g.append("g").attr("class", "bh-axis").attr("transform", `translate(0,${m.t + topH})`);
    const yAxis = g.append("g").attr("class", "bh-axis").attr("transform", `translate(${m.l},0)`);
    const overlay = g.append("rect").attr("class", "ms-overlay")
      .attr("x", m.l).attr("y", m.t).attr("width", plotW).attr("height", topH);
    const body = g.append("g").attr("clip-path", "url(#ms-clip-top)");
    const keepOut = body.append("circle").attr("class", "ms-keepout");
    const runs = body.append("g").selectAll("path").data(mc.shown).join("path").attr("class", "ms-runs");
    const nominal = body.append("path").attr("class", "ms-nominal");
    const missed = body.append("g");
    const target = body.append("g").attr("class", "ms-target");
    target.append("path").attr("d", d3.symbol(d3.symbolDiamond, 64)());
    target.append("text").attr("x", 8).attr("y", -8).text("TARGET");

    const markers = [];
    for (const [k, n] of numbers) {
      const pos = p.burns[k].position_m;
      const same = markers.find((mk) => Math.hypot(mk.pos[0] - pos[0], mk.pos[1] - pos[1], mk.pos[2] - pos[2]) < 1);
      if (same) {
        same.burns.push(k);
        same.label += `·${n}`;
      } else markers.push({ pos, burns: [k], label: String(n) });
    }
    const marker = body.selectAll(".ms-burn").data(markers).join("g").attr("class", "ms-burn");
    const badge = (mk) => Math.max(14, 8 + 6.4 * mk.label.length);
    marker.append("rect")
      .attr("x", (mk) => -badge(mk) / 2).attr("width", badge)
      .attr("y", -7).attr("height", 14).attr("rx", 7);
    marker.append("text").attr("text-anchor", "middle").attr("dy", "0.35em").text((mk) => mk.label);
    marker
      .on("mousemove", (event, mk) => {
        if (view.hoverBurns?.join() !== mk.burns.join()) setHover(mk.burns);
        showBurnTip(event, mk.burns, numbers);
      })
      .on("mouseleave", () => {
        tooltip.hidden = true;
        setHover(null);
      })
      .on("click", (event, mk) => togglePin(mk.burns));

    g.append("text").attr("class", "ms-label").attr("x", m.l + 8).attr("y", m.t + 16).text("↑ AWAY FROM EARTH");
    g.append("text").attr("class", "ms-label").attr("x", m.l + plotW - 8).attr("y", m.t + topH - 8)
      .attr("text-anchor", "end").text("DIRECTION OF FLIGHT →");
    const zoomButton = g.append("text").attr("class", "ms-zoom-btn")
      .attr("x", m.l + plotW).attr("y", m.t - 10).attr("text-anchor", "end");

    function update() {
      const x = view.zoom.rescaleX(X);
      const y = view.zoom.rescaleY(Y);
      const xTicks = x.ticks(Math.max(4, Math.floor(plotW / 90)));
      const yTicks = y.ticks(Math.max(3, Math.floor(topH / 60)));
      xAxis.call(d3.axisBottom(x).tickValues(xTicks).tickFormat(d3.format(",")).tickSizeOuter(0));
      yAxis.call(d3.axisLeft(y).tickValues(yTicks).tickFormat(d3.format(",")).tickSizeOuter(0));
      gridX.selectAll("line").data(xTicks).join("line")
        .attr("x1", (v) => x(v)).attr("x2", (v) => x(v)).attr("y1", m.t).attr("y2", m.t + topH);
      gridY.selectAll("line").data(yTicks).join("line")
        .attr("y1", (v) => y(v)).attr("y2", (v) => y(v)).attr("x1", m.l).attr("x2", m.l + plotW);
      const line = d3.line().x((q) => x(q[1])).y((q) => y(q[0]));
      runs.attr("d", line);
      nominal.attr("d", line(p.nominal));
      keepOut.attr("cx", x(0)).attr("cy", y(0)).attr("r", Math.max(1.5, x(p.keep_out_m) - x(0)));
      target.attr("transform", `translate(${x(0)},${y(0)})`);
      const shown = shownBurns();
      marker.attr("transform", (mk) => `translate(${x(mk.pos[1])},${y(mk.pos[0])})`)
        .classed("on", (mk) => mk.burns.some((k) => shown.has(k)));
      const failed = [...shown].map((k) => p.burns[k]).filter((b) => b?.missed);
      const paths = missed.selectAll("g").data(failed).join((enter) => {
        const e = enter.append("g");
        e.append("path");
        e.append("circle").attr("r", 3.5);
        e.append("text");
        return e;
      });
      paths.attr("class", (b) => `ms-missed ${b.missed.passes ? "safe" : "unsafe"}`);
      paths.select("path").attr("d", (b) => line(b.missed.path));
      paths.each(function (b) {
        const path = b.missed.path;
        const i = d3.minIndex(path, (q) => Math.hypot(q[0], q[1], q[2]));
        const px = x(path[i][1]);
        const py = y(path[i][0]);
        d3.select(this).select("circle").attr("cx", px).attr("cy", py);
        d3.select(this).select("text").attr("x", px + 7).attr("y", py + 14).text(`${fmtKm(b.missed.closest_m)} CLOSEST`);
      });
      zoomButton.text(view.zoom.k > 1.01 ? "SHOW ALL ‹" : "ZOOM TO THE ELLIPSE ›");
    }

    const extent = [[m.l, m.t], [m.l + plotW, m.t + topH]];
    const zoom = d3.zoom().scaleExtent([1, 400]).extent(extent).translateExtent(extent)
      .on("zoom", (event) => {
        view.zoom = event.transform;
        update();
      });
    overlay.call(zoom).on("dblclick.zoom", null)
      .on("dblclick", () => overlay.transition().duration(450).call(zoom.transform, d3.zoomIdentity));
    zoomButton.on("click", () => {
      if (view.zoom.k > 1.01) {
        overlay.transition().duration(450).call(zoom.transform, d3.zoomIdentity);
        return;
      }
      const reach = Math.max(p.ellipse_radial_m, p.ellipse_cross_track_m, p.keep_out_m);
      const wanted = Math.max((2 * 1.5 * 2 * p.ellipse_radial_m) / plotW, (2 * 1.4 * reach) / topH);
      const k = Math.max(1, mpp / wanted);
      const t = d3.zoomIdentity.translate(m.l + plotW / 2, m.t + topH / 2).scale(k).translate(-X(0), -Y(0));
      overlay.transition().duration(450).call(zoom.transform, t);
    });
    overlay.call(zoom.transform, view.zoom);
    proximityChart = { update };

    // Below left: the inspection seen along the orbit, radial against cross-track.
    const lowTop = m.t + topH + 56;
    const leftW = Math.min(lowH * 1.5, plotW * 0.4);
    const rightX = m.l + leftW + 64;
    const rightW = m.l + plotW - rightX;
    const [tIn, tOut] = inspectionWindow(p);
    const inside = (times) => (q, i) => times[i] >= tIn && times[i] <= tOut;
    const reach = Math.max(p.ellipse_radial_m, p.ellipse_cross_track_m, p.keep_out_m) * 1.35;
    const mppLow = Math.max((2 * reach) / leftW, (2 * reach) / lowH);
    const Z = d3.scaleLinear().domain([(-mppLow * leftW) / 2, (mppLow * leftW) / 2]).range([m.l, m.l + leftW]);
    const V = d3.scaleLinear().domain([(-mppLow * lowH) / 2, (mppLow * lowH) / 2]).range([lowTop + lowH, lowTop]);
    const h = svg.append("g");
    h.append("text").attr("class", "bh-title").attr("x", m.l).attr("y", lowTop - 10).text("LOOKING ALONG THE ORBIT");
    h.append("rect").attr("class", "bh-frame").attr("x", m.l).attr("y", lowTop).attr("width", leftW).attr("height", lowH);
    defs.append("clipPath").attr("id", "ms-clip-low")
      .append("rect").attr("x", m.l).attr("y", lowTop).attr("width", leftW).attr("height", lowH);
    h.append("g").attr("class", "bh-axis").attr("transform", `translate(0,${lowTop + lowH})`)
      .call(d3.axisBottom(Z).ticks(4).tickFormat(d3.format(",")).tickSizeOuter(0));
    h.append("g").attr("class", "bh-axis").attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(V).ticks(4).tickFormat(d3.format(",")).tickSizeOuter(0));
    const low = h.append("g").attr("clip-path", "url(#ms-clip-low)");
    const lowLine = d3.line().x((q) => Z(q[2])).y((q) => V(q[0]));
    low.append("circle").attr("class", "ms-keepout").attr("cx", Z(0)).attr("cy", V(0)).attr("r", Z(p.keep_out_m) - Z(0));
    low.append("g").selectAll("path").data(mc.shown).join("path").attr("class", "ms-runs")
      .attr("d", (run) => lowLine(run.filter(inside(mc.shown_h))));
    low.append("path").attr("class", "ms-nominal").attr("d", lowLine(p.nominal.filter(inside(p.nominal_h))));
    low.append("path").attr("class", "ms-target").attr("transform", `translate(${Z(0)},${V(0)})`)
      .attr("d", d3.symbol(d3.symbolDiamond, 48)());
    h.append("text").attr("class", "ms-label").attr("x", m.l + 6).attr("y", lowTop + lowH - 6).text("CROSS-TRACK →");
    h.append("text").attr("class", "ms-label").attr("x", m.l + 6).attr("y", lowTop + 14).text("RADIAL ↑");
    h.append("text").attr("class", "ms-label ms-label-keepout").attr("x", Z(0)).attr("y", V(0) + 4)
      .attr("text-anchor", "middle").attr("dy", Z(p.keep_out_m) - Z(0) > 34 ? 16 : -12)
      .text(Z(p.keep_out_m) - Z(0) > 34 ? "KEEP-OUT" : "");

    // Below right: the Monte Carlo, delta-v and closest approach per run.
    if (rightW < 160) return;
    const hGap = 48;
    const hH = (lowH - hGap) / 2;
    histogram(rightX, lowTop, rightW, hH, mc.dv_m_s, {
      title: `DELTA-V, ${mc.runs.toLocaleString("en")} RUNS`,
      unit: "M/S",
      marks: [
        { value: p.dv_m_s, cls: "ms-planned", label: `PLANNED ${p.dv_m_s.toFixed(2)}`, side: "left" },
        { value: mc.dv_budget_m_s, cls: "ms-p99", label: `BUDGET ${mc.dv_budget_m_s.toFixed(2)} · ${mc.percentile}TH PCT` },
      ],
    });
    histogram(rightX, lowTop + hH + hGap, rightW, hH, mc.closest_m, {
      title: "CLOSEST APPROACH",
      unit: "M",
      marks: [
        { value: p.keep_out_m, cls: "ms-keepout-line", label: `KEEP-OUT · ${fmtPercent(mc.inside_keep_out)} OF RUNS INSIDE` },
      ],
      below: p.keep_out_m,
    });
  }

  function histogram(gx, gy, w, h, values, { title, unit, marks, below = null }) {
    const g = svg.append("g");
    g.append("text").attr("class", "bh-title").attr("x", gx).attr("y", gy - 10).text(title);
    g.append("text").attr("class", "bh-unit").attr("x", gx + w).attr("y", gy - 10).attr("text-anchor", "end").text(unit);
    g.append("rect").attr("class", "bh-frame").attr("x", gx).attr("y", gy).attr("width", w).attr("height", h);
    const [a, b] = d3.extent(values.concat(marks.map((mk) => mk.value)));
    const span = b - a || 1;
    const x = d3.scaleLinear().domain([a - 0.04 * span, b + 0.04 * span]).range([gx, gx + w]);
    const bins = d3.bin().domain(x.domain()).thresholds(x.ticks(Math.max(10, Math.floor(w / 9))))(values);
    const y = d3.scaleLinear().domain([0, d3.max(bins, (bin) => bin.length)]).range([gy + h, gy + 4]);
    g.selectAll("rect.ms-bar").data(bins.filter((bin) => bin.length)).join("rect")
      .attr("class", (bin) => `ms-bar${below !== null && bin.x0 < below ? " inside" : ""}`)
      .attr("x", (bin) => x(bin.x0) + 0.5)
      .attr("width", (bin) => Math.max(1, x(bin.x1) - x(bin.x0) - 1))
      .attr("y", (bin) => Math.min(y(bin.length), gy + h - 2))
      .attr("height", (bin) => Math.max(2, gy + h - y(bin.length)));
    g.append("g").attr("class", "bh-axis").attr("transform", `translate(0,${gy + h})`)
      .call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(w / 80))).tickSizeOuter(0));
    for (const mk of marks) {
      const mx = x(mk.value);
      const fits = mx + 12 + 7.2 * mk.label.length < gx + w;
      const left = mk.side === "left" || !fits;
      g.append("line").attr("class", mk.cls).attr("x1", mx).attr("x2", mx).attr("y1", gy).attr("y2", gy + h);
      g.append("text").attr("class", `ms-label ${mk.cls}-label`).attr("x", left ? mx - 5 : mx + 5).attr("y", gy + 13)
        .attr("text-anchor", left ? "end" : "start").text(mk.label);
    }
  }

  function placeTip(event) {
    const rect = charts.getBoundingClientRect();
    tooltip.hidden = false;
    let left = event.clientX - rect.left + 16;
    if (left + tooltip.offsetWidth > rect.width - 8) left = event.clientX - rect.left - tooltip.offsetWidth - 16;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${Math.max(6, Math.min(event.clientY - rect.top - 20, rect.height - tooltip.offsetHeight - 6))}px`;
  }

  function showBurnTip(event, burns, numbers) {
    const p = view.data.proximity;
    const parts = [];
    for (const k of burns) {
      const b = p.burns[k];
      const block = el("div", "tt-event");
      block.append(
        el("div", "tt-head", `${numbers.get(k)} · ${b.label.toUpperCase()}`),
        row("AT", `${b.time_h.toFixed(1)} h`),
        row("PLANNED", `${b.dv_m_s.toFixed(2)} m/s`),
        row("99TH PCT", `${b.p99_m_s.toFixed(2)} m/s`),
        row("IF IT FAILS", `${fmtKm(b.missed.closest_m)} closest`, b.missed.passes ? "good" : "warn"),
      );
      parts.push(block);
    }
    tooltip.replaceChildren(...parts, el("div", "tt-hint", "CLICK TO KEEP THE FAILURE CASE SHOWN"));
    placeTip(event);
  }

  function setHover(burns) {
    view.hoverBurns = burns;
    proximityChart?.update();
    highlightBurnRows();
  }

  function togglePin(burns) {
    view.pinnedBurns = view.pinnedBurns?.join() === burns.join() ? null : burns;
    proximityChart?.update();
    highlightBurnRows();
  }

  function highlightBurnRows() {
    const shown = shownBurns();
    for (const tr of $("mt-burns").children) tr.classList.toggle("on", shown.has(Number(tr.dataset.burn)));
  }

  function setMode(mode) {
    if (mode === view.mode) return;
    view.mode = mode;
    if (mode === "proximity") $("mt-prox-fold").open = true;
    renderToolbar();
    renderCharts();
  }

  // -- side panel -----------------------------------------------------------

  function renderPanel() {
    const d = view.data;
    $("mt-target").textContent = view.name || "";
    $("mt-content").hidden = !d;
    const status = $("mt-status");
    status.hidden = Boolean(d);
    if (!d) {
      status.textContent = view.error ?? (view.requestKey ? "Planning." : "Open the MISSION view (M) to plan a rendezvous with this object.");
      return;
    }
    const o = d.options[view.selected];
    const days = Math.round((o.arrival_unix_ms - d.epoch_unix_ms) / DAY_MS);
    $("mt-summary").replaceChildren(
      row("ARRIVE", `${fmtDate(o.arrival_unix_ms)} · ${days} d`, view.overBudget ? "warn" : "good"),
      row("TRANSFER", `${o.transfer_m_s.toFixed(1)} m/s`),
      row("BUDGET", `${fmtV(o.dv_nominal_m_s)} · ${fmtV(o.dv_margined_m_s)} m/s with margins`),
      row("PROPELLANT", `${o.propellant_kg.toFixed(1)} kg · wet ${o.wet_mass_kg.toFixed(0)} kg`),
      row("PLANE GAP", `${d.gap.raan_deg.toFixed(1)}° node · ${d.gap.inclination_deg.toFixed(2)}° incl`, "dim"),
    );
    $("mt-over").hidden = !view.overBudget;
    if (view.overBudget) {
      $("mt-over").textContent = `Nothing fits in ${Math.round(view.budgetDays)} days: the fastest way takes ${Math.ceil(d.options[0].total_days)}.`;
    }

    const steps = [
      [
        d.dropoff.kind === "sun-synchronous"
          ? `Drop-off at ${d.dropoff.altitude_km.toFixed(0)} km, sun-synchronous, crossing the equator at ${fmtClock(d.dropoff.ltan_h)}`
          : `Drop-off at ${d.dropoff.altitude_km.toFixed(0)} km in the target's ${d.dropoff.inclination_deg.toFixed(2)}° inclination`,
        "",
      ],
      [`Enter a drift orbit at ${o.altitude_km.toFixed(0)} km, i ${o.inclination_deg.toFixed(2)}°`, `${o.enter_m_s.toFixed(1)} m/s`],
      [`Wait ${o.wait_days.toFixed(0)} days while J2 lines up the planes`, ""],
      [`Time the exit to arrive in phase, up to ${o.phasing_days.toFixed(1)} days`, ""],
      [`Transfer into ${d.target.name}'s orbit`, `${o.leave_m_s.toFixed(1)} m/s`],
      [`Lambert approach to ${d.approach.hold_km} km behind, ${d.approach.minutes.toFixed(0)} min`, `${d.approach.dv_m_s.toFixed(1)} m/s`],
    ];
    const p = d.proximity;
    const mc = p.monte_carlo;
    const waypoint = Math.abs(p.burns.find((b) => b.kind === "brake").position_m[1]);
    steps.push([
      `Hop to ${fmtKm(waypoint)}, inspect for ${p.inspection_orbits} orbit${p.inspection_orbits === 1 ? "" : "s"} on a ${p.ellipse_radial_m} m safety ellipse, leave`,
      `${mc.dv_budget_m_s.toFixed(1)} m/s`,
    ]);
    $("mt-steps").replaceChildren(
      ...steps.map(([text, dv], i) => {
        const li = el("li", "link");
        li.title = i === steps.length - 1 ? "Show the proximity operations" : "Show the transfer";
        li.append(el("span", "", text), el("b", "", dv));
        li.addEventListener("click", () => setMode(i === steps.length - 1 ? "proximity" : "transfer"));
        return li;
      }),
    );

    $("mt-prox-sum").textContent = `${mc.dv_budget_m_s.toFixed(1)} M/S`;
    $("mt-prox").replaceChildren(
      row("PLANNED", `${p.dv_m_s.toFixed(2)} m/s over ${p.duration_h.toFixed(1)} h`),
      row("BUDGETED", `${mc.dv_budget_m_s.toFixed(2)} m/s · ${mc.percentile}th pct`),
      row("IF A BURN FAILS", p.all_safe ? "safe, every one" : "not always safe", p.all_safe ? "good" : "warn"),
      row("CLOSEST, 1 IN 100", `${Math.round(mc.closest_p1_m)} m`, mc.closest_p1_m >= p.keep_out_m ? "" : "warn"),
      row("INSIDE KEEP-OUT", `${fmtPercent(mc.inside_keep_out)} of runs`, mc.inside_keep_out > 0.01 ? "warn" : "good"),
    );
    const numbers = burnNumbers(p);
    $("mt-burns").replaceChildren(
      ...[...numbers].map(([k, n]) => {
        const b = p.burns[k];
        const tr = el("tr");
        tr.dataset.burn = k;
        tr.title = `${b.p99_m_s.toFixed(2)} m/s at the 99th percentile. If it fails, the spacecraft coasts to ${fmtKm(b.missed.closest_m)} of the target ${b.missed.after_h.toFixed(1)} h later; the keep-out sphere is ${p.keep_out_m} m.`;
        tr.append(
          el("td", "", String(n)),
          el("td", "", b.label),
          el("td", "", b.dv_m_s.toFixed(2)),
          el("td", b.missed.passes ? "safe" : "unsafe", fmtKm(b.missed.closest_m)),
        );
        tr.addEventListener("mouseenter", () => setHover([k]));
        tr.addEventListener("mouseleave", () => setHover(null));
        tr.addEventListener("click", () => {
          setMode("proximity");
          togglePin([k]);
        });
        return tr;
      }),
    );
    highlightBurnRows();
    const e = mc.errors;
    const num = (v) => String(+v.toFixed(3));
    $("mt-prox-note").textContent =
      `IF IT FAILS: closest approach while coasting ${p.safety_orbits} orbits after that burn is missed. ` +
      "A hop along the V-bar is a closed loop, so a missed stop coasts back to where it began; on the ellipse, " +
      "radial and cross-track motion are a quarter orbit apart, so it never crosses the target's orbit line. " +
      "Course corrections and trims are zero in the plan and spent only on errors. " +
      `Monte Carlo errors, one sigma: range ${num(e.range_percent)}% and bearing ${num(e.bearing_deg)}° at every burn, ` +
      `burn size ${num(e.burn_magnitude_percent)}% and pointing ${num(e.burn_pointing_deg)}°, ${num(e.arrival_m)} m at the hold point. ` +
      "Each burn is recomputed from a single noisy fix; no navigation filter is modelled.";

    $("mt-budget-sum").textContent = `${fmtV(o.dv_margined_m_s)} M/S`;
    $("mt-budget").replaceChildren(
      ...d.lines.map((line, i) => {
        const tr = el("tr");
        tr.title = `${o.basis[i]} (${line.requirement})`;
        tr.append(
          el("td", "", line.label),
          el("td", "", o.lines[i][0].toFixed(1)),
          el("td", "", `${Math.round(line.margin * 100)}%`),
          el("td", "", o.lines[i][1].toFixed(1)),
        );
        return tr;
      }),
    );
    $("mt-reference").textContent = `Margins per ${d.reference}. Dry mass carries a 20% system margin (${d.spacecraft.dry_mass_with_margin_kg.toFixed(0)} kg) and propellant 2% residuals.`;
    $("mt-craft-sum").textContent = `${view.params.dry} KG · ISP ${view.params.isp} S`;
    $("mt-assume-sum").textContent = `${view.params.inj_alt} KM · ${view.params.inj_inc}° · ${view.params.ops} D`;
    if (view.params.errors !== 1) $("mt-prox-sum").textContent += ` · ERRORS ×${view.params.errors}`;
    $("mt-model").textContent = `Model: ${d.model}.`;
  }

  // -- controls -------------------------------------------------------------

  function fillForms() {
    $("mt-dry").value = view.params.dry;
    $("mt-isp").value = view.params.isp;
    const preset = ["70", "220", "300"].includes(String(view.params.isp)) ? String(view.params.isp) : "custom";
    $("mt-thruster").value = preset;
    $("mt-inj-alt").value = view.params.inj_alt;
    $("mt-inj-inc").value = view.params.inj_inc;
    $("mt-ops").value = view.params.ops;
    $("mt-disposal").value = view.params.disposal;
  }

  $("ms-alt").addEventListener("input", (e) => ($("ms-alt-v").textContent = `${e.target.value} km`));
  $("ms-offset").addEventListener("input", (e) => {
    $("ms-offset-v").textContent = fmtOffset(Number(e.target.value), view.data?.dropoff.kind === "sun-synchronous");
  });
  $("ms-alt").addEventListener("change", (e) => {
    view.params.altitude = Number(e.target.value);
    load();
  });
  $("ms-offset").addEventListener("change", (e) => {
    view.params.node_offset = Number(e.target.value);
    load();
  });
  $("ms-days").addEventListener("input", (e) => {
    view.budgetDays = Number(e.target.value);
    $("ms-days-v").textContent = `${view.budgetDays} d`;
    if (!view.data) return;
    chooseByBudget();
    renderCharts();
    renderPanel();
  });
  for (const button of $("ms-mode").querySelectorAll("button")) {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  }
  const proximitySliders = [
    ["ms-ellipse", "ellipse", (v) => `${v} m`],
    ["ms-keepout", "keep_out", (v) => `${v} m`],
    ["ms-orbits", "orbits", (v) => `${v} orbit${v === 1 ? "" : "s"}`],
    ["ms-errors", "errors", fmtErrors],
  ];
  for (const [id, key, format] of proximitySliders) {
    $(id).addEventListener("input", (e) => ($(`${id}-v`).textContent = format(Number(e.target.value))));
    $(id).addEventListener("change", (e) => {
      view.params[key] = Number(e.target.value);
      // The ellipse must pass wider than the keep-out sphere; whichever moved, the keep-out gives way.
      view.params.keep_out = Math.min(view.params.keep_out, view.params.ellipse - KEEP_OUT_GAP_M);
      renderToolbar();
      load();
    });
  }
  $("mt-thruster").addEventListener("change", (e) => {
    if (e.target.value !== "custom") $("mt-isp").value = e.target.value;
  });
  $("mt-craft").addEventListener("submit", (e) => {
    e.preventDefault();
    view.params.dry = Number($("mt-dry").value);
    view.params.isp = Number($("mt-isp").value);
    onShowRequested();
    load();
  });
  $("mt-assume").addEventListener("submit", (e) => {
    e.preventDefault();
    view.params.inj_alt = Number($("mt-inj-alt").value);
    view.params.inj_inc = Number($("mt-inj-inc").value);
    view.params.ops = Number($("mt-ops").value);
    view.params.disposal = Number($("mt-disposal").value);
    onShowRequested();
    load();
  });
  $("mt-reset").addEventListener("click", () => {
    view.params = { ...DEFAULTS };
    view.budgetDays = null;
    fillForms();
    load();
  });

  let resizeFrame = null;
  new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      if (view.active) renderCharts();
    });
  }).observe(charts);
  fillForms();

  return {
    setObject(norad, name) {
      if (norad !== view.norad) {
        view.norad = norad;
        view.data = null;
        view.key = null;
        view.error = null;
        view.budgetDays = null;
        view.zoom = d3.zoomIdentity;
        view.pinnedBurns = null;
      }
      view.name = name;
      if (view.active) load();
      else renderPanel();
    },
    activate() {
      view.active = true;
      load();
      renderCharts();
    },
    deactivate() {
      view.active = false;
      tooltip.hidden = true;
    },
    toggleMode() {
      setMode(view.mode === "proximity" ? "transfer" : "proximity");
    },
  };
}
