/*
 * ORBWATCH mission view.
 *
 * Plans a rendezvous with the tracked object from a sun-synchronous rideshare
 * and shows the one trade that matters: how fast against how much delta-v.
 * The Python API computes the whole front and a full budget for every option
 * on it, so moving the time budget or clicking a dot is instant, and only a
 * change to the drop-off, the spacecraft or the assumptions asks again.
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
const fmtOffset = (h) => `${h > 0 ? "+" : h < 0 ? "−" : ""}${Math.abs(h)} h`;
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

const DEFAULTS = { altitude: 525, offset: -1, dry: 150, isp: 220, inj_alt: 10, inj_inc: 0.1, prox: 10, ops: 90, disposal: 300 };

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
  };

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
    $("ms-sub").textContent = d
      ? `RENDEZVOUS FROM A RIDESHARE · TARGET ${d.target.altitude_km.toFixed(0)} KM, i ${d.target.inclination_deg.toFixed(2)}°, CROSSES THE EQUATOR NORTHBOUND AT ${fmtClock(d.target.ltan_h)}`
      : "RENDEZVOUS FROM A RIDESHARE";
    $("ms-alt").value = view.params.altitude;
    $("ms-alt-v").textContent = `${view.params.altitude} km`;
    $("ms-offset").value = view.params.offset;
    $("ms-offset-v").textContent = fmtOffset(view.params.offset);
    if (view.budgetDays !== null) {
      $("ms-days").value = Math.round(view.budgetDays);
      $("ms-days-v").textContent = `${Math.round(view.budgetDays)} d`;
    }
    const warning = $("ms-warning");
    warning.hidden = !d?.warning;
    if (d?.warning) warning.replaceChildren(el("b", "", "Check the drop-off. "), document.createTextNode(d.warning));
    const message = $("ms-message");
    message.hidden = !view.error;
    if (view.error) {
      message.replaceChildren(el("div", "bh-message-title", "NO MISSION PLAN"), el("p", "", view.error));
    }
  }

  function renderCharts() {
    svg.selectAll("*").remove();
    tooltip.hidden = true;
    const d = view.data;
    if (!d || !view.active) return;
    const W = charts.clientWidth;
    const H = charts.clientHeight;
    if (W < 300 || H < 260) return;
    svg.attr("width", W).attr("height", H);
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
    const y = d3.scaleLinear().domain([d.floor_m_s * 0.92, vMax * 1.06]).range([m.t + tradeH, m.t]);
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

    g.append("line").attr("class", "ms-floor").attr("x1", m.l).attr("x2", m.l + plotW)
      .attr("y1", y(d.floor_m_s)).attr("y2", y(d.floor_m_s));
    g.append("text").attr("class", "ms-label ms-label-floor").attr("x", m.l + plotW - 6).attr("y", y(d.floor_m_s) + 13)
      .attr("text-anchor", "end").text(`FLOOR ${d.floor_m_s.toFixed(0)} M/S · THE RAISE AND THE ${Math.abs(d.gap.inclination_deg).toFixed(2)}° TILT, WHATEVER THE WAIT`);
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
      [`Drop-off at ${d.dropoff.altitude_km.toFixed(0)} km, sun-synchronous, crossing at ${fmtClock(d.dropoff.ltan_h)}`, ""],
      [`Enter a drift orbit at ${o.altitude_km.toFixed(0)} km, i ${o.inclination_deg.toFixed(2)}°`, `${o.enter_m_s.toFixed(1)} m/s`],
      [`Wait ${o.wait_days.toFixed(0)} days while J2 lines up the planes`, ""],
      [`Time the exit to arrive in phase, up to ${o.phasing_days.toFixed(1)} days`, ""],
      [`Transfer into ${d.target.name}'s orbit`, `${o.leave_m_s.toFixed(1)} m/s`],
      [`Lambert approach to ${d.approach.hold_km} km behind, ${d.approach.minutes.toFixed(0)} min`, `${d.approach.dv_m_s.toFixed(1)} m/s`],
    ];
    $("mt-steps").replaceChildren(
      ...steps.map(([text, dv]) => {
        const li = el("li");
        li.append(el("span", "", text), el("b", "", dv));
        return li;
      }),
    );

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
    $("mt-prox").value = view.params.prox;
    $("mt-ops").value = view.params.ops;
    $("mt-disposal").value = view.params.disposal;
  }

  $("ms-alt").addEventListener("input", (e) => ($("ms-alt-v").textContent = `${e.target.value} km`));
  $("ms-offset").addEventListener("input", (e) => ($("ms-offset-v").textContent = fmtOffset(Number(e.target.value))));
  $("ms-alt").addEventListener("change", (e) => {
    view.params.altitude = Number(e.target.value);
    load();
  });
  $("ms-offset").addEventListener("change", (e) => {
    view.params.offset = Number(e.target.value);
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
    view.params.prox = Number($("mt-prox").value);
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
  };
}
