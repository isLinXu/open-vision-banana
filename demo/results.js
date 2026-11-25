/* Open-Vision-Banana project page. Everything numeric is read from data/results.json,
   which scripts/render_site_data.py builds from the published evaluation JSONs. */
(async function () {
  const $ = (s, el = document) => el.querySelector(s);
  const h = (tag, attrs = {}, html = "") => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    el.innerHTML = html;
    return el;
  };
  const load = async (p) => { try { const r = await fetch(p); return r.ok ? r.json() : null; } catch { return null; } };
  const R = await load("data/results.json");
  const G = await load("data/gallery.json");
  if (!R) { $("#results").insertAdjacentHTML("beforeend", '<p class="note">data/results.json not found.</p>'); return; }

  const SVG = "http://www.w3.org/2000/svg";
  const svg = (w, hgt) => { const s = document.createElementNS(SVG, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${hgt}`); return s; };
  const add = (parent, tag, attrs = {}, text) => {
    const el = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (text !== undefined) el.textContent = text;
    parent.appendChild(el);
    return el;
  };

  const runs = R.runs;
  const bench = Object.fromEntries(R.benchmarks.map((b) => [b.key, b]));
  const is2d = (k) => ["cityscapes", "refcocog", "reasonseg"].includes(k);
  const unit = (k, v) => (is2d(k) ? v / 100 : v); // common 0..1 scale for plotting
  const fmt = (v, d) => (v === undefined || v === null ? "—" : Number(v).toFixed(d));
  const cell = (run, k) => runs[run] && runs[run].values[k];

  // ------------------------------------------------------------------ tooltip
  const tip = $("#tip");
  const showTip = (e, html) => {
    tip.innerHTML = html; tip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = e.clientY + 16 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 12 : e.clientY + 16;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  };
  const hideTip = () => { tip.hidden = true; };
  const srcLine = (c) => (c ? `<span class="src">n = ${c.n} · ${c.src}</span>` : "");

  // ------------------------------------------------------------------ hero facts
  const evaluated = Object.keys(runs).filter((r) => !R.protocol_variants.includes(r)).length;
  const tasks = R.recipe.tasks.filter((t) => t !== "generation").length;
  const facts = [
    [`${tasks}`, "vision tasks in one adapter; the prompt picks the task"],
    [`${evaluated}`, `runs scored on ${R.benchmarks.length} benchmarks, full official splits`],
    [`${R.source_count}`, "evaluation files behind this page"],
    [`${R.audit.errors}`, `implementation errors in ${R.audit.checkable} audited paper statements`],
  ];
  $("#facts").innerHTML = facts.map(([n, l]) => `<div><dt>${n}</dt><dd>${l}</dd></div>`).join("");

  // ------------------------------------------------------------------ method
  const A = R.audit;
  $("#audit").innerHTML = [[A.checkable, "checkable"], [A.identical, "identical"], [A.partial, "partial"], [A.missing, "missing (closed)"]]
    .map(([n, l]) => `<div><b>${n}</b><span>${l}</span></div>`).join("");
  const rc = R.recipe;
  const rows = [
    ["Backbone", "Qwen-Image-2.1 (source image as visual tokens + reference latents)"],
    ["Tuning", `rsLoRA r${rc.lora_rank} / α${rc.lora_alpha}, flow matching, lr ${Number(rc.lr).toExponential().replace("e-", "e-").replace(/\.0+e/, "e")}, ${rc.steps.toLocaleString()} steps`],
    ["Batch", `${rc.batch_size} × ${rc.grad_accum} accumulation, ${rc.resolution}² canvas`],
    ["Tasks", rc.tasks.join(", ")],
    ["Mixture", `vision_ratio ${rc.vision_ratio} (paper: "very low", no number)`],
    ["Depth targets", rc.depth_colormaps.join(", ").replace("_", "-")],
    ["Training rows", `${rc.train_rows_total.toLocaleString()} after the strict zero-shot audit (cap ${rc.cap_per_task.toLocaleString()} per task)`],
  ];
  $("#recipe").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  // ------------------------------------------------------------------ headline dumbbell
  const base = R.headline.base, ours = R.headline.ours;
  $("#ours-label").textContent = runs[ours].label;
  $("#unlock").textContent = R.findings.unlock;
  {
    const W = 960, rowH = 46, top = 26, left = 150, right = 70, H = top + rowH * R.benchmarks.length + 8;
    const x = (u) => left + u * (W - left - right);
    const s = svg(W, H);
    const grid = add(s, "g", { class: "grid" }), axis = add(s, "g", { class: "axis" });
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      add(grid, "line", { x1: x(t), x2: x(t), y1: top - 6, y2: H - 6 });
      add(axis, "text", { x: x(t), y: top - 12, "text-anchor": "middle" }, t === 0 ? "0" : t === 1 ? "1.0" : String(t));
    }
    R.benchmarks.forEach((b, i) => {
      const cy = top + rowH * i + rowH / 2;
      const cb = cell(base, b.key), co = cell(ours, b.key);
      add(s, "text", { x: 0, y: cy - 2, class: "row-label" }, b.label);
      add(s, "text", { x: 0, y: cy + 14, class: "row-sub" }, `${b.metric}${b.key === "reasonseg" ? " · diagnostic" : ""}`);
      const g = add(s, "g");
      add(g, "line", { x1: x(unit(b.key, cb.v)), x2: x(unit(b.key, co.v)), y1: cy, y2: cy, stroke: "#2a78d6", "stroke-width": 2 });
      add(g, "line", { x1: x(unit(b.key, b.paper)), x2: x(unit(b.key, b.paper)), y1: cy - 10, y2: cy + 10, stroke: "#1f2033", "stroke-width": 2, "stroke-linecap": "round" });
      add(g, "circle", { cx: x(unit(b.key, cb.v)), cy, r: 5, fill: "#fff", stroke: "#5d6077", "stroke-width": 2 });
      add(g, "circle", { cx: x(unit(b.key, co.v)), cy, r: 6, fill: "#2a78d6", stroke: "#fff", "stroke-width": 2 });
      const lo = Math.min(unit(b.key, cb.v), unit(b.key, co.v));
      add(g, "text", { x: x(lo) - 10, y: cy + 4, "text-anchor": "end", class: "val" }, fmt(co.v >= cb.v ? cb.v : co.v, b.digits));
      add(g, "text", { x: x(Math.max(unit(b.key, cb.v), unit(b.key, co.v))) + 10, y: cy + 4, class: "val" }, fmt(co.v >= cb.v ? co.v : cb.v, b.digits));
      add(g, "text", { x: x(unit(b.key, b.paper)) + 8, y: cy - 8, class: "lbl" }, fmt(b.paper, b.digits));
      const hit = add(s, "rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, class: "hit" });
      hit.addEventListener("mousemove", (e) => showTip(e,
        `<b>${b.label} ${b.metric}</b><br>untuned base ${fmt(cb.v, b.digits)} → ${runs[ours].label} ${fmt(co.v, b.digits)}<br>paper ${fmt(b.paper, b.digits)}${srcLine(co)}`));
      hit.addEventListener("mouseleave", hideTip);
    });
    $("#dumbbell").appendChild(s);
    const p = R.inference_protocol;
    $("#protocol").textContent = `2D scores ×100, depth δ1 in [0, 1]; plotted on a common 0–1 scale. Protocol for every point: ${p.num_inference_steps} steps, CFG ${p.guidance_scale}, strength ${p.strength}, seed ${p.seed}, ${p.canvas_resolution}px canvas, scored at native ground-truth resolution. ${p.note}`;
  }

  // ------------------------------------------------------------------ tabs helper
  const tabs = (el, items, onPick) => {
    el.innerHTML = "";
    items.forEach(([id, label], i) => {
      const b = h("button", { role: "tab", "aria-selected": i === 0 ? "true" : "false" }, label);
      b.addEventListener("click", () => {
        el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", "false"));
        b.setAttribute("aria-selected", "true");
        onPick(id);
      });
      el.appendChild(b);
    });
    onPick(items[0][0]);
  };

  // ------------------------------------------------------------------ result tables
  const tableHTML = (t) => {
    let html = `<div class="table-wrap"><table class="data"><thead><tr><th>${t.title}</th>${t.columns.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead><tbody>`;
    for (const row of t.rows) {
      const sub = row.run ? `<span class="sub">${runs[row.run].recipe}</span>` : row.kind === "paper" ? '<span class="sub">paper, different closed base model</span>' : "";
      html += `<tr class="${row.kind}"><td>${row.label}${sub}</td>` + t.columns.map((c) => {
        const val = row.values[c.key];
        return val === undefined ? '<td class="na">—</td>' : `<td>${fmt(val, c.digits)}</td>`;
      }).join("") + "</tr>";
    }
    return html + `</tbody></table></div><p class="note">${t.note || ""}</p>`;
  };
  tabs($("#table-tabs"), R.tables.map((t, i) => [i, t.title.split(" (")[0].split(" — ")[0]]), (i) => { $("#table-view").innerHTML = tableHTML(R.tables[i]); });

  // ------------------------------------------------------------------ findings
  const F = R.findings;
  const cards = [
    ["Data", "A quarter of the data was not worse", F.quarter],
    ["Steps", "More steps did not help referring", F.longer],
    ["Noise", "Training-seed noise is large", F.noise],
    ["Depth", "A fixed colormap helps metric depth", F.cube],
    ["Segmentation", "Re-mixing tasks does not fix segmentation", F.seg],
    ["Capacity", "Full fine-tuning at 3e-5 hurt", F.full],
  ];
  $("#findings").innerHTML = cards.map(([tag, t, p]) => `<article class="finding"><span class="tag">${tag}</span><h3>${t}</h3><p>${p}</p></article>`).join("");

  // ------------------------------------------------------------------ E1 curve, small multiples
  {
    const C = R.curve;
    $("#curve-ref").textContent = runs[C.compare].label;
    for (const k of C.keys) {
      const b = bench[k];
      const pts = C.runs.map((r, i) => ({ s: C.steps[i], c: cell(r, k), r }));
      const ref = cell(C.compare, k), bse = cell(base, k);
      const W = 260, H = 190, l = 34, r = 30, t = 16, bt = 26;
      const step = is2d(k) ? 10 : 0.05;
      const vmax = Math.ceil((Math.max(...pts.map((p) => p.c.v), ref.v, bse.v) * 1.12) / step) * step;
      const x = (s) => l + (s / 30000) * (W - l - r);
      const y = (v) => t + (1 - v / vmax) * (H - t - bt);
      const s = svg(W, H);
      const grid = add(s, "g", { class: "grid" }), axis = add(s, "g", { class: "axis" });
      [0, vmax / 2, vmax].forEach((v) => {
        add(grid, "line", { x1: l, x2: W - r, y1: y(v), y2: y(v) });
        add(axis, "text", { x: l - 6, y: y(v) + 4, "text-anchor": "end" }, fmt(v, is2d(k) ? 0 : 2));
      });
      [0, 10000, 20000, 30000].forEach((st) => add(axis, "text", { x: x(st), y: H - 6, "text-anchor": "middle" }, st ? `${st / 1000}k` : "0"));
      add(s, "line", { x1: l, x2: W - r, y1: y(bse.v), y2: y(bse.v), stroke: "#5d6077", "stroke-width": 1.5, "stroke-dasharray": "4 4" });
      add(s, "polyline", { points: pts.map((p) => `${x(p.s)},${y(p.c.v)}`).join(" "), fill: "none", stroke: "#2a78d6", "stroke-width": 2, "stroke-linejoin": "round" });
      add(s, "circle", { cx: x(10000), cy: y(ref.v), r: 5, fill: "#fff", stroke: "#2a78d6", "stroke-width": 2 });
      const first = pts[0], last = pts[pts.length - 1];
      add(s, "text", { x: x(first.s), y: y(first.c.v) - 9, "text-anchor": "middle", class: "val" }, fmt(first.c.v, b.digits));
      add(s, "text", { x: x(last.s) + 7, y: y(last.c.v) + 4, class: "val" }, fmt(last.c.v, b.digits));
      for (const p of pts) {
        add(s, "circle", { cx: x(p.s), cy: y(p.c.v), r: 4, fill: "#2a78d6", stroke: "#fff", "stroke-width": 1.5 });
        const hit = add(s, "circle", { cx: x(p.s), cy: y(p.c.v), r: 14, class: "hit" });
        hit.addEventListener("mousemove", (e) => showTip(e, `<b>${runs[p.r].label}</b><br>${b.label} ${b.metric} ${fmt(p.c.v, b.digits)}${srcLine(p.c)}`));
        hit.addEventListener("mouseleave", hideTip);
      }
      const hr = add(s, "circle", { cx: x(10000), cy: y(ref.v), r: 12, class: "hit" });
      hr.addEventListener("mousemove", (e) => showTip(e, `<b>${runs[C.compare].label}</b><br>${b.label} ${b.metric} ${fmt(ref.v, b.digits)}${srcLine(ref)}`));
      hr.addEventListener("mouseleave", hideTip);
      const box = h("div", {}, `<h4>${b.label} <span class="small">${b.metric}</span></h4>`);
      box.appendChild(s);
      $("#curve").appendChild(box);
    }
  }

  // ------------------------------------------------------------------ ablations
  {
    const cols = [
      ...R.benchmarks.map((b) => ({ key: b.key, name: b.label, metric: b.metric, digits: b.digits, up: true })),
    ];
    cols.splice(4, 0, { key: "nyu_absrel", name: "NYUv2", metric: "AbsRel", digits: 3, up: false });
    cols.push({ key: "gen", name: "Generation", metric: "FID-proxy", digits: 2, up: false });
    cols.forEach((c) => { c.label = `${c.name} ${c.metric}`; });
    const val = (run, k) => (k === "gen" ? (runs[run].gen ? { v: runs[run].gen.fid_proxy, n: runs[run].gen.n, src: runs[run].gen.src } : undefined) : cell(run, k));
    const notes = { i88: [R.findings.cube, R.findings.seg], i91: [R.findings.fresh, R.findings.full], e2: [R.findings.noise] };
    const render = (id) => {
      const g = R.ablations.find((a) => a.id === id);
      let html = `<div class="table-wrap"><table class="data"><thead><tr><th>Run</th>${cols.map((c) => `<th>${c.name}<span class="sub">${c.metric} ${c.up ? "↑" : "↓"}</span></th>`).join("")}</tr></thead><tbody>`;
      const ext = {};
      for (const c of cols) {
        const vs = g.rows.map((r) => val(r, c.key)).filter(Boolean).map((x) => x.v).filter((v) => !(c.key === "gen" && v === 0));
        ext[c.key] = vs.length ? { max: Math.max(...vs), min: Math.min(...vs) } : null;
      }
      for (const r of g.rows) {
        const step0 = /step 0|untuned/.test(runs[r].label);
        html += `<tr class="${step0 ? "base-ref" : ""}"><td>${runs[r].label}<span class="sub">${runs[r].recipe}</span></td>` + cols.map((c) => {
          const x = val(r, c.key);
          if (!x) return '<td class="na">—</td>';
          if (c.key === "gen" && x.v === 0) return '<td class="na">reference</td>';
          const e = ext[c.key];
          const frac = !e ? 0 : c.up ? x.v / e.max : (x.v === 0 ? 0 : e.min / x.v);
          const best = e && g.rows.length > 2 && (c.up ? x.v === e.max : x.v === e.min && x.v !== 0);
          return `<td data-run="${r}" data-k="${c.key}"><span class="cell${best ? " best" : ""}">${fmt(x.v, c.digits)}<i style="width:${Math.round(frac * 100)}%"></i></span></td>`;
        }).join("") + "</tr>";
      }
      html += `</tbody></table></div><p class="note">${g.note} <a href="https://github.com/isLinXu/open-vision-banana/issues/${g.issue.slice(1)}">${g.issue}</a></p>`;
      html += (notes[id] || []).map((n) => `<p class="note">${n}</p>`).join("");
      const view = $("#abl-view");
      view.innerHTML = html;
      view.querySelectorAll("td[data-run]").forEach((td) => {
        const x = val(td.dataset.run, td.dataset.k);
        td.addEventListener("mousemove", (e) => showTip(e, `<b>${runs[td.dataset.run].label}</b><br>${cols.find((c) => c.key === td.dataset.k).label} ${fmt(x.v, 3)}${srcLine(x)}`));
        td.addEventListener("mouseleave", hideTip);
      });
    };
    tabs($("#abl-tabs"), R.ablations.map((a) => [a.id, `${a.title} · ${a.issue}`]), render);
  }

  // ------------------------------------------------------------------ samples
  if (G) {
    const names = { depth: "Depth", normal: "Normals", semantic: "Semantic" };
    const avail = Object.keys(names).filter((t) => (G[t] || []).length);
    const render = (task) => {
      const view = $("#sample-view");
      view.innerHTML = "";
      for (const it of G[task]) {
        const card = h("div", { class: "sample", "data-show": "output" },
          `<div class="stack"><img src="${it.input}" alt="input image" loading="lazy"><img class="tgt" src="${it.target}" alt="ground-truth target" loading="lazy"><img class="out" src="${it.output}" alt="model output" loading="lazy"></div>
           <div class="seg">${["input", "target", "output"].map((m) => `<button type="button" data-m="${m}" aria-pressed="${m === "output"}">${m[0].toUpperCase() + m.slice(1)}</button>`).join("")}</div>`);
        card.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
          card.dataset.show = b.dataset.m;
          card.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        }));
        view.appendChild(card);
      }
      const it = G[task][0];
      $("#sample-meta").textContent = `${it.backbone} · ${it.checkpoint} · ${it.split}. Depth is closer-brighter grayscale for this adapter.`;
    };
    if (avail.length) tabs($("#sample-tabs"), avail.map((t) => [t, names[t]]), render);
  }

  // ------------------------------------------------------------------ status board
  const label = { measured: "measured", done: "done", running: "in progress", planned: "planned" };
  $("#board").innerHTML = R.status.map((s) =>
    `<li><span class="id">${s.id}</span><span class="t">${s.title}</span><span class="state ${s.state}">${label[s.state]}</span><span class="d">${s.detail}</span></li>`).join("");

  $("#provenance").textContent = `Open-Vision-Banana · unofficial, community-driven · rendered ${R.generated_at} from ${R.source_count} evaluation files${R.commit ? ` at ${R.commit}` : ""} by scripts/render_site_data.py`;

  // ------------------------------------------------------------------ active nav
  const links = [...document.querySelectorAll(".site-header nav a")];
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-40% 0px -55% 0px" });
  links.forEach((a) => { const t = document.querySelector(a.getAttribute("href")); if (t) io.observe(t); });
})();
