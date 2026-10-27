/* Open-Vision-Banana project page: render gallery and results from JSON (no build step). */
(async function () {
  const TASKS = [
    ["semantic", "Semantic Segmentation", "Open-vocabulary; the legend lives in the prompt (four styles)."],
    ["instance", "Instance Segmentation", "One class at a time; the model picks a colour per instance; Appendix A decoder."],
    ["referring", "Referring Expression Segmentation", "Fig. 4 template, white target on black."],
    ["depth", "Monocular Metric Depth", "Power transform + cube-edge colormap; decoded to metres without camera intrinsics."],
    ["normal", "Surface Normal Estimation", "Camera-frame normals mapped linearly to RGB."],
  ];

  async function load(path) {
    try { const r = await fetch(path); if (!r.ok) throw new Error(r.status); return await r.json(); }
    catch (e) { return null; }
  }

  // ---------------------------------------------------------------- gallery
  const gallery = await load("data/gallery.json");
  const galleryEl = document.getElementById("gallery");
  for (const [task, title, blurb] of TASKS) {
    const items = (gallery && gallery[task]) || [];
    const block = document.createElement("div");
    block.className = "task-block";
    block.innerHTML = `<h3>${title}</h3><p class="note">${blurb}</p><div class="tiles"></div>`;
    const tiles = block.querySelector(".tiles");
    const list = items.length ? items : [{ placeholder: true }, { placeholder: true }, { placeholder: true }];
    for (const it of list) {
      const tile = document.createElement("div");
      tile.className = "tile";
      if (it.placeholder) {
        tile.innerHTML = `<div class="img"><div class="placeholder">pending — populated by experiment E5 (issue #58)</div></div>
                          <div class="meta">checkpoint · prompt · seed recorded per tile</div>`;
      } else {
        tile.innerHTML = `<div class="img"><img src="${it.input}" alt="input"><img class="pred" src="${it.output}" alt="prediction"><span class="tag">hover: output</span></div>
                          <div class="meta"><b>Prompt:</b> ${it.prompt}<br><span>${it.checkpoint || ""}${it.seed !== undefined ? " · seed " + it.seed : ""}</span></div>`;
        tile.querySelector(".img").addEventListener("click", (e) => e.currentTarget.classList.toggle("show"));
      }
      tiles.appendChild(tile);
    }
    galleryEl.appendChild(block);
  }

  // ---------------------------------------------------------------- results
  const results = await load("data/results.json");
  const resEl = document.getElementById("results-tables");
  const protoEl = document.getElementById("protocol-box");
  if (!results) { resEl.innerHTML = '<p class="note">results.json not found.</p>'; return; }

  const fmt = (v, digits) => (v === null || v === undefined || v === "") ? "—" : (typeof v === "number" ? v.toFixed(digits) : String(v));
  for (const table of results.tables || []) {
    const wrap = document.createElement("div");
    wrap.className = "task-block";
    const cols = table.columns; // [{key, label, digits}]
    let html = `<h3>${table.title}</h3>`;
    if (table.note) html += `<p class="note">${table.note}</p>`;
    html += "<table><thead><tr><th>row</th>" + cols.map(c => `<th class="num">${c.label}</th>`).join("") + "</tr></thead><tbody>";
    for (const row of table.rows) {
      const cls = row.kind === "ours" ? "ours" : row.kind === "base" ? "base" : "";
      html += `<tr class="${cls}"><td>${row.label}${row.diagnostic ? ' <span class="diag">(diagnostic)</span>' : ""}${row.source ? ` <span class="diag">${row.source}</span>` : ""}</td>`
        + cols.map(c => `<td class="num">${fmt(row.values[c.key], c.digits ?? 3)}</td>`).join("") + "</tr>";
    }
    html += "</tbody></table>";
    wrap.innerHTML = html;
    resEl.appendChild(wrap);
  }
  const p = results.inference_protocol;
  if (p) {
    protoEl.innerHTML = `<b>Inference protocol for every "ours" row:</b> ${p.num_inference_steps} denoising steps · CFG ${p.guidance_scale} · seed ${p.seed} · ${p.canvas_resolution}² canvas · ${p.native_resolution_scoring ? "scored at native GT resolution" : "scored on the canvas"}. ${p.note || ""}`;
  }
  if (results.generated_at) protoEl.innerHTML += `<br><span>results.json generated ${results.generated_at} from ${results.source_count || "?"} evaluation JSON(s), commit ${results.commit || "?"}.</span>`;
  if (results.report_url) document.getElementById("report-link").href = results.report_url;
})();
