/* Open-Vision-Banana showcase. All imagery is real model output listed in data/gallery.json;
   all numbers come from data/results.json (built by scripts/render_site_data.py).
   The relighting and the point cloud are rendered here, in the browser, from those outputs. */
(async function () {
  const $ = (s, el = document) => el.querySelector(s);
  const load = async (p) => { try { const r = await fetch(p); return r.ok ? r.json() : null; } catch { return null; } };
  const loadImg = (src) => new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = bad; i.src = src; });
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [R, G] = await Promise.all([load("data/results.json"), load("data/gallery.json")]);
  if (!G) return;

  // --------------------------------------------------------------- WebGL helpers
  function glProgram(gl, vs, fs) {
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  function texture(gl, img, unit) {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  const onVisible = (el, fn, once = true) => {
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { fn(true); if (once) io.disconnect(); } else if (!once) fn(false); }), { threshold: 0.25 });
    io.observe(el);
  };
  const thumbRow = (el, items, key, pick) => {
    el.innerHTML = "";
    items.forEach((it, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("aria-pressed", String(i === 0)); b.setAttribute("aria-label", `example ${i + 1}`);
      b.innerHTML = `<img src="${it[key]}" alt="" loading="lazy">`;
      b.addEventListener("click", () => { el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); pick(it, i); });
      el.appendChild(b);
    });
  };

  // --------------------------------------------------------------- 1 · relight
  {
    const canvas = $("#relight");
    const gl = canvas.getContext("webgl", { antialias: true, premultipliedAlpha: false });
    const VS = `attribute vec2 p; varying vec2 uv; void main(){ uv = p * .5 + .5; gl_Position = vec4(p, 0., 1.); }`;
    // Decode the paper's normal code: R=(1-x)/2, G=(1+y)/2, B=(1+z)/2  ->  x=1-2R, y=2G-1, z=2B-1.
    const FS = `precision highp float; varying vec2 uv;
      uniform sampler2D N, A; uniform vec3 L; uniform float mode, blend;
      vec3 shade(vec3 n, vec3 alb){
        vec3 l = normalize(vec3(L.xy - uv, L.z));
        vec3 f = normalize(vec3(-l.xy, .55));
        float d = max(dot(n, l), 0.), d2 = max(dot(n, f), 0.);
        float s = pow(max(dot(n, normalize(l + vec3(0., 0., 1.))), 0.), 42.);
        float rim = pow(1. - max(n.z, 0.), 3.);
        vec3 key = vec3(1., .86, .55), fill = vec3(.38, .42, 1.);
        return alb * (.035 + .95 * d * d * key + .32 * d2 * fill) + .35 * s * key + .12 * rim * fill;
      }
      void main(){
        vec3 nc = texture2D(N, uv).rgb, ph = texture2D(A, uv).rgb;
        vec3 n = normalize(vec3(1. - 2. * nc.r, 2. * nc.g - 1., 2. * nc.b - 1.));
        vec3 c0 = shade(n, vec3(.92, .9, .88));
        vec3 c1 = shade(n, pow(ph, vec3(1.2)) * 1.25);
        vec3 a = mode < .5 ? c0 : mode < 1.5 ? c1 : mode < 2.5 ? nc : ph;
        gl_FragColor = vec4(a, 1.);
      }`;
    if (!gl) { canvas.replaceWith(Object.assign(new Image(), { src: G.normal[0].output, alt: "model output" })); }
    else {
      const prog = glProgram(gl, VS, FS);
      gl.useProgram(prog);
      const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      const U = (n) => gl.getUniformLocation(prog, n);
      gl.uniform1i(U("N"), 0); gl.uniform1i(U("A"), 1);
      let mode = 0, light = [0.78, 0.8, 0.32], target = light.slice(), lastMove = -1e9, ready = false;
      // The headline is lit by the same light. It follows the cursor without the canvas clamp, so the type lights up under the pointer.
      const h1 = $("#headline"); let typeLight = [0.78, 0.8], typeTarget = typeLight.slice();
      const lightType = () => {
        const c = canvas.getBoundingClientRect(), h = h1.getBoundingClientRect();
        h1.style.setProperty("--lx", `${(c.left + typeLight[0] * c.width - h.left).toFixed(1)}px`);
        h1.style.setProperty("--ly", `${(c.top + (1 - typeLight[1]) * c.height - h.top).toFixed(1)}px`);
      };
      const paint = () => {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.uniform3fv(U("L"), light); gl.uniform1f(U("mode"), mode);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      };
      const setScene = async (it) => {
        const [n, a] = await Promise.all([loadImg(it.output), loadImg(it.input)]);
        texture(gl, n, 0); texture(gl, a, 1); ready = true; paint(); lightType();
      };
      const order = [3, 5, 4, 0, 1, 2].filter((i) => G.normal[i]);
      const items = order.map((i) => G.normal[i]);
      thumbRow($("#relight-thumbs"), items, "input", (it) => setScene(it));
      await setScene(items[0]);
      const toUV = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height]; };
      window.addEventListener("pointermove", (e) => {
        const [u, v] = toUV(e);
        target = [Math.max(-0.6, Math.min(1.6, u)), Math.max(-0.6, Math.min(1.6, v)), 0.32];
        typeTarget = [u, v];
        lastMove = performance.now();
      });
      document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => {
        mode = +b.dataset.mode; paint();
        document.querySelectorAll(".modes button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      }));
      let visible = true;
      onVisible(canvas, (v) => { visible = v; }, false);
      const frame = (t) => {
        if (visible && ready) {
          if (t - lastMove > 2500 && !reduced) {
            const a = t / 2600;
            target = [0.5 + 0.55 * Math.cos(a), 0.5 + 0.45 * Math.sin(a * 1.3), 0.32];
            typeTarget = [target[0], target[1]];
          }
          for (let i = 0; i < 3; i++) light[i] += (target[i] - light[i]) * 0.12;
          for (let i = 0; i < 2; i++) typeLight[i] += (typeTarget[i] - typeLight[i]) * 0.12;
          paint(); lightType();
        }
        requestAnimationFrame(frame);
      };
      requestAnimationFrame(frame);
    }
  }

  // --------------------------------------------------------------- 2 · scan
  {
    const tasks = [
      ["normal", "Surface normals", "Which way each surface faces"],
      ["depth", "Depth", "How far away each pixel is"],
      ["semantic", "Segmentation", "Which class each pixel belongs to"],
    ];
    const box = $("#scanner"), under = $("#scan-under"), over = $("#scan-over"), gt = $("#scan-gt");
    let task = "normal", item = G.normal[0];
    const set = (x) => { box.style.setProperty("--x", x + "%"); $("#scan-handle").setAttribute("aria-valuenow", Math.round(x)); };
    const show = () => {
      under.src = item.input; over.src = gt.checked ? item.target : item.output;
      $("#scan-tag").textContent = gt.checked ? "Ground truth" : "Model output";
    };
    const pickTask = (t) => {
      task = t;
      const items = G[t].filter((it) => !it.inverted);
      item = items[0];
      thumbRow($("#scan-thumbs"), items, "input", (it) => { item = it; show(); });
      show();
    };
    const seg = $("#scan-tasks");
    tasks.forEach(([t, label, what], i) => {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "tab"); b.setAttribute("aria-selected", String(i === 0)); b.innerHTML = `<b>${label}</b><span>${what}</span>`;
      b.addEventListener("click", () => { seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b))); pickTask(t); });
      seg.appendChild(b);
    });
    gt.addEventListener("change", show);
    pickTask("normal");
    let drag = false;
    const at = (e) => { const r = box.getBoundingClientRect(); set(Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100))); };
    box.addEventListener("pointerdown", (e) => { drag = true; box.setPointerCapture(e.pointerId); at(e); });
    box.addEventListener("pointermove", (e) => { if (drag || e.pointerType === "mouse") at(e); });
    box.addEventListener("pointerup", () => { drag = false; });
    box.addEventListener("pointercancel", () => { drag = false; });
    $("#scan-handle").addEventListener("keydown", (e) => {
      const cur = parseFloat(box.style.getPropertyValue("--x")) || 50;
      if (e.key === "ArrowLeft") set(Math.max(0, cur - 5)); if (e.key === "ArrowRight") set(Math.min(100, cur + 5));
    });
    set(50);
    if (!reduced) onVisible(box, () => {
      const t0 = performance.now(), path = (k) => (k < .45 ? 96 - 92 * (k / .45) : 4 + 46 * ((k - .45) / .55));
      const step = (t) => { const k = Math.min(1, (t - t0) / 2600); const e = k < .45 ? k : .45 + (1 - Math.pow(1 - (k - .45) / .55, 3)) * .55; set(path(e)); if (k < 1) requestAnimationFrame(step); };
      requestAnimationFrame(step);
    });
  }

  // --------------------------------------------------------------- 3 · point cloud
  {
    const canvas = $("#cloud");
    const gl = canvas.getContext("webgl", { antialias: true });
    const good = G.depth.filter((d) => !d.inverted);
    if (gl && good.length) {
      const VS = `attribute vec3 p; attribute vec3 c; uniform mat4 M; uniform float S; varying vec3 vc;
        void main(){ vec4 q = M * vec4(p, 1.); gl_Position = q; gl_PointSize = S / q.w; vc = c; }`;
      const FS = `precision mediump float; varying vec3 vc; void main(){ vec2 d = gl_PointCoord - .5; if (dot(d, d) > .25) discard; gl_FragColor = vec4(vc, 1.); }`;
      const prog = glProgram(gl, VS, FS); gl.useProgram(prog);
      const pb = gl.createBuffer(), cb = gl.createBuffer();
      const lp = gl.getAttribLocation(prog, "p"), lc = gl.getAttribLocation(prog, "c");
      let count = 0, yaw = 0, pitch = 0, userYaw = 0, userPitch = 0, dragging = false, lastUser = -1e9, last = [0, 0];
      const N = 256;
      const pixels = (img) => { const c = document.createElement("canvas"); c.width = c.height = N; const x = c.getContext("2d"); x.drawImage(img, 0, 0, N, N); return x.getImageData(0, 0, N, N).data; };
      const build = async (it) => {
        const [photo, depth] = await Promise.all([loadImg(it.input), loadImg(it.output)]);
        const P = pixels(photo), D = pixels(depth);
        const near = (i) => D[i * 4] / 255;
        const pos = [], col = [];
        for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
          const i = y * N + x, g = near(i);
          // drop pixels on depth discontinuities so edges do not smear into "curtains"
          const grad = Math.max(Math.abs(g - near(i + 1)), Math.abs(g - near(i - 1)), Math.abs(g - near(i + N)), Math.abs(g - near(i - N)));
          if (grad > 0.045) continue;
          const z = 1.0 + 2.2 * (1 - g);                         // brighter = closer
          const u = (x / N - 0.5), v = (0.5 - y / N);
          pos.push(u * z * 0.95, v * z * 0.95, -z);
          col.push(P[i * 4] / 255, P[i * 4 + 1] / 255, P[i * 4 + 2] / 255);
        }
        let mz = 0; for (let k = 2; k < pos.length; k += 3) mz += pos[k]; mz /= pos.length / 3;
        for (let k = 2; k < pos.length; k += 3) pos[k] -= mz;
        gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(col), gl.STATIC_DRAW);
        count = pos.length / 3;
      };
      const mul = (a, b) => { const o = new Float32Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; } return o; };
      const persp = (f, asp, n, fa) => { const t = 1 / Math.tan(f / 2); return new Float32Array([t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (fa + n) / (n - fa), -1, 0, 0, (2 * fa * n) / (n - fa), 0]); };
      const rotY = (a) => new Float32Array([Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, 0, 1]);
      const rotX = (a) => new Float32Array([1, 0, 0, 0, 0, Math.cos(a), Math.sin(a), 0, 0, -Math.sin(a), Math.cos(a), 0, 0, 0, 0, 1]);
      const trans = (x, y, z) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
      canvas.addEventListener("pointerdown", (e) => { dragging = true; last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
      canvas.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        userYaw += (e.clientX - last[0]) * 0.008; userPitch = Math.max(-0.9, Math.min(0.9, userPitch + (e.clientY - last[1]) * 0.006));
        last = [e.clientX, e.clientY]; lastUser = performance.now();
      });
      canvas.addEventListener("pointerup", () => { dragging = false; });
      canvas.addEventListener("pointercancel", () => { dragging = false; });
      const order = [2, 4, 0].map((r) => good.find((d) => d.row === r)).filter(Boolean).concat(good.filter((d) => ![2, 4, 0].includes(d.row)));
      thumbRow($("#cloud-thumbs"), order, "input", async (it) => { await build(it); draw(performance.now()); });
      await build(order[0]);
      let visible = false;
      onVisible(canvas, (v) => { visible = v; }, false);
      const draw = (t) => {
          const idle = t - lastUser > 3000 && !dragging && !reduced;
          if (idle) { userYaw *= 0.97; userPitch *= 0.97; }
          yaw = (idle ? 0.55 * Math.sin(t / 2200) : 0) + userYaw;
          pitch = (idle ? 0.12 * Math.sin(t / 3100) : 0) + userPitch;
          const dpr = canvas.width / canvas.clientWidth;
          gl.viewport(0, 0, canvas.width, canvas.height);
          gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
          const M = mul(persp(0.75, canvas.width / canvas.height, 0.1, 20), mul(trans(0, 0, -3.5), mul(rotX(pitch), rotY(yaw))));
          gl.uniformMatrix4fv(gl.getUniformLocation(prog, "M"), false, M);
          gl.uniform1f(gl.getUniformLocation(prog, "S"), 6.5 * Math.max(1, dpr));
          gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 0, 0);
          gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.enableVertexAttribArray(lc); gl.vertexAttribPointer(lc, 3, gl.FLOAT, false, 0, 0);
          gl.drawArrays(gl.POINTS, 0, count);
      };
      const frame = (t) => { if (visible && count) draw(t); requestAnimationFrame(frame); };
      draw(1500);   // first frame at an angle, even when animation frames are throttled
      requestAnimationFrame(frame);
    }

    // failures: the depth rows whose prediction is anti-correlated with the target
    const inv = G.depth.filter((d) => d.inverted);
    const absr = Math.min(...G.depth.map((d) => Math.abs(d.pearson_r)));
    const v = (run, k) => R.runs[run].values[k].v;
    $("#fail-text").textContent = `On all ${G.depth.length} held-out images the predicted depth tracks the truth with |r| ≥ ${absr.toFixed(2)} — the shape is right. On ${inv.length} of them the sign is flipped: near painted as far. Grayscale depth has no fixed polarity. Training on the paper's single cube-edge code instead of five colormaps took NYU δ1 from ${v("t2", "nyu").toFixed(3)} to ${v("c1", "nyu").toFixed(3)} on Qwen-Image-2.1 (runs T2 → C1).`;
    $("#fail-row").innerHTML = inv.map((d) => `<div class="fcard"><div class="pair"><img src="${d.target}" alt="ground-truth depth" loading="lazy"><img src="${d.output}" alt="predicted depth" loading="lazy"></div>
      <div class="cap"><span>Ground truth</span><span>Model</span></div><div class="lab"><span>Pearson r</span><b>${d.pearson_r.toFixed(2)}</b></div></div>`).join("");
  }

  // --------------------------------------------------------------- 4 · encodings
  {
    const K = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]];
    const cube = (f) => { const s = Math.min(6.9999, Math.max(0, f * 7)), i = Math.floor(s), t = s - i; return K[i].map((a, j) => a + (K[i + 1][j] - a) * t); };
    const lam = -3, c = 10 / 3;
    const f = (d) => 1 - Math.pow(1 - d / (lam * c), lam + 1);
    const slider = $("#d-slider");
    const upd = () => {
      const d = 0.1 * Math.pow(10, (3 * slider.value) / 1000), fv = f(d), rgb = cube(fv).map((x) => Math.round(x * 255));
      $("#d-swatch").style.background = `rgb(${rgb})`;
      $("#d-m").textContent = d < 10 ? `${d.toFixed(2)} m` : `${d.toFixed(1)} m`;
      $("#d-rgb").textContent = `f(d) = ${fv.toFixed(3)}  →  rgb(${rgb.join(", ")})`;
      $("#d-mark").style.left = `${fv * 100}%`;
    };
    slider.addEventListener("input", upd); upd();
    if (!reduced) onVisible(slider, () => {
      const t0 = performance.now();
      const step = (t) => { const k = Math.min(1, (t - t0) / 3000); slider.value = Math.round(1000 * (0.5 - 0.5 * Math.cos(k * Math.PI * 2)) * 0.95 + 25); upd(); if (k < 1) requestAnimationFrame(step); else { slider.value = 520; upd(); } };
      requestAnimationFrame(step);
    });

    const sp = $("#sphere"), x2 = sp.getContext("2d"), W = sp.width, r = W / 2 - 4;
    const img = x2.createImageData(W, W);
    for (let j = 0; j < W; j++) for (let i = 0; i < W; i++) {
      const x = (i - W / 2) / r, y = (W / 2 - j) / r, q = x * x + y * y, o = (j * W + i) * 4;
      if (q > 1) continue;
      const z = Math.sqrt(1 - q);
      img.data[o] = 255 * (1 - x) / 2; img.data[o + 1] = 255 * (1 + y) / 2; img.data[o + 2] = 255 * (1 + z) / 2; img.data[o + 3] = 255;
    }
    x2.putImageData(img, 0, 0);
    sp.addEventListener("pointermove", (e) => {
      const b = sp.getBoundingClientRect(), x = ((e.clientX - b.left) / b.width - 0.5) * 2 * (W / 2) / r, y = (0.5 - (e.clientY - b.top) / b.height) * 2 * (W / 2) / r;
      if (x * x + y * y > 1) { $("#n-read").textContent = "Hover the sphere"; return; }
      const z = Math.sqrt(1 - x * x - y * y), rgb = [(1 - x) / 2, (1 + y) / 2, (1 + z) / 2].map((v) => Math.round(v * 255));
      $("#n-read").innerHTML = `n = (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}) → <span style="color:rgb(${rgb})">■</span> rgb(${rgb.join(", ")})`;
    });
  }

  // --------------------------------------------------------------- 5 · numbers
  if (R) {
    const runs = R.runs, variants = new Set(R.protocol_variants);
    const evaluated = Object.keys(runs).filter((k) => !variants.has(k));
    const best = (k) => evaluated.filter((r) => r !== "base" && runs[r].values[k]).reduce((a, r) => (runs[r].values[k].v > runs[a].values[k].v ? r : a), "e2b");
    $("#num-sub").textContent = `${evaluated.length} Qwen-Image-2.1 runs, scored on full official splits with identical prompts, decoders and inference settings. ${R.source_count} evaluation files stand behind this page.`;
    // One row per benchmark. The axis runs from 0 to the paper's score, so the bar is how much of the gap fine-tuning closed.
    const pct = (v, b) => `${Math.min(100, (100 * v) / b.paper).toFixed(2)}%`;
    const rows = R.benchmarks.map((b) => {
      const r = best(b.key), v = runs[r].values[b.key].v, v0 = runs.base.values[b.key].v;
      return `<div class="row" style="--from:${pct(v0, b)};--to:${pct(v, b)}">
        <div class="name"><b>${b.label}${b.key === "reasonseg" ? "*" : ""}</b><span>${b.metric}, best run: ${runs[r].label.replace(/ · /g, ", ")}</span></div>
        <div class="track" role="img" aria-label="${b.label}: ${v0.toFixed(b.digits)} untuned, ${v.toFixed(b.digits)} fine-tuned, ${b.paper} in the paper"><i class="gain"></i><i class="dot base"></i><i class="dot ours"></i></div>
        <div class="val"><b>${v.toFixed(b.digits)}</b> <span>of ${b.paper}</span><small>up from ${v0.toFixed(b.digits)}</small></div>
      </div>`;
    }).join("");
    const chart = $("#chart");
    chart.innerHTML = rows + `<p class="key"><span><i class="k-base"></i>Qwen-Image-2.1, untuned</span><span><i class="k-ours"></i>Best fine-tuned run</span><span><i class="k-paper"></i>Vision Banana (paper, closed Nano Banana Pro base)</span><span>*ReasonSeg without the paper's Gemini query rewrite</span></p>`;
    if (reduced) chart.classList.add("in"); else onVisible(chart, () => chart.classList.add("in"));
    const a = R.audit;
    $("#audit").innerHTML = `Of ${a.checkable} statements in the paper that can be checked against code, <b>${a.identical} are implemented identically and ${a.errors} are wrong</b>. ${a.partial} are places where the paper is silent, and ${a.missing} depend on things that are closed.`;

    const st = Object.fromEntries(R.status.map((s) => [s.id, s]));
    $("#road").innerHTML = `
      <li class="done"><b>${evaluated.length} runs measured</b><span>Six-task main run, data and mixing ablations, task composition, LoRA against full fine-tuning.</span></li>
      <li class="live"><b>In progress</b><span>${st.R1 ? "First public checkpoint (E2b merged into Qwen-Image-2.1). " : ""}Learning-rate scan for full fine-tuning and a resolution study for segmentation.</span></li>
      <li><b>Next</b><span>A Qwen-Image-2.1 gallery (E5), robustness to prompt style, and training on SenseNova-Vision-Corpus-50M.</span></li>`;
    $("#prov").textContent = `Rendered ${R.generated_at} from ${R.source_count} evaluation files${R.commit ? ` at ${R.commit}` : ""}.`;
  }

  // --------------------------------------------------------------- scroll depth, in the depth ramp
  {
    const bar = $("#depthbar"); let queued = false;
    const upd = () => { queued = false; const max = document.documentElement.scrollHeight - innerHeight; bar.style.setProperty("--p", `${max > 0 ? Math.min(100, (100 * scrollY) / max).toFixed(2) : 0}%`); };
    const ask = () => { if (!queued) { queued = true; requestAnimationFrame(upd); } };
    addEventListener("scroll", ask, { passive: true }); addEventListener("resize", ask); upd();
  }
})();
