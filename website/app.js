/* Open-Vision-Banana project page: the case viewer and the results are rendered from data/*.json (no build step).
   Every picture is a model output listed in data/gallery.json; relighting and the point cloud are computed
   here, in the browser, from those outputs. */
(async function () {
  const $ = (s, el = document) => el.querySelector(s);
  const load = async (p) => { try { const r = await fetch(p); return r.ok ? r.json() : null; } catch { return null; } };
  const loadImg = (src) => new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = bad; i.src = src; });
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [R, G] = await Promise.all([load("data/results.json"), load("data/gallery.json")]);

  // --------------------------------------------------------------- helpers
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
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { fn(true); if (once) io.disconnect(); } else if (!once) fn(false); }), { threshold: 0.2 });
    io.observe(el);
  };
  const button = (label, attrs = {}) => { const b = document.createElement("button"); b.type = "button"; b.textContent = label; for (const [k, v] of Object.entries(attrs)) b.setAttribute(k, v); return b; };

  // --------------------------------------------------------------- the selected cases
  // One task at a time, every case of that task side by side. The order of the views is the order things
  // happen in: the photo, what the model painted, the ground truth, and then what the painted picture is good for.
  if (G) {
    const TASKS = [
      ["semantic", "Semantic", "Semantic segmentation", "Open-vocabulary; the legend lives in the prompt (four styles)."],
      ["instance", "Instance", "Instance segmentation", "One class at a time; the model picks a colour per instance; Appendix A decoder."],
      ["referring", "Referring", "Referring expression segmentation", "Fig. 4 template, white target on black."],
      ["depth", "Depth", "Monocular metric depth", "Power transform + cube-edge colormap; decoded to metres without camera intrinsics."],
      ["normal", "Normals", "Surface normal estimation", "Camera-frame normals mapped linearly to RGB."],
    ].filter(([t]) => (G[t] || []).length);            // a task shows up here once data/gallery.json lists pictures for it
    const LABEL = { input: "Input photo", output: "Model output", target: "Ground truth", relit: "Relight", cloud: "Lift to 3D" };
    const HINT = {
      input: "The photos the model was given.",
      output: "Drag across a picture: the photo is left of the line, what the model painted is right of it.",
      target: "Drag across a picture: the photo is left of the line, the ground truth is right of it.",
      relit: "Move the cursor: it is the light. Each scene is shaded in the browser with the normals the model painted.",
      cloud: "Drag a picture to orbit. Each point is a pixel of the photo, pushed back by the depth the model painted.",
    };
    const glOK = !!document.createElement("canvas").getContext("webgl");
    const viewsFor = (t) => ["input", "output", "target", ...(glOK && t === "normal" ? ["relit"] : []), ...(glOK && t === "depth" ? ["cloud"] : [])];
    const grid = $("#case-grid"), tasksEl = $("#case-tasks"), viewsEl = $("#case-views"), modesEl = $("#relit-modes");
    let task = null, view = "output", cards = [], relitMode = 0;

    // ---- relight: a scene shaded with the normals the model painted; the cursor is the light
    const RELIT_VS = `attribute vec2 p; varying vec2 uv; void main(){ uv = p * .5 + .5; gl_Position = vec4(p, 0., 1.); }`;
    // Decode the paper's normal code: R=(1-x)/2, G=(1+y)/2, B=(1+z)/2  ->  x=1-2R, y=2G-1, z=2B-1.
    const RELIT_FS = `precision highp float; varying vec2 uv;
      uniform sampler2D N, A; uniform vec3 L; uniform float mode;
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
        gl_FragColor = vec4(mode < .5 ? shade(n, vec3(.92, .9, .88)) : shade(n, pow(ph, vec3(1.2)) * 1.25), 1.);
      }`;
    const makeRelit = (canvas, it) => {
      const gl = canvas.getContext("webgl", { antialias: true, premultipliedAlpha: false });
      if (!gl) return null;
      const prog = glProgram(gl, RELIT_VS, RELIT_FS); gl.useProgram(prog);
      const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      const uL = gl.getUniformLocation(prog, "L"), uMode = gl.getUniformLocation(prog, "mode");
      gl.uniform1i(gl.getUniformLocation(prog, "N"), 0); gl.uniform1i(gl.getUniformLocation(prog, "A"), 1);
      let ready = false, dead = false;
      Promise.all([loadImg(it.output), loadImg(it.input)]).then(([n, a]) => { if (dead) return; texture(gl, n, 0); texture(gl, a, 1); ready = true; }).catch(() => {});
      return {
        paint(light, mode) { if (!ready) return; gl.viewport(0, 0, canvas.width, canvas.height); gl.uniform3fv(uL, light); gl.uniform1f(uMode, mode); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); },
        dispose() { dead = true; const x = gl.getExtension("WEBGL_lose_context"); if (x) x.loseContext(); },
      };
    };

    // ---- point cloud: each point is a pixel of the photo, pushed back by the depth the model painted
    const mul = (a, b) => { const o = new Float32Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; } return o; };
    const persp = (f, asp, n, fa) => { const t = 1 / Math.tan(f / 2); return new Float32Array([t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (fa + n) / (n - fa), -1, 0, 0, (2 * fa * n) / (n - fa), 0]); };
    const rotY = (a) => new Float32Array([Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, 0, 1]);
    const rotX = (a) => new Float32Array([1, 0, 0, 0, 0, Math.cos(a), Math.sin(a), 0, 0, -Math.sin(a), Math.cos(a), 0, 0, 0, 0, 1]);
    const trans = (x, y, z) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
    const N = 256;
    const pixels = (img) => { const c = document.createElement("canvas"); c.width = c.height = N; const x = c.getContext("2d"); x.drawImage(img, 0, 0, N, N); return x.getImageData(0, 0, N, N).data; };
    const makeCloud = (canvas, it, phase) => {
      const gl = canvas.getContext("webgl", { antialias: true });
      if (!gl) return null;
      const VS = `attribute vec3 p; attribute vec3 c; uniform mat4 M; uniform float S; varying vec3 vc;
        void main(){ vec4 q = M * vec4(p, 1.); gl_Position = q; gl_PointSize = S / q.w; vc = c; }`;
      const FS = `precision mediump float; varying vec3 vc; void main(){ vec2 d = gl_PointCoord - .5; if (dot(d, d) > .25) discard; gl_FragColor = vec4(vc, 1.); }`;
      const prog = glProgram(gl, VS, FS); gl.useProgram(prog);
      const pb = gl.createBuffer(), cb = gl.createBuffer(), lp = gl.getAttribLocation(prog, "p"), lc = gl.getAttribLocation(prog, "c");
      const uM = gl.getUniformLocation(prog, "M"), uS = gl.getUniformLocation(prog, "S");
      let count = 0, dead = false, userYaw = 0, userPitch = 0, dragging = false, lastUser = -1e9, last = [0, 0];
      canvas.addEventListener("pointerdown", (e) => { dragging = true; last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
      canvas.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        userYaw += (e.clientX - last[0]) * 0.008; userPitch = Math.max(-0.9, Math.min(0.9, userPitch + (e.clientY - last[1]) * 0.006));
        last = [e.clientX, e.clientY]; lastUser = performance.now();
      });
      canvas.addEventListener("pointerup", () => { dragging = false; });
      canvas.addEventListener("pointercancel", () => { dragging = false; });
      Promise.all([loadImg(it.input), loadImg(it.output)]).then(([photo, depth]) => {
        if (dead) return;
        const P = pixels(photo), D = pixels(depth), near = (i) => D[i * 4] / 255, pos = [], col = [];
        for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
          const i = y * N + x, g = near(i);
          // drop pixels on depth discontinuities so edges do not smear into "curtains"
          const grad = Math.max(Math.abs(g - near(i + 1)), Math.abs(g - near(i - 1)), Math.abs(g - near(i + N)), Math.abs(g - near(i - N)));
          if (grad > 0.045) continue;
          const z = 1.0 + 2.2 * (1 - g);                         // brighter = closer
          pos.push((x / N - 0.5) * z * 0.95, (0.5 - y / N) * z * 0.95, -z);
          col.push(P[i * 4] / 255, P[i * 4 + 1] / 255, P[i * 4 + 2] / 255);
        }
        let mz = 0; for (let k = 2; k < pos.length; k += 3) mz += pos[k]; mz /= pos.length / 3;
        for (let k = 2; k < pos.length; k += 3) pos[k] -= mz;
        gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(col), gl.STATIC_DRAW);
        count = pos.length / 3;
      }).catch(() => {});
      return {
        draw(t) {
          if (!count || !canvas.clientWidth) return;
          const idle = t - lastUser > 3000 && !dragging;
          if (idle) { userYaw *= 0.97; userPitch *= 0.97; }
          const sway = idle && !reduced;
          const yaw = (sway ? 0.55 * Math.sin(t / 2200 + phase) : reduced ? 0.4 : 0) + userYaw, pitch = (sway ? 0.12 * Math.sin(t / 3100 + phase) : 0) + userPitch;
          gl.viewport(0, 0, canvas.width, canvas.height);
          gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
          gl.uniformMatrix4fv(uM, false, mul(persp(0.82, canvas.width / canvas.height, 0.1, 20), mul(trans(0, 0, -3.5), mul(rotX(pitch), rotY(yaw)))));
          gl.uniform1f(uS, 6.5 * Math.max(1, canvas.width / canvas.clientWidth));
          gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 0, 0);
          gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.enableVertexAttribArray(lc); gl.vertexAttribPointer(lc, 3, gl.FLOAT, false, 0, 0);
          gl.drawArrays(gl.POINTS, 0, count);
        },
        dispose() { dead = true; const x = gl.getExtension("WEBGL_lose_context"); if (x) x.loseContext(); },
      };
    };

    // ---- one card per case: a wipe between the photo and the picture, swapped for a canvas when relit or lifted
    const makeCard = (it, i) => {
      const el = document.createElement("figure"); el.className = "case";
      let tag = "";
      if (typeof it.pearson_r === "number") {
        const r = it.pearson_r.toFixed(2).replace("-", "−");
        tag = `<span class="tag${it.inverted ? " flag" : ""}" title="Pearson correlation of the painted depth with ground truth${it.inverted ? ". The sign is flipped: near is painted as far." : ""}">${it.inverted ? "Sign flipped, " : ""}r = ${r}</span>`;
      }
      el.innerHTML = `<div class="scanner"><img class="under" src="${it.input}" alt="Case ${i + 1}, input photo"><img class="over" alt="">
        <div class="handle" role="slider" tabindex="0" aria-label="Reveal case ${i + 1}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50"><span></span></div></div>${tag}`;
      const scanner = $(".scanner", el), handle = $(".handle", el);
      const c = { it, el, over: $(".over", el), canvas: null, gl: null, kind: null, drag: false };
      c.setX = (x) => { scanner.style.setProperty("--x", x + "%"); handle.setAttribute("aria-valuenow", Math.round(x)); };
      const at = (e) => { const r = scanner.getBoundingClientRect(); c.setX(Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100))); };
      scanner.addEventListener("pointerdown", (e) => { c.drag = true; scanner.setPointerCapture(e.pointerId); at(e); });
      scanner.addEventListener("pointermove", (e) => { if (c.drag || e.pointerType === "mouse") at(e); });
      scanner.addEventListener("pointerup", () => { c.drag = false; });
      scanner.addEventListener("pointercancel", () => { c.drag = false; });
      handle.addEventListener("keydown", (e) => {
        const cur = parseFloat(scanner.style.getPropertyValue("--x")) || 50;
        if (e.key === "ArrowLeft") c.setX(Math.max(0, cur - 5)); if (e.key === "ArrowRight") c.setX(Math.min(100, cur + 5));
      });
      c.setX(50);
      return c;
    };
    const apply = (c, i) => {
      let v = view;
      if (v === "cloud" && c.it.inverted) v = "output";          // a flipped depth map is shown as painted, not lifted
      const want = v === "relit" || v === "cloud" ? v : null;
      if (want && c.kind !== want) {
        if (c.gl) c.gl.dispose(); if (c.canvas) c.canvas.remove();
        c.canvas = document.createElement("canvas"); c.canvas.width = c.canvas.height = 768; c.canvas.className = want;
        c.canvas.setAttribute("aria-label", want === "relit" ? `Case ${i + 1}, relit with the painted normals` : `Case ${i + 1}, point cloud from the painted depth`);
        c.el.appendChild(c.canvas);
        c.gl = want === "relit" ? makeRelit(c.canvas, c.it) : makeCloud(c.canvas, c.it, i * 1.7);
        c.kind = c.gl ? want : null;
      }
      if (want && !c.gl) v = "output";
      if (v === "output" || v === "target") { c.over.src = c.it[v]; c.over.alt = LABEL[v].toLowerCase(); }
      c.el.dataset.view = v;
    };
    const refresh = () => {
      const [, , full, blurb] = TASKS.find((x) => x[0] === task);
      cards.forEach(apply);
      modesEl.hidden = view !== "relit";
      viewsEl.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
      const flipped = cards.filter((c) => c.it.inverted).length;
      $("#case-what").innerHTML = `<b>${full}.</b> ${blurb} ${HINT[view]}${view === "cloud" && flipped ? ` ${flipped} of the ${cards.length} have the depth sign flipped and stay as painted.` : ""}`;
      const it = cards[0].it;
      $("#case-meta").textContent = [it.backbone, it.checkpoint, it.split, it.prompt && `prompt: ${it.prompt}`, it.seed != null && `seed ${it.seed}`].filter(Boolean).join(", ").replace(/ · /g, ", ");
    };
    const setTask = (t) => {
      task = t; view = "output";
      cards.forEach((c) => { if (c.gl) c.gl.dispose(); });
      grid.innerHTML = "";
      cards = (G[t] || []).map(makeCard);
      cards.forEach((c) => grid.appendChild(c.el));
      tasksEl.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.task === t)));
      viewsEl.innerHTML = "";
      for (const v of viewsFor(t)) {
        const b = button(LABEL[v], { "data-view": v });
        if (v === "relit" || v === "cloud") b.classList.add("extra");
        b.addEventListener("click", () => { view = v; refresh(); });
        viewsEl.appendChild(b);
      }
      refresh();
    };
    TASKS.forEach(([t, short, full]) => {
      const b = button(short, { role: "tab", "data-task": t, title: full });
      b.addEventListener("click", () => setTask(t));
      tasksEl.appendChild(b);
    });
    modesEl.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      relitMode = +b.dataset.mode;
      modesEl.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    }));
    if (TASKS.length) setTask(TASKS[0][0]); else $("#cases").hidden = true;

    // the first time the cases come into view, the line sweeps across each one in turn
    if (!reduced && TASKS.length) onVisible(grid, () => cards.forEach((c, i) => {
      const t0 = performance.now() + i * 140, path = (k) => (k < .45 ? 96 - 92 * (k / .45) : 4 + 46 * ((k - .45) / .55));
      const step = (t) => {
        const k = Math.max(0, Math.min(1, (t - t0) / 2400)), e = k < .45 ? k : .45 + (1 - Math.pow(1 - (k - .45) / .55, 3)) * .55;
        c.setX(path(e)); if (k < 1 && !c.drag) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }));

    // ---- one light, in page coordinates, for every relit case and for the title
    const h1 = $("#headline");
    let light = [innerWidth * 0.7, innerHeight * 0.2], target = light.slice(), lastMove = -1e9, visible = false;
    window.addEventListener("pointermove", (e) => { target = [e.clientX, e.clientY]; lastMove = performance.now(); });
    onVisible(grid, (v) => { visible = v; }, false);
    const tick = (t) => {
      if (t - lastMove > 2500 && !reduced) {
        const a = t / 2600;
        target = [innerWidth * (0.5 + 0.42 * Math.cos(a)), innerHeight * (0.5 + 0.36 * Math.sin(a * 1.3))];
      }
      for (let i = 0; i < 2; i++) light[i] += (target[i] - light[i]) * 0.12;
      const h = h1.getBoundingClientRect();
      h1.style.setProperty("--lx", `${(light[0] - h.left).toFixed(1)}px`);
      h1.style.setProperty("--ly", `${(light[1] - h.top).toFixed(1)}px`);
      if (visible) for (const c of cards) {
        if (!c.gl) continue;
        if (c.el.dataset.view === "relit") {
          const r = c.el.getBoundingClientRect(), clamp = (x) => Math.max(-2.5, Math.min(3.5, x));
          c.gl.paint([clamp((light[0] - r.left) / r.width), clamp(1 - (light[1] - r.top) / r.height), 0.32], relitMode);
        } else if (c.el.dataset.view === "cloud") c.gl.draw(t);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // --------------------------------------------------------------- encodings
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
      $("#d-rgb").textContent = `f(d) = ${fv.toFixed(3)} → rgb(${rgb.join(", ")})`;
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

  // --------------------------------------------------------------- pipeline: the ramp runs through the five steps once
  { const flow = $("#flow"); if (reduced) flow.classList.add("in"); else onVisible(flow, () => flow.classList.add("in")); }

  // --------------------------------------------------------------- results, straight from data/results.json
  if (R && R.tables) {
    const fmt = (v, d) => (v === undefined || v === null || v === "" ? "—" : typeof v === "number" ? v.toFixed(d) : String(v));
    // One row per higher-is-better column that has an untuned base, a fine-tuned run and the paper's number.
    // The axis runs from 0 to the paper's score, so the bar is how much of the gap fine-tuning closed.
    const rows = [], pendingTables = [];
    for (const t of R.tables) {
      const short = t.title.split(" (")[0];
      if (t.rows.some((r) => r.kind === "ours") && !t.rows.some((r) => r.kind === "ours" && Object.values(r.values).some((v) => typeof v === "number"))) pendingTables.push(short);
      for (const c of t.columns) {
        if (!/↑/.test(c.label)) continue;
        const num = (r) => (r && typeof r.values[c.key] === "number" ? r.values[c.key] : null);
        const base = num(t.rows.find((r) => r.kind === "base"));
        const paper = num(t.rows.find((r) => r.kind === "paper" && /Vision Banana/.test(r.label)));
        const ours = t.rows.filter((r) => r.kind === "ours" && num(r) !== null);
        if (base === null || paper === null || !ours.length) continue;
        const top = ours.reduce((a, r) => (num(r) > num(a) ? r : a));
        const m = c.label.replace(/\s*↑\s*$/, "").match(/^(.*)\s(\S+)$/);
        rows.push({ name: m ? m[1] : c.label, metric: m ? m[2] : "", d: c.digits ?? 2, base, v: num(top), run: top.label, paper, diag: /ReasonSeg/.test(c.label) });
      }
    }
    const pct = (v, r) => `${Math.max(0, Math.min(100, (100 * v) / r.paper)).toFixed(2)}%`;
    const chart = $("#chart");
    chart.innerHTML = rows.map((r) => `<div class="row" style="--from:${pct(r.base, r)};--to:${pct(r.v, r)}">
        <div class="name"><b>${r.name}${r.diag ? "*" : ""}</b><span>${r.metric}, best run: ${r.run.replace(/ · /g, ", ")}</span></div>
        <div class="track" role="img" aria-label="${r.name} ${r.metric}: ${fmt(r.base, r.d)} untuned, ${fmt(r.v, r.d)} fine-tuned, ${fmt(r.paper, r.d)} in the paper"><i class="gain"></i><i class="dot base"></i><i class="dot ours"></i></div>
        <div class="val"><b>${fmt(r.v, r.d)}</b> <span>of ${fmt(r.paper, r.d)}</span><small>${r.v >= r.base ? "up" : "down"} from ${fmt(r.base, r.d)}</small></div>
      </div>`).join("")
      + `<p class="key"><span><i class="k-base"></i>Untuned base</span><span><i class="k-ours"></i>Best fine-tuned run</span><span><i class="k-paper"></i>Vision Banana (paper, closed base model)</span>`
      + (rows.some((r) => r.diag) ? `<span>*ReasonSeg uses the original queries, so it is diagnostic</span>` : "")
      + pendingTables.map((n) => `<span>${n}: not scored yet</span>`).join("") + `</p>`;
    if (reduced) chart.classList.add("in"); else onVisible(chart, () => chart.classList.add("in"));

    // the last step of the pipeline strip shows three measured values
    const picks = ["NYUv2", "Cityscapes", "RefCOCOg"].map((n) => rows.find((r) => r.name === n)).filter(Boolean);
    if (picks.length === 3) $("#flow-score").innerHTML = picks.map((r) => `${r.metric.replace("δ1", "δ<sub>1</sub>")} ${fmt(r.v, r.d)}`).join("<br>");

    // Every row. Rows are marked like the chart's key; the best fine-tuned value in each column is picked out.
    const tableHTML = (t) => {
      const best = Object.fromEntries(t.columns.map((c) => {
        const vs = t.rows.filter((r) => r.kind === "ours").map((r) => r.values[c.key]).filter((v) => typeof v === "number");
        return [c.key, vs.length > 1 ? (/↓/.test(c.label) ? Math.min(...vs) : Math.max(...vs)) : null];
      }));
      const body = t.rows.map((r) => `<tr class="${r.kind}"><td><span><i></i>${r.label}${r.diagnostic ? " <em>diagnostic</em>" : ""}</span></td>${t.columns.map((c) => {
        const v = r.values[c.key];
        return v === undefined || v === null || v === "" ? '<td class="na">—</td>' : `<td${r.kind === "ours" && v === best[c.key] ? ' class="best"' : ""}>${fmt(v, c.digits ?? 3)}</td>`;
      }).join("")}</tr>`).join("");
      return `<div class="table-wrap"><table class="data"><thead><tr><th scope="col">${t.title}</th>${t.columns.map((c) => `<th scope="col">${c.label}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>${t.note ? `<p class="fine">${t.note}</p>` : ""}`;
    };
    const tabs = $("#table-tabs");
    R.tables.forEach((t, i) => {
      const b = button(t.title.split(" (")[0].split(" — ")[0], { role: "tab", "aria-selected": String(i === 0) });
      b.addEventListener("click", () => { tabs.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b))); $("#table-view").innerHTML = tableHTML(t); });
      tabs.appendChild(b);
    });
    if (R.tables.length) $("#table-view").innerHTML = tableHTML(R.tables[0]);

    const p = R.inference_protocol, proto = $("#protocol");
    if (p) proto.textContent = `Inference protocol for every fine-tuned row: ${p.num_inference_steps} denoising steps, CFG ${p.guidance_scale}, seed ${p.seed}, ${p.canvas_resolution}² canvas, ${p.native_resolution_scoring ? "scored at native ground-truth resolution" : "scored on the canvas"}. ${p.note || ""}`;
    if (R.generated_at) proto.textContent += ` results.json generated ${R.generated_at} from ${R.source_count || "?"} evaluation JSON(s), commit ${R.commit || "?"}.`;
    if (R.report_url) $("#report-link").href = R.report_url;
  } else {
    $("#chart").innerHTML = '<p class="fine">data/results.json not found.</p>';
  }

  // --------------------------------------------------------------- copy buttons
  document.querySelectorAll(".copy").forEach((b) => b.addEventListener("click", async () => {
    const el = $("#" + b.dataset.copy);
    try { await navigator.clipboard.writeText(el.innerText); }
    catch { const r = document.createRange(); r.selectNodeContents(el); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); document.execCommand("copy"); sel.removeAllRanges(); }
    b.textContent = "Copied"; b.classList.add("done");
    setTimeout(() => { b.textContent = "Copy"; b.classList.remove("done"); }, 1600);
  }));

  // --------------------------------------------------------------- which section the nav points at
  {
    const links = [...document.querySelectorAll('.nav nav a[href^="#"]')];
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
    }), { rootMargin: "-35% 0px -60% 0px" });
    links.forEach((a) => { const t = $(a.getAttribute("href")); if (t) io.observe(t); });
  }

  // --------------------------------------------------------------- scroll depth, in the depth ramp
  {
    const bar = $("#depthbar"); let queued = false;
    const upd = () => { queued = false; const max = document.documentElement.scrollHeight - innerHeight; bar.style.setProperty("--p", `${max > 0 ? Math.min(100, (100 * scrollY) / max).toFixed(2) : 0}%`); };
    const ask = () => { if (!queued) { queued = true; requestAnimationFrame(upd); } };
    addEventListener("scroll", ask, { passive: true }); addEventListener("resize", ask); upd();
  }
})();
