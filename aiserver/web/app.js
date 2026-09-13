/* CaseSorter AI Server — admin UI. Vanilla JS, no build step, no CDN. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ utils
  const $ = (sel, root = document) => root.querySelector(sel);
  const h = (tag, attrs = {}, ...children) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (v === false || v == null) continue;
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtBytes = (n) => { if (n == null) return "—"; const u = ["B", "KB", "MB", "GB"]; let i = 0; while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; } return `${n.toFixed(i ? 1 : 0)} ${u[i]}`; };
  const fmtPct = (v, d = 1) => (v == null ? "—" : `${Number(v).toFixed(d)}%`);
  const fmtDate = (s) => { if (!s) return "—"; const d = new Date(s); return isNaN(d) ? s : d.toLocaleString(); };
  const fmtDur = (s) => { if (s == null) return "—"; s = Math.round(s); const m = Math.floor(s / 60), sec = s % 60; return m ? `${m}m ${sec}s` : `${sec}s`; };

  function toast(msg, kind = "") {
    const el = h("div", { class: `toast ${kind}` }, msg);
    $("#toasts").append(el);
    setTimeout(() => el.remove(), kind === "bad" ? 7000 : 3500);
  }

  async function api(method, url, body, opts = {}) {
    const init = { method, headers: {}, credentials: "same-origin" };
    if (body instanceof FormData) init.body = body;
    else if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
    const res = await fetch(url, init);
    if (res.status === 401 && !opts.quiet) { state.authed = false; render(); throw new Error("Not signed in"); }
    const ct = res.headers.get("content-type") || "";
    const data = ct.includes("application/json") ? await res.json() : await res.text();
    if (!res.ok) {
      const msg = (data && data.detail) ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`;
      const err = new Error(msg); err.status = res.status; throw err;
    }
    return data;
  }
  const GET = (u, o) => api("GET", u, undefined, o), POST = (u, b, o) => api("POST", u, b, o), PATCH = (u, b) => api("PATCH", u, b), PUT = (u, b) => api("PUT", u, b), DEL = (u) => api("DELETE", u);

  function modal(title, bodyEl, actions = []) {
    const root = $("#modal-root");
    root.innerHTML = "";
    const close = () => { root.hidden = true; root.innerHTML = ""; };
    const m = h("div", { class: "modal stack" },
      h("div", { class: "row between" }, h("h2", {}, title), h("button", { class: "sm", onclick: close }, "✕")),
      bodyEl,
      h("div", { class: "row", style: "justify-content:flex-end" }, actions.map((a) => h("button", { class: a.class || "", onclick: async () => { try { const r = await a.onclick(); if (r !== false) close(); } catch (e) { toast(e.message, "bad"); } } }, a.label)))
    );
    root.append(m);
    root.hidden = false;
    root.onclick = (e) => { if (e.target === root) close(); };
    return close;
  }
  const confirmModal = (title, text) => new Promise((resolve) => {
    const close = modal(title, h("p", {}, text), [
      { label: "Cancel", onclick: () => { resolve(false); } },
      { label: "Confirm", class: "danger", onclick: () => { resolve(true); } },
    ]);
    $("#modal-root").addEventListener("click", (e) => { if (e.target === $("#modal-root")) resolve(false); }, { once: true });
  });

  function field(label, input) { return h("label", { class: "field" }, label, input); }
  function input(attrs) { return h("input", { type: "text", ...attrs }); }
  function select(options, value, attrs = {}) {
    const s = h("select", attrs);
    for (const o of options) { const [v, t] = Array.isArray(o) ? o : [o, o]; s.append(h("option", { value: v, selected: v === value }, t)); }
    return s;
  }
  function badge(text, kind = "") { return h("span", { class: `badge ${kind}` }, text); }
  function statusBadge(job) {
    const k = { queued: "", running: "accent", done: "ok", failed: "bad", cancelled: "warn" }[job.status] || "";
    return badge(job.status, k);
  }

  // ------------------------------------------------------------------ state
  const state = { authed: false, auth: null, page: "dashboard", params: {}, pollers: [], serverInfo: null };
  function stopPollers() { for (const p of state.pollers) clearInterval(p); state.pollers = []; }
  function poll(fn, ms) { fn(); state.pollers.push(setInterval(fn, ms)); }

  // ------------------------------------------------------------------ router
  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, "") || "dashboard";
    const [path, query = ""] = raw.split("?");
    const parts = path.split("/");
    const params = Object.fromEntries(new URLSearchParams(query));
    return { page: parts[0], id: parts[1], sub: parts[2], params };
  }
  window.addEventListener("hashchange", render);

  async function render() {
    stopPollers();
    const main = $("#main");
    const route = parseHash();
    state.page = route.page;
    document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === route.page));
    try {
      if (!state.auth) state.auth = await GET("/api/admin/auth", { quiet: true });
      if (!state.auth.authenticated) { main.innerHTML = ""; main.append(state.auth.setup_required ? setupView() : loginView()); $("#logout-btn").hidden = true; return; }
      state.authed = true;
      $("#logout-btn").hidden = !state.auth.password_required;
      main.innerHTML = "";
      const pages = { dashboard: dashboardView, models: route.id ? () => modelView(Number(route.id), route.sub || "overview") : modelsView, jobs: jobsView, community: communityView, clients: clientsView, settings: settingsView };
      const view = pages[route.page] || dashboardView;
      main.append(await view());
    } catch (e) {
      main.innerHTML = "";
      main.append(h("div", { class: "card" }, h("h2", {}, "Something went wrong"), h("p", { class: "mono" }, e.message)));
    }
  }

  $("#logout-btn").addEventListener("click", async () => { await POST("/api/admin/logout"); state.auth = null; render(); });

  // ------------------------------------------------------------------ auth views
  function setupView() {
    const pw = input({ type: "password", placeholder: "Choose an admin password (6+ characters)" });
    const pw2 = input({ type: "password", placeholder: "Repeat password" });
    return h("div", { class: "card login stack" },
      h("h1", {}, "Welcome"),
      h("p", { class: "muted" }, `This server listens on ${esc(state.auth.host)}, so the web UI needs an admin password before anyone can use it.`),
      field("Password", pw), field("Repeat", pw2),
      h("button", { class: "primary", onclick: async () => {
        if (pw.value !== pw2.value) return toast("Passwords do not match", "bad");
        try { await POST("/api/admin/setup", { password: pw.value }); state.auth = null; render(); } catch (e) { toast(e.message, "bad"); }
      } }, "Create password")
    );
  }
  function loginView() {
    const pw = input({ type: "password", placeholder: "Admin password" });
    const go = async () => { try { await POST("/api/admin/login", { password: pw.value }); state.auth = null; render(); } catch (e) { toast(e.message, "bad"); } };
    pw.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    return h("div", { class: "card login stack" }, h("h1", {}, "Sign in"), field("Admin password", pw), h("button", { class: "primary", onclick: go }, "Sign in"),
      h("p", { class: "muted small" }, "The password is set on first visit or in config.py (ADMIN_PASSWORD)."));
  }

  // ------------------------------------------------------------------ dashboard
  async function dashboardView() {
    const wrap = h("div", { class: "stack" }, h("h1", {}, "Dashboard"));
    const statsRow = h("div", { class: "grid cols-3" });
    const jobsCard = h("div", { class: "card" }, h("h2", {}, "Active jobs"));
    const servedCard = h("div", { class: "card" }, h("h2", {}, "Served models (OpenAI API)"));
    const modelsCard = h("div", { class: "card" }, h("h2", {}, "Models"));
    wrap.append(statsRow, h("div", { class: "grid cols-2" }, jobsCard, servedCard), modelsCard);

    const refresh = async () => {
      const [info, models] = await Promise.all([GET("/api/v1/server"), GET("/api/v1/models")]);
      state.serverInfo = info;
      const dev = info.device || {};
      $("#device-badge").textContent = dev.device === "cuda" ? `GPU: ${dev.gpu_name || "CUDA"}` : `Device: ${(dev.device || "cpu").toUpperCase()}`;
      statsRow.innerHTML = "";
      const stat = (v, l) => h("div", { class: "card stat" }, h("div", { class: "value" }, v), h("div", { class: "label" }, l));
      statsRow.append(
        stat(models.length, "Models in registry"),
        stat(info.served_models.length, "Models served"),
        stat(info.active_jobs.length, "Active jobs"),
        stat(dev.device === "cuda" ? "GPU" : (dev.device || "cpu").toUpperCase(), dev.gpu_name || `PyTorch ${dev.torch || ""}`),
        stat(fmtDur(info.uptime_seconds), "Uptime"),
        stat(`${info.host}:${info.port}`, "Listening on")
      );
      jobsCard.querySelectorAll(":scope > :not(h2)").forEach((n) => n.remove());
      if (!info.active_jobs.length) jobsCard.append(h("p", { class: "muted" }, "Nothing running."));
      for (const j of info.active_jobs) jobsCard.append(jobRow(j, models));
      servedCard.querySelectorAll(":scope > :not(h2)").forEach((n) => n.remove());
      if (!info.served_models.length) servedCard.append(h("p", { class: "muted" }, "No models are being served. Enable serving on a trained model, or add one to config.MODELS."));
      else servedCard.append(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Alias"), h("th", {}, "Source"), h("th", {}, "Loaded"), h("th", {}, "Classes"))),
        h("tbody", {}, info.served_models.map((s) => h("tr", {}, h("td", { class: "mono" }, s.alias), h("td", {}, s.source === "config" ? "config.py" : "registry"), h("td", {}, s.loaded ? badge("loaded", "ok") : badge("lazy")), h("td", {}, s.classes ?? "—"))))));
      modelsCard.querySelectorAll(":scope > :not(h2)").forEach((n) => n.remove());
      modelsCard.append(modelsTable(models));
    };
    poll(() => refresh().catch((e) => toast(e.message, "bad")), 4000);
    return wrap;
  }

  function jobRow(j, models = []) {
    const m = models.find((x) => x.id === j.model_id);
    const p = j.progress || {};
    let text = p.phase || j.status;
    if (j.kind === "train" && p.total) text = `epoch ${p.epoch || 0}/${p.total}` + (p.batch ? ` · ${p.batch.phase} batch ${p.batch.batch}/${p.batch.batches}` : "");
    if (j.kind === "evaluate" && p.total) text = `${p.current}/${p.total} images`;
    if (j.kind === "download" && p.total) text = `${fmtBytes(p.bytes)} / ${fmtBytes(p.total)}`;
    if (j.kind === "share" && p.total) text = `${p.phase} ${fmtBytes(p.bytes)} / ${fmtBytes(p.total)}`;
    const pct = jobPct(j);
    return h("div", { class: "stack", style: "margin-bottom:10px" },
      h("div", { class: "row between" },
        h("div", {}, statusBadge(j), " ", h("strong", {}, j.kind), " ", m ? h("a", { href: `#/models/${m.id}/training` }, m.name) : (j.request && j.request.name) || "", " ", h("span", { class: "muted small" }, text)),
        (j.status === "running" || j.status === "queued") ? h("button", { class: "sm danger", onclick: async () => { await POST(`/api/v1/jobs/${j.id}/cancel`); toast("Cancel requested"); } }, "Cancel") : null),
      pct != null ? h("div", { class: "progress" }, h("div", { style: `width:${pct}%` })) : null);
  }
  function jobPct(j) {
    const p = j.progress || {};
    if (j.status === "done") return 100;
    if (j.kind === "train" && p.total) { const b = p.batch; const e = (p.epoch || 0); const frac = b && b.batches ? (b.phase === "train" ? b.batch / b.batches * 0.8 : 0.8 + b.batch / b.batches * 0.2) : 0; return Math.min(100, ((e + frac) / p.total) * 100); }
    if (j.kind === "evaluate" && p.total) return p.current / p.total * 100;
    if ((j.kind === "download" || j.kind === "share") && p.total) return p.bytes / p.total * 100;
    return null;
  }

  // ------------------------------------------------------------------ models list
  function modelsTable(models) {
    if (!models.length) return h("p", { class: "muted" }, "No models yet. Create one, import a ZIP, or download from the community.");
    return h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Model"), h("th", {}, "Cartridge"), h("th", {}, "Mode"), h("th", {}, "Type"), h("th", {}, "Images"), h("th", {}, "Trained"), h("th", {}, "Val acc"), h("th", {}, "Serving"), h("th", {}, "Job"))),
      h("tbody", {}, models.map((m) => h("tr", { class: "clickable", onclick: () => { location.hash = `#/models/${m.id}`; } },
        h("td", {}, h("strong", {}, m.name)),
        h("td", {}, m.cartridge_name || "—"),
        h("td", {}, m.mode_label),
        h("td", {}, m.model_type === "CommunityManaged" ? badge("community", "accent") : m.model_type === "ReadOnly" ? badge("read-only", "warn") : badge("standard")),
        h("td", {}, m.image_count),
        h("td", {}, m.has_checkpoint ? badge("yes", "ok") : badge("no")),
        h("td", {}, m.last_val_acc != null ? fmtPct(m.last_val_acc * 100) : "—"),
        h("td", {}, m.is_serving ? h("span", {}, badge("serving", "ok"), " ", h("code", {}, m.alias)) : m.serve_enabled ? badge("enabled (no checkpoint)", "warn") : badge("off")),
        h("td", {}, m.active_job ? statusBadge(m.active_job) : "")
      )))));
  }

  async function modelsView() {
    const models = await GET("/api/v1/models");
    const wrap = h("div", { class: "stack" },
      h("div", { class: "row between" }, h("h1", {}, "Models"), h("div", { class: "row" },
        h("button", { class: "primary", onclick: () => createModelDialog() }, "＋ New model"),
        h("button", { onclick: () => importDialog() }, "Import ZIP…"),
        h("a", { class: "btn", href: "#/community" }, "Community…"))),
      h("div", { class: "card" }, modelsTable(models)));
    return wrap;
  }

  const MODES = [["convnext_tiny", "ConvNeXt-Tiny (fast)"], ["convnext_small", "ConvNeXt-Small"], ["convnext_base", "ConvNeXt-Base"], ["convnext_large", "ConvNeXt-Large (slow, most accurate)"]];

  function createModelDialog() {
    const name = input({ placeholder: "e.g. 9mm headstamps" });
    const cart = input({ placeholder: "e.g. 9mm" });
    const mode = select(MODES, "convnext_tiny");
    const hs = h("textarea", { placeholder: "Optional: one headstamp per line (they are also created automatically from image labels)" });
    modal("New model", h("div", { class: "stack" }, field("Name", name), field("Cartridge", cart), field("Training mode", mode), field("Headstamps", hs)), [
      { label: "Create", class: "primary", onclick: async () => {
        const m = await POST("/api/v1/models", { name: name.value, cartridge_name: cart.value, model_mode: mode.value, headstamps: hs.value.split("\n").map((s) => s.trim()).filter(Boolean) });
        toast(`Created ${m.name}`, "ok"); location.hash = `#/models/${m.id}/images`;
      } }]);
  }

  function importDialog() {
    const file = h("input", { type: "file", accept: ".zip" });
    const name = input({ placeholder: "Leave empty to use the archive's name" });
    const upd = h("input", { type: "checkbox", checked: true });
    modal("Import model ZIP", h("div", { class: "stack" },
      h("p", { class: "muted small" }, "Accepts exports from this server, the CaseSorter desktop client and the Windows app (manifest.json + model/ + images/)."),
      field("Archive", file), field("Name override", name), h("label", { class: "check" }, upd, "Update the existing copy when the archive is a newer version of an installed community model")), [
      { label: "Import", class: "primary", onclick: async () => {
        if (!file.files[0]) { toast("Pick a .zip first", "bad"); return false; }
        const fd = new FormData(); fd.append("file", file.files[0]); if (name.value) fd.append("name", name.value); fd.append("update_existing", upd.checked ? "true" : "false");
        toast("Importing…"); const m = await POST("/api/v1/models/import", fd); toast(`Imported ${m.name}`, "ok"); location.hash = `#/models/${m.id}`;
      } }]);
  }

  // ------------------------------------------------------------------ model detail
  async function modelView(id, tab) {
    let m;
    try { m = await GET(`/api/v1/models/${id}`); } catch (e) { return h("div", { class: "card" }, h("p", {}, e.message), h("a", { href: "#/models" }, "Back to models")); }
    const tabs = [["overview", "Overview"], ["images", `Images (${m.image_count})`], ["training", "Training"], ["evaluate", "Evaluate"], ["share", "Share"]];
    const body = h("div", {});
    const wrap = h("div", { class: "stack" },
      h("div", { class: "row between" },
        h("div", {}, h("a", { href: "#/models", class: "small" }, "← Models"), h("h1", { style: "margin-top:4px" }, m.name, " ", m.model_type === "CommunityManaged" ? badge("community", "accent") : null, " ", m.is_serving ? badge("serving", "ok") : null)),
        h("div", { class: "row" }, h("span", { class: "muted small" }, `${m.mode_label} · ${m.cartridge_name || "no cartridge"} · id ${m.id}`))),
      h("div", { class: "tabs" }, tabs.map(([k, t]) => h("button", { class: k === tab ? "active" : "", onclick: () => { location.hash = `#/models/${id}/${k}`; } }, t))),
      body);
    const views = { overview: overviewTab, images: imagesTab, training: trainingTab, evaluate: evaluateTab, share: shareTab };
    body.append(await (views[tab] || overviewTab)(m));
    return wrap;
  }

  async function overviewTab(m) {
    const name = input({ value: m.name }), cart = input({ value: m.cartridge_name || "" }), mode = select(MODES, m.model_mode, { disabled: m.model_mode === "openai" });
    const notes = h("textarea", {}, m.notes || "");
    const hide = h("input", { type: "checkbox", checked: m.hide_primer }), primer = input({ type: "number", value: m.primer_mask_size, min: 0, max: 512 });
    const alias = input({ value: m.serve_alias || "", placeholder: m.name });
    const serve = h("input", { type: "checkbox", checked: m.serve_enabled, disabled: !m.has_checkpoint });
    const hsList = h("div", { class: "row" });
    const renderHs = (names) => { hsList.innerHTML = ""; if (!names.length) hsList.append(h("span", { class: "muted" }, "None yet — they appear as images are labelled.")); for (const n of names) hsList.append(h("span", { class: "badge" }, n, " ", h("button", { class: "link", title: "Rename", onclick: () => renameHs(n) }, "✎"), " ", m.trainable ? h("button", { class: "link", title: "Remove", onclick: async () => { if (!(await confirmModal("Remove headstamp", `Remove ${n}? Its images stay on disk.`))) return; renderHs(await DEL(`/api/v1/models/${m.id}/headstamps/${encodeURIComponent(n)}`)); } }, "✕") : null)); };
    const renameHs = (old) => { const nn = input({ value: old }); modal("Rename headstamp", h("div", { class: "stack" }, field("New name", nn), h("p", { class: "muted small" }, "Training images with this label are renamed too.")), [{ label: "Rename", class: "primary", onclick: async () => { const r = await POST(`/api/v1/models/${m.id}/headstamps/rename`, { old, new: nn.value }); renderHs(r.headstamps); toast(`Renamed; ${r.images_renamed} image(s) updated`, "ok"); } }]); };
    renderHs(m.headstamps);
    const newHs = input({ placeholder: "Add headstamp" });
    const addHs = async () => { if (!newHs.value.trim()) return; renderHs(await POST(`/api/v1/models/${m.id}/headstamps`, { name: newHs.value })); newHs.value = ""; };
    newHs.addEventListener("keydown", (e) => { if (e.key === "Enter") addHs(); });

    const ckpt = m.has_checkpoint ? `${fmtBytes(m.checkpoint_size)} · trained ${m.last_training_date || "?"} (${fmtDur(m.last_training_duration)}, ${m.trained_image_count} images)` + (m.last_val_acc != null ? ` · val acc ${fmtPct(m.last_val_acc * 100)}` : "") : "No checkpoint yet";
    const env = m.checkpoint_env && m.checkpoint_env.torch ? `PyTorch ${m.checkpoint_env.torch}, torchvision ${m.checkpoint_env.torchvision}, numpy ${m.checkpoint_env.numpy}` : "not recorded";

    return h("div", { class: "grid cols-2" },
      h("div", { class: "card stack" }, h("h2", {}, "Details"),
        field("Name", name), field("Cartridge", cart), field("Training mode (backbone)", mode),
        h("div", { class: "row" }, h("label", { class: "check" }, hide, "Hide primer in cropped image"), field("Primer mask size (px)", primer)),
        field("Notes", notes),
        h("div", { class: "row" }, h("button", { class: "primary", onclick: async () => { try { await PATCH(`/api/v1/models/${m.id}`, { name: name.value, cartridge_name: cart.value, model_mode: mode.value, notes: notes.value, hide_primer: hide.checked, primer_mask_size: Number(primer.value) }); toast("Saved", "ok"); render(); } catch (e) { toast(e.message, "bad"); } } }, "Save"),
          h("button", { class: "danger", onclick: async () => { if (!(await confirmModal("Delete model", `Delete "${m.name}" with all its images, checkpoints and reports? This cannot be undone.`))) return; try { await DEL(`/api/v1/models/${m.id}`); toast("Deleted"); location.hash = "#/models"; } catch (e) { toast(e.message, "bad"); } } }, "Delete model"))),
      h("div", { class: "stack" },
        h("div", { class: "card stack" }, h("h2", {}, "Serving"),
          h("p", { class: "muted small" }, "When enabled, the model answers on the OpenAI-compatible endpoint. Point the CaseSorter client at this server with the alias as the model name."),
          h("div", { class: "row" }, h("label", { class: "check" }, serve, "Serve this model"), field("Alias (model name for clients)", alias)),
          h("p", { class: "mono small" }, `POST http://${location.host}/v1/chat/completions  model="${esc(m.alias)}"`),
          h("button", { class: "primary", onclick: async () => { try { await POST(`/api/v1/models/${m.id}/serve`, { enabled: serve.checked, alias: alias.value }); toast("Serving updated", "ok"); render(); } catch (e) { toast(e.message, "bad"); } } }, "Apply")),
        h("div", { class: "card stack" }, h("h2", {}, "Checkpoint"), h("p", {}, ckpt), h("p", { class: "muted small" }, `Built with: ${env}`),
          h("div", { class: "row" },
            h("a", { class: "btn", href: `/api/v1/models/${m.id}/export?mode=ModelAndImages` }, "Export ZIP (model + images)"),
            h("a", { class: "btn", href: `/api/v1/models/${m.id}/export?mode=ModelOnly` }, "Model only"),
            h("a", { class: "btn", href: `/api/v1/models/${m.id}/export?mode=ImagesOnly` }, "Images only"),
            m.has_checkpoint ? h("a", { class: "btn", href: `/api/v1/models/${m.id}/checkpoint` }, "Download .pth") : null)),
        h("div", { class: "card stack" }, h("h2", {}, "Headstamps"), hsList, m.trainable ? h("div", { class: "row" }, newHs, h("button", { onclick: addHs }, "Add")) : h("p", { class: "muted small" }, "Community models keep the publisher's headstamps.")),
        m.community_model_uid ? h("div", { class: "card" }, h("h2", {}, "Community"), h("dl", { class: "kv" }, h("dt", {}, "UID"), h("dd", { class: "mono" }, m.community_model_uid), h("dt", {}, "Version"), h("dd", {}, m.model_version), h("dt", {}, "Feedback loop"), h("dd", {}, m.feedback_loop_enabled ? `on (floor ${m.feedback_loop_confidence_floor}%)` : "off"))) : null));
  }

  // ---- images
  async function imagesTab(m) {
    const st = { label: "All", page: 1, size: 100, selected: new Set(), search: "" };
    const labelSel = h("select", { onchange: () => { st.label = labelSel.value; st.page = 1; load(); } });
    const search = h("input", { type: "search", placeholder: "Search filename", style: "max-width:220px", oninput: () => { st.search = search.value; st.page = 1; load(); } });
    const sizeSel = select([50, 100, 200, 500].map((n) => [String(n), `${n} / page`]), "100", { onchange: () => { st.size = Number(sizeSel.value); st.page = 1; load(); } });
    const grid = h("div", { class: "img-grid" });
    const pager = h("div", { class: "row" });
    const selInfo = h("span", { class: "muted small" }, "0 selected");
    const reclassTarget = h("select", {});
    const bulk = h("div", { class: "row" }, selInfo,
      h("button", { class: "sm", onclick: () => { grid.querySelectorAll(".tile").forEach((t) => { st.selected.add(t.dataset.name); t.classList.add("selected"); }); updateSel(); } }, "Select page"),
      h("button", { class: "sm", onclick: () => { st.selected.clear(); grid.querySelectorAll(".tile").forEach((t) => t.classList.remove("selected")); updateSel(); } }, "Clear"),
      m.trainable ? h("span", { class: "row" }, "Reclassify to", reclassTarget, h("button", { class: "sm", onclick: async () => { if (!st.selected.size) return; const r = await POST(`/api/v1/models/${m.id}/images/bulk`, { action: "reclassify", filenames: [...st.selected], label: reclassTarget.value }); toast(`${r.changed.length} reclassified${r.failed.length ? `, ${r.failed.length} failed` : ""}`, r.failed.length ? "bad" : "ok"); st.selected.clear(); load(); } }, "Apply")) : null,
      m.trainable ? h("button", { class: "sm danger", onclick: async () => { if (!st.selected.size) return; if (!(await confirmModal("Delete images", `Delete ${st.selected.size} image(s)? This cannot be undone.`))) return; const r = await POST(`/api/v1/models/${m.id}/images/bulk`, { action: "delete", filenames: [...st.selected] }); toast(`${r.deleted.length} deleted`); st.selected.clear(); load(); } }, "Delete selected") : null);
    const updateSel = () => { selInfo.textContent = `${st.selected.size} selected`; };

    const uploadLabel = h("input", { list: "hs-list", placeholder: "Label for uploads (or name files LABEL__123.jpg)", style: "max-width:300px" });
    const datalist = h("datalist", { id: "hs-list" });
    const fileIn = h("input", { type: "file", multiple: true, accept: "image/*" });
    const uploadBtn = h("button", { class: "primary", onclick: async () => {
      if (!fileIn.files.length) return toast("Pick some images first", "bad");
      const files = [...fileIn.files]; let saved = 0, errors = 0;
      uploadBtn.disabled = true;
      for (let i = 0; i < files.length; i += 20) {
        const fd = new FormData(); for (const f of files.slice(i, i + 20)) fd.append("files", f, f.name); if (uploadLabel.value.trim()) fd.append("label", uploadLabel.value.trim());
        try { const r = await POST(`/api/v1/models/${m.id}/images`, fd); saved += r.saved.length; errors += r.errors.length; if (r.errors.length) console.warn(r.errors); } catch (e) { toast(e.message, "bad"); break; }
        uploadBtn.textContent = `Uploading ${Math.min(i + 20, files.length)}/${files.length}…`;
      }
      uploadBtn.disabled = false; uploadBtn.textContent = "Upload";
      toast(`${saved} image(s) added${errors ? `, ${errors} skipped (no label?)` : ""}`, errors ? "bad" : "ok"); fileIn.value = ""; load();
    } }, "Upload");

    const load = async () => {
      const data = await GET(`/api/v1/models/${m.id}/images?label=${encodeURIComponent(st.label)}&page=${st.page}&page_size=${st.size}&search=${encodeURIComponent(st.search)}`);
      const labels = Object.entries(data.labels);
      const cur = labelSel.value || "All";
      labelSel.innerHTML = ""; labelSel.append(h("option", { value: "All" }, `All (${labels.reduce((a, [, n]) => a + n, 0)})`)); for (const [l, n] of labels) labelSel.append(h("option", { value: l }, `${l} (${n})`)); labelSel.append(h("option", { value: "UNKNOWN HEADSTAMP" }, "Not a headstamp"));
      labelSel.value = cur;
      const hsNames = m.headstamps.slice(); for (const [l] of labels) if (!hsNames.includes(l)) hsNames.push(l);
      reclassTarget.innerHTML = ""; datalist.innerHTML = ""; for (const n of hsNames.sort()) { reclassTarget.append(h("option", { value: n }, n)); datalist.append(h("option", { value: n })); }
      grid.innerHTML = "";
      if (!data.total) grid.append(h("p", { class: "muted" }, "No images here yet."));
      for (const it of data.items) {
        const tile = h("div", { class: `tile ${st.selected.has(it.filename) ? "selected" : ""}`, "data-name": it.filename, title: it.filename },
          h("img", { loading: "lazy", src: `/api/v1/models/${m.id}/images/${encodeURIComponent(it.filename)}?thumb=true` }), h("div", { class: "name" }, it.label || it.filename));
        tile.addEventListener("click", (e) => { if (e.detail === 2) return; if (st.selected.has(it.filename)) { st.selected.delete(it.filename); tile.classList.remove("selected"); } else { st.selected.add(it.filename); tile.classList.add("selected"); } updateSel(); });
        tile.addEventListener("dblclick", () => previewImage(m, it, hsNames, load));
        grid.append(tile);
      }
      const pages = Math.max(1, Math.ceil(data.total / st.size));
      pager.innerHTML = "";
      pager.append(h("button", { class: "sm", disabled: st.page <= 1, onclick: () => { st.page--; load(); } }, "‹ Prev"), h("span", { class: "small muted" }, `Page ${st.page} of ${pages} · ${data.total} image(s)`), h("button", { class: "sm", disabled: st.page >= pages, onclick: () => { st.page++; load(); } }, "Next ›"));
    };
    await load();
    return h("div", { class: "stack" },
      m.trainable ? h("div", { class: "card row" }, h("strong", {}, "Add training images"), fileIn, uploadLabel, datalist, uploadBtn, h("span", { class: "muted small" }, "Images are stored as {label}__{ticks}.jpg, the same convention as the desktop client.")) : h("div", { class: "notice" }, "This is a community model: its images are read-only."),
      h("div", { class: "card stack" }, h("div", { class: "row between" }, h("div", { class: "row" }, labelSel, search, sizeSel), pager), bulk, grid, h("p", { class: "muted small" }, "Click to select, double-click to preview.")));
  }

  function previewImage(m, it, hsNames, reload) {
    const target = select(hsNames, it.label);
    modal(it.filename, h("div", { class: "stack" }, h("img", { src: `/api/v1/models/${m.id}/images/${encodeURIComponent(it.filename)}`, style: "max-width:100%;max-height:60vh;object-fit:contain;background:#111;border-radius:6px" }),
      h("p", { class: "muted small" }, `${it.label} · ${fmtBytes(it.size)} · ${fmtDate(it.modified)}`),
      m.trainable ? h("div", { class: "row" }, "Reclassify to", target, h("button", { class: "sm", onclick: async () => { const r = await POST(`/api/v1/models/${m.id}/images/bulk`, { action: "reclassify", filenames: [it.filename], label: target.value }); toast(r.changed.length ? "Reclassified" : "Nothing changed"); reload(); } }, "Apply"),
        h("button", { class: "sm danger", onclick: async () => { if (!(await confirmModal("Delete image", `Delete ${it.filename}?`))) return; await DEL(`/api/v1/models/${m.id}/images/${encodeURIComponent(it.filename)}`); toast("Deleted"); reload(); $("#modal-root").hidden = true; } }, "Delete")) : null), []);
  }

  // ---- training
  const TRAIN_FIELDS = [
    ["epochs", "Epochs", "number", { min: 1, max: 500 }], ["batch_size", "Batch size", "number", { min: 1, max: 1024 }], ["learning_rate", "Learning rate", "number", { step: "any" }], ["weight_decay", "Weight decay", "number", { step: "any" }],
    ["dropout_rate", "Dropout", "number", { step: 0.05, min: 0, max: 0.95 }], ["val_split", "Validation split", "number", { step: 0.05, min: 0, max: 0.95 }], ["image_size", "Image size (px)", "number", { min: 64, max: 640 }], ["max_workers", "Loader workers (-1 auto)", "number", { min: -1, max: 64 }],
    ["stochastic_depth_prob", "Stochastic depth (-1 default)", "number", { step: 0.05, min: -1, max: 0.5 }], ["focal_gamma", "Focal gamma", "number", { step: 0.1 }], ["swa_start", "SWA start (fraction)", "number", { step: 0.05, min: 0, max: 1 }], ["swa_acc_threshold", "SWA acc threshold", "number", { step: 0.01 }], ["swa_patience", "SWA patience", "number", { min: 1 }], ["swa_min_epoch", "SWA min epoch", "number", { min: 1 }],
  ];
  const TRAIN_BOOLS = [["train_all", "Train on full dataset (no validation)"], ["freeze_backbone", "Freeze backbone (train classifier only)"], ["use_focal_loss", "Use focal loss"], ["use_swa", "Use SWA (stochastic weight averaging)"]];

  async function trainingTab(m) {
    const tc = m.training_config;
    const inputs = {};
    const form = h("div", { class: "form-grid" });
    for (const [k, label, type, extra] of TRAIN_FIELDS) { inputs[k] = h("input", { type, value: tc[k], ...extra }); form.append(field(label, inputs[k])); }
    const bools = {};
    const boolRow = h("div", { class: "row" });
    for (const [k, label] of TRAIN_BOOLS) { bools[k] = h("input", { type: "checkbox", checked: !!tc[k] }); boolRow.append(h("label", { class: "check" }, bools[k], label)); }
    const swaMode = select([["scheduled", "scheduled"], ["adaptive", "adaptive"]], tc.swa_mode);
    const collect = () => { const o = {}; for (const [k, , type] of TRAIN_FIELDS) o[k] = type === "number" ? Number(inputs[k].value) : inputs[k].value; for (const k in bools) o[k] = bools[k].checked; o.swa_mode = swaMode.value; return o; };
    const save = async () => { await PATCH(`/api/v1/models/${m.id}`, { training_config: collect() }); toast("Training settings saved", "ok"); };

    const status = h("div", { class: "stack" });
    const logEl = h("pre", { class: "log" }, "");
    const epochTable = h("div", {});
    let logCursor = 0, curJob = null, logJobId = null;
    const startBtn = h("button", { class: "primary", disabled: !m.trainable || !!m.active_job, onclick: async () => {
      try { await save(); const job = await POST(`/api/v1/models/${m.id}/train`, { training_config: collect() }); toast(`Training started (job ${job.id})`, "ok"); curJob = job; startBtn.disabled = true; refresh(); } catch (e) { toast(e.message, "bad"); }
    } }, "▶ Start training");
    const cancelBtn = h("button", { class: "danger", hidden: true, onclick: async () => { if (curJob) { await POST(`/api/v1/jobs/${curJob.id}/cancel`); toast("Cancel requested"); } } }, "■ Cancel");

    const renderEpochs = (epochs) => {
      epochTable.innerHTML = "";
      if (!epochs || !epochs.length) return;
      epochTable.append(h("table", { class: "epoch-table" }, h("thead", {}, h("tr", {}, h("th", {}, "Epoch"), h("th", {}, "Train loss"), h("th", {}, "Train acc"), h("th", {}, "Val loss"), h("th", {}, "Val acc"), h("th", {}, "LR"), h("th", {}, "Saved"), h("th", {}, "Time"))),
        h("tbody", {}, epochs.slice().reverse().map((e) => h("tr", {}, h("td", {}, `${e.epoch}/${e.total}`), h("td", {}, e.train_loss.toFixed(4)), h("td", {}, fmtPct(e.train_acc * 100)), h("td", {}, e.val_loss != null ? e.val_loss.toFixed(4) : "—"), h("td", {}, e.val_acc != null ? fmtPct(e.val_acc * 100) : "—"), h("td", {}, Number(e.lr).toExponential(2)), h("td", {}, e.saved === "new-best" ? badge("best", "ok") : e.saved), h("td", {}, fmtDur(e.epoch_seconds)))))));
    };
    const refresh = async () => {
      const jobs = await GET(`/api/v1/models/${m.id}/jobs?kind=train&limit=10`);
      const live = jobs.find((j) => j.status === "running" || j.status === "queued");
      curJob = live || jobs[0] || null;
      status.innerHTML = "";
      startBtn.disabled = !m.trainable || !!live;
      cancelBtn.hidden = !live;
      if (!curJob) { status.append(h("p", { class: "muted" }, "No training runs yet.")); return; }
      const p = curJob.progress || {}; const pct = jobPct(curJob);
      status.append(h("div", { class: "row" }, statusBadge(curJob), h("strong", {}, curJob.status === "running" ? (p.phase === "training" ? `Epoch ${p.epoch || 0} of ${p.total || "?"}` : p.phase || "starting…") : `Last run: ${curJob.status}`),
        p.batch ? h("span", { class: "muted small" }, `${p.batch.phase} batch ${p.batch.batch}/${p.batch.batches} · loss ${p.batch.loss.toFixed(4)} · acc ${fmtPct(p.batch.acc * 100)}`) : null,
        h("span", { class: "muted small" }, `started ${fmtDate(curJob.started_at)}`)));
      if (pct != null) status.append(h("div", { class: "progress" }, h("div", { style: `width:${pct}%` })));
      if (curJob.error) status.append(h("div", { class: "notice warn mono small" }, curJob.error));
      if (curJob.result) { const r = curJob.result; status.append(h("p", {}, `Done in ${fmtDur(r.duration_seconds)} · best val acc ${r.best_val_acc != null ? fmtPct(r.best_val_acc * 100) : "n/a (no validation)"} · ${r.images} images · ${(r.classes || []).length} classes`)); }
      renderEpochs(p.epochs || (curJob.result && curJob.result.epochs_detail));
      if (logJobId !== curJob.id) { logJobId = curJob.id; logCursor = 0; logEl.textContent = ""; }
      const log = await GET(`/api/v1/jobs/${curJob.id}/log?after=${logCursor}&limit=500`);
      if (log.lines.length) { logEl.textContent += (logEl.textContent ? "\n" : "") + log.lines.join("\n"); logCursor = log.cursor; logEl.scrollTop = logEl.scrollHeight; }
      if (!live && jobs[0] && jobs[0].status === "done" && !m.has_checkpoint) { m = await GET(`/api/v1/models/${m.id}`); }
    };
    poll(() => refresh().catch((e) => console.warn(e)), 1500);

    const history = await GET(`/api/v1/models/${m.id}/jobs?kind=train&limit=20`);
    return h("div", { class: "stack" },
      !m.trainable ? h("div", { class: "notice warn" }, "Community and read-only models cannot be trained here. Export it and import it back as your own to fork it.") : null,
      h("div", { class: "card stack" }, h("div", { class: "row between" }, h("h2", {}, "Run"), h("div", { class: "row" }, startBtn, cancelBtn)),
        h("p", { class: "muted small" }, `${m.image_count} images across ${Object.keys(m.class_counts).length} labels · backbone ${m.mode_label} · device ${state.serverInfo ? state.serverInfo.device.device : "?"}`), status, epochTable, logEl),
      h("div", { class: "card stack" }, h("div", { class: "row between" }, h("h2", {}, "Training settings"), h("button", { onclick: () => save().catch((e) => toast(e.message, "bad")) }, "Save settings")),
        form, boolRow, h("div", { class: "row" }, field("SWA mode", swaMode)),
        h("p", { class: "muted small" }, "Same options as the desktop client's Training settings dialog. Defaults match it (232 px, 10 epochs, lr 1e-4).")),
      history.length ? h("div", { class: "card" }, h("h2", {}, "History"), h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Started"), h("th", {}, "Status"), h("th", {}, "Epochs"), h("th", {}, "Best val acc"), h("th", {}, "Duration"))),
        h("tbody", {}, history.map((j) => h("tr", {}, h("td", {}, fmtDate(j.created_at)), h("td", {}, statusBadge(j)), h("td", {}, (j.request && j.request.training_config && j.request.training_config.epochs) || "—"), h("td", {}, j.result && j.result.best_val_acc != null ? fmtPct(j.result.best_val_acc * 100) : "—"), h("td", {}, j.result ? fmtDur(j.result.duration_seconds) : "—")))))) : null);
  }

  // ---- evaluate
  async function evaluateTab(m) {
    const list = h("div", {});
    const detail = h("div", {});
    const status = h("div", {});
    const fileIn = h("input", { type: "file", multiple: true, accept: "image/*" });
    const runOwn = h("button", { class: "primary", disabled: !m.has_checkpoint, onclick: async () => { try { const j = await POST(`/api/v1/models/${m.id}/evaluate`, {}); toast("Evaluation started"); track(j.id); } catch (e) { toast(e.message, "bad"); } } }, "Evaluate training images");
    const runUpload = h("button", { disabled: !m.has_checkpoint, onclick: async () => { if (!fileIn.files.length) return toast("Pick images named LABEL__anything.jpg", "bad"); const fd = new FormData(); for (const f of fileIn.files) fd.append("files", f, f.name); try { const j = await POST(`/api/v1/models/${m.id}/evaluate/upload`, fd); toast("Evaluation started"); track(j.id); } catch (e) { toast(e.message, "bad"); } } }, "Evaluate uploaded folder");
    let tracking = null;
    const track = (id) => { tracking = id; };
    const refresh = async () => {
      if (tracking) { const j = await GET(`/api/v1/jobs/${tracking}`); status.innerHTML = ""; const p = j.progress || {}; status.append(h("div", { class: "row" }, statusBadge(j), p.total ? `${p.current}/${p.total} ${p.file || ""}` : "", j.error ? h("span", { class: "mono small" }, j.error) : null)); if (j.status !== "running" && j.status !== "queued") { tracking = null; if (j.result) { showReport(j.result.report); } loadList(); } }
    };
    const loadList = async () => {
      const reports = await GET(`/api/v1/models/${m.id}/evaluations`);
      list.innerHTML = "";
      if (!reports.length) { list.append(h("p", { class: "muted" }, "No evaluations yet.")); return; }
      list.append(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "When"), h("th", {}, "Source"), h("th", {}, "Images"), h("th", {}, "Accuracy"), h("th", {}, "Avg conf"))),
        h("tbody", {}, reports.map((r) => h("tr", { class: "clickable", onclick: () => showReport(r.name) }, h("td", {}, fmtDate(r.created_at)), h("td", {}, r.source || "—"), h("td", {}, r.total), h("td", {}, fmtPct(r.total_accuracy)), h("td", {}, fmtPct(r.avg_confidence)))))));
    };
    const showReport = async (name) => {
      const r = await GET(`/api/v1/models/${m.id}/evaluations/${encodeURIComponent(name)}`);
      detail.innerHTML = "";
      const s = r.summary, c = r.confusion;
      const stat = (v, l) => h("div", { class: "card tight stat" }, h("div", { class: "value" }, v), h("div", { class: "label" }, l));
      detail.append(h("h2", {}, `Report ${fmtDate(r.created_at)}`),
        h("div", { class: "grid cols-3" }, stat(s.total, "Images"), stat(fmtPct(s.total_accuracy), "Accuracy"), stat(fmtPct(s.avg_confidence), "Avg confidence"), stat(s.total - s.total_matches - (s.total - s.with_original), "Mismatches"), stat(s.errors || 0, "Errors")),
        h("h3", { style: "margin-top:12px" }, "Confusion matrix (rows = actual, columns = predicted)"),
        c.labels.length ? h("div", { class: "table-wrap" }, h("table", { class: "matrix" }, h("thead", {}, h("tr", {}, h("th", {}), c.labels.map((l) => h("th", {}, l)))),
          h("tbody", {}, c.labels.map((l, i) => h("tr", {}, h("th", { class: "rowh" }, l), c.matrix[i].map((v, j) => h("td", { class: v ? (i === j ? "diag" : "off") : "" }, v || ""))))))) : h("p", { class: "muted" }, "No ground truth labels in this set."),
        h("h3", { style: "margin-top:12px" }, "Per predicted class"),
        h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Class"), h("th", {}, "Count"), h("th", {}, "Avg conf"), h("th", {}, "Low"), h("th", {}, "High"), h("th", {}, "Mismatches"))),
          h("tbody", {}, s.per_class.map((p) => h("tr", {}, h("td", {}, p.cls), h("td", {}, p.count), h("td", {}, fmtPct(p.avg)), h("td", {}, fmtPct(p.low)), h("td", {}, fmtPct(p.high)), h("td", {}, p.mismatches ? badge(p.mismatches, "bad") : "0"))))),
        h("h3", { style: "margin-top:12px" }, "Mismatches"),
        (() => { const mm = r.results.filter((x) => x.match === "mismatch"); if (!mm.length) return h("p", { class: "muted" }, "None 🎉"); return h("div", { class: "img-grid" }, mm.slice(0, 200).map((x) => h("div", { class: "tile", title: x.filename }, r.source === "training-images" ? h("img", { loading: "lazy", src: `/api/v1/models/${m.id}/images/${encodeURIComponent(x.filename)}?thumb=true` }) : null, h("div", { class: "name" }, `${x.original} → ${x.predicted} (${fmtPct(x.confidence, 0)})`)))); })());
    };
    poll(() => refresh().catch(() => {}), 1500);
    await loadList();
    return h("div", { class: "stack" },
      h("div", { class: "card stack" }, h("h2", {}, "Run an evaluation"),
        h("p", { class: "muted small" }, "Scores the checkpoint against labelled images (label = filename prefix before __). Evaluating the training images shows what the model learned; upload a held-out folder for an honest accuracy figure."),
        h("div", { class: "row" }, runOwn, fileIn, runUpload), status),
      h("div", { class: "card" }, h("h2", {}, "Reports"), list),
      h("div", { class: "card" }, detail));
  }

  // ---- share
  async function shareTab(m) {
    const cs = await GET("/api/v1/community/status");
    if (!cs.signed_in) return h("div", { class: "card" }, h("p", {}, "Sign in to the community first (Community page) to share this model."));
    const desc = h("textarea", { placeholder: "Describe the model: cartridge, image source, quirks…" });
    const mode = select([["ModelAndImages", "Model and images"], ["ModelOnly", "Model only"], ["ImagesOnly", "Images only"]], "ModelAndImages");
    const fb = h("input", { type: "checkbox" }); const floor = input({ type: "number", value: 95, min: 50, max: 100 });
    const status = h("div", {});
    let tracking = null;
    poll(async () => { if (!tracking) return; const j = await GET(`/api/v1/jobs/${tracking}`); status.innerHTML = ""; status.append(jobRow(j)); if (j.status === "done") { toast("Shared", "ok"); tracking = null; } else if (j.status === "failed") { tracking = null; } }, 1500);
    const profile = cs.profile || {};
    return h("div", { class: "card stack" }, h("h2", {}, "Share with the community"),
      profile.can_contribute === false ? h("div", { class: "notice warn" }, "Your community account does not have the Contribute role, so uploads will be refused.") : null,
      m.community_model_uid ? h("p", { class: "muted small" }, `Already shared as ${m.community_model_uid} (v${m.model_version}); sharing again publishes v${m.model_version + 1}.`) : null,
      field("Description", desc), h("div", { class: "row" }, field("What to include", mode), h("label", { class: "check" }, fb, "Enable feedback loop"), field("Confidence floor %", floor)),
      h("button", { class: "primary", onclick: async () => { try { const j = await POST(`/api/v1/models/${m.id}/share`, { description: desc.value, mode: mode.value, feedback_enabled: fb.checked, feedback_floor: Number(floor.value) }); tracking = j.id; toast("Upload started"); } catch (e) { toast(e.message, "bad"); } } }, "Share"), status);
  }

  // ------------------------------------------------------------------ jobs
  async function jobsView() {
    const wrap = h("div", { class: "stack" }, h("h1", {}, "Jobs"));
    const card = h("div", { class: "card" });
    wrap.append(card);
    const refresh = async () => {
      const [jobs, models] = await Promise.all([GET("/api/v1/jobs?limit=100"), GET("/api/v1/models")]);
      card.innerHTML = "";
      if (!jobs.length) { card.append(h("p", { class: "muted" }, "No jobs yet.")); return; }
      card.append(h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Created"), h("th", {}, "Kind"), h("th", {}, "Model"), h("th", {}, "Status"), h("th", {}, "Progress"), h("th", {}, "Result / error"), h("th", {}))),
        h("tbody", {}, jobs.map((j) => { const m = models.find((x) => x.id === j.model_id); const pct = jobPct(j); return h("tr", {}, h("td", {}, fmtDate(j.created_at)), h("td", {}, j.kind), h("td", {}, m ? h("a", { href: `#/models/${m.id}` }, m.name) : (j.request && j.request.name) || (j.model_id ? `#${j.model_id}` : "—")), h("td", {}, statusBadge(j)), h("td", { style: "min-width:120px" }, pct != null ? h("div", { class: "progress" }, h("div", { style: `width:${pct}%` })) : ""), h("td", { class: "small mono" }, j.error ? j.error.slice(0, 120) : j.result && j.result.summary ? `acc ${fmtPct(j.result.summary.total_accuracy)}` : j.result && j.result.best_val_acc != null ? `val acc ${fmtPct(j.result.best_val_acc * 100)}` : ""), h("td", {}, (j.status === "running" || j.status === "queued") ? h("button", { class: "sm danger", onclick: async () => { await POST(`/api/v1/jobs/${j.id}/cancel`); refresh(); } }, "Cancel") : null)); })))));
    };
    poll(() => refresh().catch((e) => toast(e.message, "bad")), 3000);
    return wrap;
  }

  // ------------------------------------------------------------------ community
  async function communityView() {
    const wrap = h("div", { class: "stack" }, h("h1", {}, "Community"));
    const authCard = h("div", { class: "card stack" });
    const catCard = h("div", { class: "card stack" });
    wrap.append(authCard, catCard);
    const renderAuth = async () => {
      const s = await GET("/api/admin/community/status");
      authCard.innerHTML = "";
      if (s.available === false) { authCard.append(h("div", { class: "notice warn" }, `Community features unavailable: ${s.error}`)); return s; }
      if (s.signed_in) {
        authCard.append(h("div", { class: "row between" }, h("div", {}, h("h2", {}, "Signed in"), h("p", {}, `${s.name || ""} ${s.email ? `<${s.email}>` : ""}`.trim() || "Community account", s.profile && s.profile.profile_name ? ` · handle ${s.profile.profile_name}` : "", s.profile ? (s.profile.can_contribute ? " · can share models" : " · read-only account") : "")),
          h("button", { onclick: async () => { if (!(await confirmModal("Sign out", "Sign out of the community?"))) return; await POST("/api/admin/community/logout"); renderAuth(); loadCatalogue(); } }, "Sign out")));
      } else {
        const pasteIn = h("input", { type: "text", placeholder: "http://localhost:44300/?code=…" });
        const pending = s.login_pending;
        authCard.append(h("h2", {}, "Sign in to reloadingrecipes.com"),
          h("p", { class: "muted small" }, "Uses the same account as the desktop client. The sign-in page redirects to http://localhost:44300/ — if your browser runs on this server the sign-in completes by itself; otherwise copy the address of the page you land on and paste it below."),
          h("div", { class: "row" },
            h("button", { class: "primary", onclick: async () => { try { const r = await POST("/api/admin/community/login/start"); window.open(r.auth_url, "_blank"); renderAuth(); } catch (e) { toast(e.message, "bad"); } } }, pending ? "Open sign-in page again" : "Sign in…"),
            pending && s.auth_url ? h("a", { href: s.auth_url, target: "_blank", class: "small" }, "sign-in link") : null,
            pending ? h("button", { class: "sm", onclick: async () => { await POST("/api/admin/community/login/cancel"); renderAuth(); } }, "Cancel") : null),
          pending ? h("div", { class: "stack" }, h("p", { class: "small" }, s.listener_active ? "Waiting for the browser redirect on port 44300…" : "Port 44300 is busy on the server, so paste the redirect address below."), h("div", { class: "row" }, pasteIn, h("button", { onclick: async () => { try { await POST("/api/admin/community/login/complete", { redirect_url: pasteIn.value }); toast("Signed in", "ok"); renderAuth(); loadCatalogue(); } catch (e) { toast(e.message, "bad"); } } }, "Complete sign-in"))) : null,
          s.last_error ? h("div", { class: "notice warn small" }, s.last_error) : null);
      }
      return s;
    };
    const search = h("input", { type: "search", placeholder: "Search models", style: "max-width:240px" });
    const typeSel = select([["", "All types"], ["ModelAndImages", "Model and images"], ["ModelOnly", "Model only"], ["ImagesOnly", "Images only"]], "");
    const table = h("div", {});
    const loadCatalogue = async () => {
      table.innerHTML = "";
      try {
        const s = await GET("/api/v1/community/status");
        if (!s.signed_in) { table.append(h("p", { class: "muted" }, "Sign in to browse the catalogue.")); return; }
        table.append(h("p", { class: "muted" }, "Loading catalogue…"));
        const models = await GET(`/api/v1/community/models?search=${encodeURIComponent(search.value)}&model_type=${encodeURIComponent(typeSel.value)}`);
        table.innerHTML = "";
        if (!models.length) { table.append(h("p", { class: "muted" }, "No models found.")); return; }
        table.append(h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Model"), h("th", {}, "Cartridge"), h("th", {}, "Author"), h("th", {}, "Version"), h("th", {}, "Contents"), h("th", {}, "Images"), h("th", {}, "Size"), h("th", {}, "Published"), h("th", {}))),
          h("tbody", {}, models.map((c) => h("tr", {}, h("td", { title: c.model_description }, h("strong", {}, c.model_name), c.model_description ? h("div", { class: "muted small" }, c.model_description.slice(0, 120)) : null), h("td", {}, c.cartridge_name), h("td", {}, c.author), h("td", {}, `v${c.model_version}`), h("td", {}, c.export_mode), h("td", {}, c.image_count), h("td", {}, fmtBytes(c.download_size)), h("td", {}, fmtDate(c.publish_date)),
            h("td", {}, c.state === "installed" ? h("a", { href: `#/models/${c.local_model_id}` }, badge("installed", "ok")) : h("button", { class: c.state === "update" ? "primary sm" : "sm", onclick: async () => { try { const j = await POST(`/api/v1/community/models/${encodeURIComponent(c.model_uid)}/download`, { update_existing: true }); toast(`${c.state === "update" ? "Updating" : "Downloading"} ${c.model_name}…`); trackJobs(); } catch (e) { toast(e.message, "bad"); } } }, c.state === "update" ? "Update" : "Download"))))))));
      } catch (e) { table.innerHTML = ""; table.append(h("div", { class: "notice warn" }, e.message)); }
    };
    const jobsBox = h("div", {});
    const trackJobs = async () => { const jobs = (await GET("/api/v1/jobs?kind=download&limit=10")).filter((j) => j.status === "running" || j.status === "queued" || (Date.now() - new Date(j.finished_at || 0)) < 30000); jobsBox.innerHTML = ""; for (const j of jobs) jobsBox.append(jobRow(j)); if (jobs.some((j) => j.status === "done" && (Date.now() - new Date(j.finished_at)) < 5000)) loadCatalogue(); };
    catCard.append(h("div", { class: "row between" }, h("h2", {}, "Model catalogue"), h("div", { class: "row" }, search, typeSel, h("button", { onclick: loadCatalogue }, "Search"))), jobsBox, table);
    search.addEventListener("keydown", (e) => { if (e.key === "Enter") loadCatalogue(); });
    typeSel.addEventListener("change", loadCatalogue);
    const s = await renderAuth();
    if (s.login_pending) poll(async () => { const n = await GET("/api/admin/community/status"); if (n.signed_in) { toast("Signed in", "ok"); render(); } }, 3000);
    poll(() => trackJobs().catch(() => {}), 3000);
    loadCatalogue();
    return wrap;
  }

  // ------------------------------------------------------------------ clients
  async function clientsView() {
    const wrap = h("div", { class: "stack" }, h("h1", {}, "Bound clients"));
    const codeCard = h("div", { class: "card stack" });
    const listCard = h("div", { class: "card stack" });
    wrap.append(h("div", { class: "notice" }, "A light-weight CaseSorter client in remote mode binds to this server with a one-time pairing code, then creates, trains and manages its models here. The server does the GPU/CPU work; the client keeps the camera and the sorting machine."), codeCard, listCard);
    const refresh = async () => {
      const data = await GET("/api/admin/clients");
      codeCard.innerHTML = "";
      const label = input({ placeholder: "Label (e.g. Shop PC)", style: "max-width:240px" });
      codeCard.append(h("h2", {}, "Pair a new client"), h("div", { class: "row" }, label, h("button", { class: "primary", onclick: async () => { await POST("/api/admin/clients/pairing-code", { label: label.value }); refresh(); } }, "Generate pairing code")),
        data.pairing_codes.length ? h("div", { class: "stack" }, data.pairing_codes.map((c) => h("div", { class: "row" }, h("code", { style: "font-size:20px;letter-spacing:2px" }, c.code), h("span", { class: "muted small" }, `${c.label || ""} · expires ${fmtDate(c.expires_at)}`)))) : h("p", { class: "muted small" }, "No active pairing codes."),
        h("p", { class: "muted small" }, `In the client, enter this server's address (http://${location.host}) and the code. Codes are single-use and expire after 15 minutes.`),
        h("details", {}, h("summary", { class: "small" }, "Manual / scripted binding"), h("pre", { class: "log" }, `POST http://${location.host}/api/v1/bind\n{"pairing_code": "XXXX-XXXX", "client_name": "Shop PC"}\n→ {"token": "csk_…"}   then   Authorization: Bearer csk_…`),
          h("div", { class: "row" }, h("button", { class: "sm", onclick: async () => { const r = await POST("/api/admin/clients/token", { name: label.value || "scripted client" }); modal("Client token", h("div", { class: "stack" }, h("p", {}, "Copy it now — it is not shown again."), h("pre", { class: "log" }, r.token)), []); refresh(); } }, "Issue a token directly"))));
      listCard.innerHTML = "";
      listCard.append(h("h2", {}, "Clients"));
      if (!data.clients.length) listCard.append(h("p", { class: "muted" }, "No clients bound yet."));
      else listCard.append(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Bound"), h("th", {}, "Last seen"), h("th", {}, "Version"), h("th", {}, "Status"), h("th", {}))),
        h("tbody", {}, data.clients.map((c) => h("tr", {}, h("td", {}, c.name), h("td", {}, fmtDate(c.created_at)), h("td", {}, fmtDate(c.last_seen_at)), h("td", {}, c.client_version || "—"), h("td", {}, c.revoked ? badge("revoked", "bad") : badge("active", "ok")),
          h("td", { class: "row" }, h("button", { class: "sm", onclick: () => { const n = input({ value: c.name }); modal("Rename client", field("Name", n), [{ label: "Save", class: "primary", onclick: async () => { await POST(`/api/admin/clients/${c.id}/rename`, { name: n.value }); refresh(); } }]); } }, "Rename"),
            !c.revoked ? h("button", { class: "sm danger", onclick: async () => { if (!(await confirmModal("Revoke", `Revoke ${c.name}? It will have to pair again.`))) return; await POST(`/api/admin/clients/${c.id}/revoke`); refresh(); } }, "Revoke") : h("button", { class: "sm", onclick: async () => { await DEL(`/api/admin/clients/${c.id}`); refresh(); } }, "Remove")))))));
    };
    await refresh();
    return wrap;
  }

  // ------------------------------------------------------------------ settings
  async function settingsView() {
    const s = await GET("/api/admin/settings");
    const allow = h("input", { type: "checkbox", checked: s.allow_remote_clients });
    const autoDl = h("input", { type: "checkbox", checked: s.auto_serve_downloads });
    const autoTr = h("input", { type: "checkbox", checked: s.auto_serve_trained });
    const dev = select([["auto", "Auto (GPU when available)"], ["cuda", "GPU (CUDA)"], ["cpu", "CPU"]], s.training_device);
    const cur = input({ type: "password", placeholder: "Current password" }), pw = input({ type: "password", placeholder: "New password" });
    const cfg = s.config;
    return h("div", { class: "grid cols-2" },
      h("div", { class: "card stack" }, h("h2", {}, "Behaviour"),
        h("label", { class: "check" }, allow, "Allow light-weight clients to bind and use the remote API"),
        h("label", { class: "check" }, autoDl, "Serve community downloads automatically"),
        h("label", { class: "check" }, autoTr, "Serve a model automatically when its training finishes"),
        field("Training device", dev),
        h("button", { class: "primary", onclick: async () => { try { await PATCH("/api/admin/settings", { allow_remote_clients: allow.checked, auto_serve_downloads: autoDl.checked, auto_serve_trained: autoTr.checked, training_device: dev.value }); toast("Saved", "ok"); } catch (e) { toast(e.message, "bad"); } } }, "Save")),
      h("div", { class: "stack" },
        h("div", { class: "card stack" }, h("h2", {}, "Admin password"),
          s.admin_password_set ? field("Current", cur) : h("p", { class: "muted small" }, "No password set (loopback-only server). Set one to require sign-in."),
          field("New password", pw),
          h("button", { onclick: async () => { try { if (s.admin_password_set) { await POST("/api/admin/password", { password: pw.value, current: cur.value }); } else { await POST("/api/admin/setup", { password: pw.value }); } toast("Password updated", "ok"); state.auth = null; render(); } catch (e) { toast(e.message, "bad"); } } }, "Change password")),
        h("div", { class: "card" }, h("h2", {}, "config.py (read-only)"),
          h("dl", { class: "kv" }, h("dt", {}, "HOST"), h("dd", { class: "mono" }, cfg.HOST), h("dt", {}, "PORT"), h("dd", { class: "mono" }, cfg.PORT), h("dt", {}, "API_KEY"), h("dd", {}, cfg.API_KEY_set ? "set" : "not set (OpenAI endpoints are open)"), h("dt", {}, "PRELOAD_MODELS"), h("dd", {}, String(cfg.PRELOAD_MODELS)), h("dt", {}, "LOG_LEVEL"), h("dd", {}, cfg.LOG_LEVEL), h("dt", {}, "DATA_DIR"), h("dd", { class: "mono" }, cfg.DATA_DIR), h("dt", {}, "MODELS"), h("dd", { class: "mono" }, Object.keys(cfg.MODELS).length ? Object.entries(cfg.MODELS).map(([a, p]) => `${a} → ${p}`).join("\n") : "none")),
          h("p", { class: "muted small" }, "Edit config.py and restart the server to change these. Models trained or downloaded here are served from the registry and don't need config.py entries."))));
  }

  render();
})();
