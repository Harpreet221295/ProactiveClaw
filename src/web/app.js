/* ProactiveClaw web UI — vanilla JS, no build step. */
(() => {
  const $ = (s) => document.querySelector(s);
  const messagesEl = $("#messages"), inputEl = $("#input"), typingEl = $("#typing"), typingText = $("#typing-text");
  const careList = $("#care-list"), statusPill = $("#status-pill"), statusText = $("#status-text");
  const levelPill = $("#level-pill"), modePill = $("#mode-pill"), toastEl = $("#toast");
  let ws = null, state = null, configData = null, careView = "open", pendingImages = [], reconnectDelay = 1000;
  let lastDay = null;

  // ── helpers ────────────────────────────────────────────────────────────
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const md = (text) => {
    let html;
    try { html = window.marked ? marked.parse(text || "", { breaks: true, gfm: true }) : esc(text).replace(/\n/g, "<br>"); }
    catch { html = esc(text).replace(/\n/g, "<br>"); }
    return window.DOMPurify ? DOMPurify.sanitize(html, { ADD_ATTR: ["target"] }) : html;
  };
  const fmtTime = (iso) => { try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); } catch { return ""; } };
  const fmtDay = (iso) => { try { return new Date(iso).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" }); } catch { return ""; } };
  const fmtWhen = (iso) => { if (!iso) return ""; const d = new Date(iso); const now = new Date();
    const same = d.toDateString() === now.toDateString(); return (same ? "" : fmtDay(iso) + " ") + fmtTime(iso); };
  const toast = (msg, ms = 2600) => { toastEl.textContent = msg; toastEl.classList.remove("hidden"); clearTimeout(toast._t); toast._t = setTimeout(() => toastEl.classList.add("hidden"), ms); };
  const api = async (path, opts = {}) => {
    const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    if (!r.ok) { let t = ""; try { t = (await r.json()).detail || ""; } catch {} throw new Error(t || r.statusText); }
    return r.json();
  };
  const scrollBottom = () => { messagesEl.scrollTop = messagesEl.scrollHeight; };

  // ── messages ───────────────────────────────────────────────────────────
  const TAGS = { user: "You", assistant: "ProactiveClaw", nudge: "Nudge", reminder: "Reminder", morning_review: "Morning review", system: "System" };
  function addMessage(m, { scroll = true } = {}) {
    const day = fmtDay(m.at);
    if (day && day !== lastDay) { const sep = document.createElement("div"); sep.className = "day-sep"; sep.textContent = day; messagesEl.appendChild(sep); lastDay = day; }
    const el = document.createElement("div");
    el.className = `msg ${m.role}`;
    let tag = TAGS[m.role] || m.role;
    if (m.role === "nudge" && m.source === "reengagement") tag = "Checking in";
    if (m.role === "assistant" && m.source === "pre_exit") tag = "ProactiveClaw · signing off";
    if (m.role === "assistant" && m.source === "cron") tag = "Scheduled task";
    const files = (m.files || []).map((f) => `<a href="/files/${encodeURI(f)}" target="_blank">📄 ${esc(f.split("/").pop())}</a>`).join("");
    el.innerHTML = `<div class="tag"><span>${esc(tag)}</span><span>${fmtTime(m.at)}</span></div><div class="body">${m.role === "user" ? esc(m.text).replace(/\n/g, "<br>") : md(m.text)}</div>${files ? `<div class="files">${files}</div>` : ""}`;
    if (m.role === "user" && m.meta && m.meta.images) { const n = document.createElement("div"); n.className = "care-meta"; n.textContent = `📎 ${m.meta.images} image(s)`; el.appendChild(n); }
    messagesEl.appendChild(el);
    if (scroll) scrollBottom();
    if (["nudge", "reminder", "morning_review"].includes(m.role)) notify(tag, m.text);
  }
  function notify(title, body) {
    if (!("Notification" in window) || Notification.permission !== "granted" || document.visibilityState === "visible") return;
    try { new Notification(`ProactiveClaw — ${title}`, { body: (body || "").slice(0, 180) }); } catch {}
  }

  // ── status ─────────────────────────────────────────────────────────────
  function setStatus(stateName, detail) {
    statusPill.className = "pill " + stateName;
    const labels = { awake: "awake", sleeping: "sleeping", thinking: "thinking…", tool: `using ${detail || "a tool"}`, pre_exit: "wrapping up the session…", morning_review: "morning review running…", offline: "disconnected" };
    statusText.textContent = labels[stateName] || stateName;
    const busy = ["thinking", "tool", "pre_exit", "morning_review"].includes(stateName);
    typingEl.classList.toggle("hidden", !busy);
    typingText.textContent = labels[stateName] || "";
  }
  function applyState(s) {
    state = s;
    levelPill.textContent = `level: ${s.level}`;
    modePill.textContent = `mode: ${s.care_mode}`;
    setStatus(s.morning_review_running ? "morning_review" : s.activity && s.activity !== "idle" ? (s.activity.startsWith("tool") ? "tool" : "thinking") : (s.sleeping ? "sleeping" : "awake"), s.activity?.split(":")[1]);
    if (careView === "nudges") renderNudges();
  }
  async function refreshState() { try { applyState(await api("/api/state")); } catch {} }

  // ── websocket ──────────────────────────────────────────────────────────
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/ui`);
    ws.onopen = () => { reconnectDelay = 1000; };
    ws.onclose = () => { setStatus("offline"); setTimeout(connect, reconnectDelay); reconnectDelay = Math.min(reconnectDelay * 2, 15000); };
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      switch (m.type) {
        case "hello": messagesEl.innerHTML = ""; lastDay = null; (m.history || []).forEach((h) => addMessage(h, { scroll: false })); scrollBottom(); applyState(m.state); loadCare(); break;
        case "message": addMessage(m); if (m.role !== "user") refreshState(); break;
        case "status": setStatus(m.state, m.detail); break;
        case "care_updated": loadCare(); refreshState(); break;
        case "queue_updated": refreshState(); if (careView === "nudges") renderNudges(); break;
        case "config_updated": refreshState(); if (!$("#settings").classList.contains("hidden")) loadSettings(); break;
        case "morning_review": if (m.state === "running") toast("Morning review running…"); if (m.state === "done") toast(m.delivered ? "Morning review complete" : "Morning review complete — nothing to report"); if (m.state === "error") toast("Morning review failed: " + m.error, 5000); break;
      }
    };
  }

  // ── composer ───────────────────────────────────────────────────────────
  function send() {
    const text = inputEl.value.trim();
    if ((!text && !pendingImages.length) || !ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify({ type: "message", text, images: pendingImages }));
    inputEl.value = ""; inputEl.style.height = "auto"; pendingImages = []; renderAttachments();
  }
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
  inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  inputEl.addEventListener("input", () => { inputEl.style.height = "auto"; inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px"; });
  $("#btn-attach").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (e) => { [...e.target.files].forEach(addImageFile); e.target.value = ""; });
  inputEl.addEventListener("paste", (e) => { [...(e.clipboardData?.items || [])].filter((i) => i.type.startsWith("image/")).forEach((i) => addImageFile(i.getAsFile())); });
  function addImageFile(file) { const r = new FileReader(); r.onload = () => { pendingImages.push(r.result); renderAttachments(); }; r.readAsDataURL(file); }
  function renderAttachments() { const a = $("#attachments"); a.innerHTML = pendingImages.map((src, i) => `<img src="${src}" title="click to remove" data-i="${i}">`).join(""); a.classList.toggle("hidden", !pendingImages.length);
    a.querySelectorAll("img").forEach((img) => img.addEventListener("click", () => { pendingImages.splice(+img.dataset.i, 1); renderAttachments(); })); }

  // ── care panel ─────────────────────────────────────────────────────────
  document.querySelectorAll(".sidebar-tabs .tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".sidebar-tabs .tab").forEach((x) => x.classList.remove("active")); t.classList.add("active");
    careView = t.dataset.view; careView === "nudges" ? renderNudges() : loadCare();
  }));
  async function loadCare() {
    if (careView === "nudges") return;
    try {
      const data = await api(`/api/care?status=${careView}`);
      renderCare(data.items);
    } catch (e) { careList.innerHTML = `<div class="care-empty">${esc(e.message)}</div>`; }
  }
  function renderCare(items) {
    if (!items.length) { careList.innerHTML = `<div class="care-empty">${careView === "open" ? "Nothing tracked yet. Tell me about a commitment, or add one below." : "Nothing here."}</div>`; return; }
    const now = new Date();
    const groups = { "Overdue": [], "Due soon": [], "Open": [], "Snoozed": [], "Closed": [] };
    for (const it of items) {
      const dl = it.deadline ? new Date(it.deadline) : null;
      if (["done", "dismissed", "archived"].includes(it.status)) groups["Closed"].push(it);
      else if (it._overdue || (dl && dl < now)) groups["Overdue"].push(it);
      else if (it.status === "snoozed") groups["Snoozed"].push(it);
      else if (dl && dl - now < 36 * 3600e3) groups["Due soon"].push(it);
      else groups["Open"].push(it);
    }
    careList.innerHTML = "";
    for (const [name, list] of Object.entries(groups)) {
      if (!list.length) continue;
      const h = document.createElement("div"); h.className = "care-group"; h.textContent = `${name} · ${list.length}`; careList.appendChild(h);
      list.forEach((it) => careList.appendChild(careCard(it)));
    }
  }
  function careCard(it) {
    const el = document.createElement("div");
    el.className = "care-item" + (it._overdue ? " overdue" : "");
    const meta = [];
    if (it.sender) meta.push(`from ${esc(it.sender)}`);
    if (it.deadline) meta.push(`due ${fmtWhen(it.deadline)}`);
    if (it.snooze_until) meta.push(`snoozed until ${fmtWhen(it.snooze_until)}`);
    meta.push(esc(it.type) + (it.source && it.source !== "conversation" ? ` · ${esc(it.source)}` : ""));
    if (it.reminder_count) meta.push(`nudged ${it.reminder_count}×`);
    if (it.deferral_count) meta.push(`deferred ${it.deferral_count}×`);
    const closed = ["done", "dismissed", "archived"].includes(it.status);
    const actions = closed
      ? `<button data-a="reopen">Reopen</button><button data-a="delete">Delete</button>`
      : `<button data-a="done">✓ Done</button><button data-a="snooze" data-until="3h">Snooze 3h</button><button data-a="snooze" data-until="tomorrow">Tomorrow</button><button data-a="dismiss">Dismiss</button>`;
    el.innerHTML = `<div class="care-title"><span class="urg ${esc(it.urgency)}" title="${esc(it.urgency)}"></span><span>${esc(it.title)}</span></div>
      <div class="care-meta">${meta.join(" · ")}</div>
      ${it.user_intent ? `<div class="care-intent">“${esc(it.user_intent)}”</div>` : ""}
      ${it.details ? `<div class="care-meta">${esc(it.details.slice(0, 160))}</div>` : ""}
      <div class="care-actions">${actions}</div>`;
    el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => careAction(it.id, b.dataset.a, b.dataset.until)));
    return el;
  }
  async function careAction(id, action, until) {
    const body = { action };
    if (action === "snooze") {
      const d = new Date();
      if (until === "3h") d.setHours(d.getHours() + 3); else { d.setDate(d.getDate() + 1); d.setHours(9, 0, 0, 0); }
      body.until = d.toISOString();
    }
    try { await api(`/api/care/${id}`, { method: "POST", body: JSON.stringify(body) }); loadCare(); }
    catch (e) { toast("Failed: " + e.message); }
  }
  $("#care-add").addEventListener("submit", async (e) => {
    e.preventDefault(); const title = $("#care-add-title").value.trim(); if (!title) return;
    try { await api("/api/care", { method: "POST", body: JSON.stringify({ title }) }); $("#care-add-title").value = ""; careView = "open"; document.querySelector('.tab[data-view="open"]').click(); }
    catch (err) { toast("Failed: " + err.message); }
  });
  function renderNudges() {
    if (!state) return;
    const rows = [...state.pending_nudges.map((n) => ({ ...n, kind: "nudge" })), ...state.reminders.map((r) => ({ ...r, kind: "reminder" }))]
      .sort((a, b) => (a.timestamp || "").localeCompare(b.timestamp || ""));
    careList.innerHTML = "";
    const info = document.createElement("div"); info.className = "care-meta";
    info.style.padding = "2px 4px 6px";
    info.innerHTML = `Nudges today: <b>${state.nudges_delivered_today}</b> delivered · cap <b>${state.effective.max_nudges_per_day}</b>/day · quiet ${esc(state.effective.quiet_hours.start)}–${esc(state.effective.quiet_hours.end)}<br>Morning review: ${state.morning_review.enabled ? `daily at ${esc(state.morning_review.time)}` : "off"}${state.morning_review.last_run ? ` · last ${fmtWhen(state.morning_review.last_run)}` : ""}`;
    careList.appendChild(info);
    if (!rows.length) { const e = document.createElement("div"); e.className = "care-empty"; e.textContent = "Nothing scheduled."; careList.appendChild(e); return; }
    for (const r of rows) {
      const el = document.createElement("div"); el.className = "care-item nudge-item";
      el.innerHTML = `<div class="care-title">${r.kind === "reminder" ? "🔔" : "📣"} <span>${esc(r.message)}</span></div><div class="care-meta">${fmtWhen(r.timestamp)}${r.source ? " · " + esc(r.source) : ""}</div><div class="care-actions"><button data-id="${esc(r.id)}">Cancel</button></div>`;
      el.querySelector("button").addEventListener("click", async () => { try { await api(`/api/nudges/${r.id}`, { method: "DELETE" }); refreshState(); } catch (e) { toast(e.message); } });
      careList.appendChild(el);
    }
  }

  // ── top bar actions ────────────────────────────────────────────────────
  $("#btn-review").addEventListener("click", async () => { try { await api("/api/morning-review/run", { method: "POST" }); toast("Morning review started"); } catch (e) { toast(e.message); } });
  $("#btn-sleep").addEventListener("click", async () => { try { const r = await api("/api/sleep", { method: "POST" }); toast(r.already_sleeping ? "Already sleeping" : "Wrapping up the session…"); } catch (e) { toast(e.message); } });
  levelPill.addEventListener("click", openSettings); modePill.addEventListener("click", openSettings);

  // ── settings ───────────────────────────────────────────────────────────
  const settingsEl = $("#settings");
  $("#btn-settings").addEventListener("click", openSettings);
  $("#btn-settings-close").addEventListener("click", () => settingsEl.classList.add("hidden"));
  settingsEl.addEventListener("click", (e) => { if (e.target === settingsEl) settingsEl.classList.add("hidden"); });
  let draft = {};
  async function openSettings() { settingsEl.classList.remove("hidden"); $("#settings-msg").textContent = ""; await loadSettings(); }
  async function loadSettings() {
    try { configData = await api("/api/config"); } catch (e) { toast(e.message); return; }
    const c = configData.config;
    draft = { level: c.proactiveness.level, care_mode: c.care_mode, connectors: { ...c.connectors } };
    renderLevels(); renderModes(); renderConnectors();
    const ov = c.overrides || {};
    $("#ov-max_nudges_per_day").value = ov.max_nudges_per_day ?? "";
    $("#ov-qh-start").value = ov.quiet_hours?.start ?? ""; $("#ov-qh-end").value = ov.quiet_hours?.end ?? "";
    $("#ov-urgency_threshold").value = ov.urgency_threshold ?? ""; $("#ov-commitment_capture").value = ov.commitment_capture ?? "";
    $("#ov-morning_review_dm").value = ov.morning_review_dm ?? "";
    $("#ov-boosted_topics").value = (ov.boosted_topics || []).join(", "); $("#ov-muted_topics").value = (ov.muted_topics || []).join(", ");
    $("#mr-time").value = c.morning_review.time || "";
    $("#user-name").value = c.user.name || ""; $("#user-tz").value = c.user.timezone || ""; $("#llm-model").value = c.llm.model || "";
    $("#notion-db").value = c.notion.tasks_database_id || "";
    $("#config-summary").textContent = configData.summary;
  }
  function renderLevels() {
    $("#levels").innerHTML = configData.level_order.map((l) => `<div class="level ${draft.level === l ? "active" : ""}" data-l="${l}"><b>${l}</b><span>${esc(configData.levels[l])}</span></div>`).join("");
    $("#levels").querySelectorAll(".level").forEach((el) => el.addEventListener("click", () => { draft.level = el.dataset.l; renderLevels(); }));
  }
  function renderModes() {
    const modes = Object.keys(configData.modes);
    $("#modes").innerHTML = modes.map((m) => `<div class="mode ${draft.care_mode === m ? "active" : ""}" data-m="${m}">${m.replace("_", " ")}</div>`).join("") + `<div class="mode-desc">${esc(configData.modes[draft.care_mode])}</div>`;
    $("#modes").querySelectorAll(".mode").forEach((el) => el.addEventListener("click", () => { draft.care_mode = el.dataset.m; renderModes(); }));
  }
  function renderConnectors() {
    const st = configData.connectors;
    const help = { gmail: "Read your inbox for things that need action (never sends unless you ask).", calendar: "See your events, avoid nudging during meetings, create events on request.", notion: "Read/query your Notion pages and task database.", web_search: "Look things up on the web (Tavily).", browser: "Drive your Chrome via the extension, only when you ask." };
    $("#connectors").innerHTML = Object.keys(st).map((n) => `<div class="connector"><div><div class="name">${n.replace("_", " ")}</div><div class="reason">${esc(help[n] || "")}${st[n].reason && !st[n].active ? ` — <i>${esc(st[n].reason)}</i>` : ""}</div></div><div style="display:flex;gap:8px;align-items:center"><span class="st ${st[n].active ? "active" : "inactive"}">${st[n].active ? "active" : st[n].enabled ? "not configured" : "off"}</span>${(n === "gmail" || n === "calendar") && st[n].enabled && !st[n].reason.includes("credentials.json") && !st[n].active ? `<button class="btn btn-small" data-g="1">Connect Google</button>` : ""}<label class="switch"><input type="checkbox" data-c="${n}" ${draft.connectors[n] ? "checked" : ""}><span class="slider"></span></label></div></div>`).join("");
    $("#connectors").querySelectorAll("input[data-c]").forEach((i) => i.addEventListener("change", () => { draft.connectors[i.dataset.c] = i.checked; }));
    $("#connectors").querySelectorAll("button[data-g]").forEach((b) => b.addEventListener("click", async () => { b.disabled = true; b.textContent = "Check your browser…"; try { await api("/api/connectors/google/connect", { method: "POST" }); toast("Google connected"); await loadSettings(); } catch (e) { toast(e.message, 6000); b.disabled = false; b.textContent = "Connect Google"; } }));
  }
  $("#btn-settings-save").addEventListener("click", async () => {
    const list = (v) => v.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
    const overrides = {};
    const n = $("#ov-max_nudges_per_day").value; if (n !== "") overrides.max_nudges_per_day = parseInt(n, 10);
    const qs = $("#ov-qh-start").value, qe = $("#ov-qh-end").value; if (qs && qe) overrides.quiet_hours = { start: qs, end: qe };
    for (const k of ["urgency_threshold", "commitment_capture", "morning_review_dm"]) { const v = $(`#ov-${k}`).value; if (v) overrides[k] = v; }
    const bt = list($("#ov-boosted_topics").value); if (bt.length) overrides.boosted_topics = bt;
    const mt = list($("#ov-muted_topics").value); if (mt.length) overrides.muted_topics = mt;
    const body = { level: draft.level, care_mode: draft.care_mode, connectors: draft.connectors, overrides,
      user: { name: $("#user-name").value.trim(), timezone: $("#user-tz").value.trim() },
      llm: { model: $("#llm-model").value.trim() }, notion: { tasks_database_id: $("#notion-db").value.trim() },
      morning_review: { time: $("#mr-time").value || "07:30" } };
    try { configData = await api("/api/config", { method: "PUT", body: JSON.stringify(body) }); $("#settings-msg").textContent = "Saved."; $("#config-summary").textContent = configData.summary; renderConnectors(); refreshState(); toast("Settings saved"); }
    catch (e) { $("#settings-msg").textContent = "Error: " + e.message; }
  });

  // ── boot ───────────────────────────────────────────────────────────────
  if ("Notification" in window && Notification.permission === "default") { document.addEventListener("click", () => Notification.requestPermission(), { once: true }); }
  connect();
  setInterval(refreshState, 60000);
})();
