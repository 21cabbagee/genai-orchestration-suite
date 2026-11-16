import { createPipeline, estimateRoi } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const LABELS = { expense: "Expense", it_access: "IT access", procurement: "Procurement", refund: "Refund", hr_leave: "HR / leave", data_privacy: "Data privacy" };
const RES = { auto_resolved: ["✅ Auto-resolved", "ok", 15], agent_assist: ["🤝 Drafted for review", "warn", 10], escalated: ["🚨 Escalated", "bad", 10] };
const AGENTS = [
  ["intake", "🛡️", "Intake"], ["router", "🧭", "Router"], ["research", "🔎", "Research"],
  ["analyst", "✍️", "Analyst"], ["critic", "🧐", "Critic"], ["responder", "📬", "Responder"],
];

const QUICK = [
  ["💳", "Late expense claim", "My expense claim from the Delhi trip is 40 days old, can I still submit it? Reach me at ravi.k@acme.in"],
  ["🔁", "Quotes for new monitors", "How many quotes do we need to buy 20 monitors worth 4 lakh?"],
  ["🚨", "Lost laptop, urgent", "URGENT: I lost my laptop at the airport this morning"],
  ["🛡️", "Data request with PII", "Customer wants a copy of their data, call them on 9876543210 or email priya@example.com"],
  ["🕳️", "Something off-policy", "What's on the cafeteria menu this Friday?"],
];

const ACHIEVEMENTS = [
  { id: "boot", icon: "🚀", title: "Boot sequence", xp: 20, desc: "Ran the agent squad for the first time.", hint: "Deploy the agents on any request." },
  { id: "shield", icon: "🛡️", title: "Privacy shield", xp: 30, desc: "Intake redacted personal data before any model saw it.", hint: "Include an email address or phone number in a request." },
  { id: "fortknox", icon: "🔐", title: "Fort Knox", xp: 60, desc: "Redacted three different kinds of PII in a single request.", hint: "Mix an email, a phone number and a PAN like ABCDE1234F in one message." },
  { id: "retry", icon: "🔁", title: "Second opinion", xp: 40, desc: "The critic rejected a weak draft and forced a better retrieval.", hint: "Use wording the policies don't, like “quotes” instead of “quotations”." },
  { id: "redalert", icon: "🚨", title: "Red alert", xp: 30, desc: "A high-priority request was escalated to a human team.", hint: "Send something urgent: a lost device, a breach, a chargeback…" },
  { id: "loop", icon: "🤝", title: "Human in the loop", xp: 40, desc: "The squad wasn't confident, so it drafted for human review.", hint: "Ask something the policies only partly cover." },
  { id: "stump", icon: "🕳️", title: "Off the map", xp: 40, desc: "Asked something no policy covers, and the squad admitted it.", hint: "Ask about a topic none of the six policies mention." },
  { id: "coverage", icon: "🗂️", title: "Full coverage", xp: 100, goal: 6, desc: "Auto-resolved a request in all six intents.", hint: "Auto-resolve one request each for expense, IT, procurement, refund, HR and privacy." },
  { id: "author", icon: "✍️", title: "Free thinker", xp: 20, desc: "Ran a request you wrote yourself.", hint: "Write your own request instead of using a quick mission." },
  { id: "whale", icon: "💰", title: "Business case closed", xp: 50, desc: "Pushed modelled first-year ROI past 1,000%.", hint: "In the boss level, tune assumptions until first-year ROI tops 1,000%." },
  { id: "master", icon: "👑", title: "Chief Orchestrator", xp: 150, master: true, desc: "Unlocked every mission. The squad salutes you.", hint: "Unlock every other mission." },
];

const game = Game.create({
  storageKey: "agentops.v1",
  achievements: ACHIEVEMENTS,
  levelNames: ["Intern", "Operator", "Analyst", "Engineer", "Architect", "Principal", "Legend"],
  onRender: (st) => {
    const next = ACHIEVEMENTS.filter((a) => !a.master && !st.unlocked[a.id]).slice(0, 3);
    const el = $("objectives");
    if (!el) return;
    el.innerHTML = next.length
      ? next.map((a) => `<div class="goal"><span class="gi">${a.icon}</span><span><b>${esc(a.title)}</b> · ${esc(a.hint)}</span></div>`).join("")
      : `<div class="goal done"><span class="gi">👑</span><span>All missions complete. You're a Chief Orchestrator.</span></div>`;
  },
});

function fmtDetail(d) {
  return Object.entries(d)
    .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(" | ") : typeof v === "object" && v ? JSON.stringify(v) : v}`)
    .join("\n");
}

async function main() {
  const [model, runs, roi] = await Promise.all(
    ["data/model.json", "data/runs.json", "data/roi.json"].map((p) => fetch(p).then((r) => r.json()))
  );
  const pipeline = createPipeline(model);

  // lane
  $("lane").insertAdjacentHTML("beforeend", AGENTS.map(([id, icon, name]) => `<div class="ag" data-agent="${id}"><div class="orb">${icon}</div>${name}</div>`).join(""));
  const laneNodes = Object.fromEntries([...document.querySelectorAll(".ag")].map((n) => [n.dataset.agent, n]));
  function laneReset() {
    Object.values(laneNodes).forEach((n) => n.classList.remove("active", "done", "retry"));
    $("laneFill").style.width = "0";
  }
  function laneSet(agent) {
    const idx = AGENTS.findIndex((a) => a[0] === agent);
    Object.entries(laneNodes).forEach(([id, n]) => {
      const i = AGENTS.findIndex((a) => a[0] === id);
      n.classList.toggle("active", id === agent);
      if (i < idx) n.classList.add("done");
    });
    $("laneFill").style.width = `${(idx / (AGENTS.length - 1)) * 84}%`;
  }

  // quick missions
  $("chips").innerHTML = QUICK.map(([icon, title], i) => `<button data-i="${i}"><span class="ci">${icon}</span><span class="ct">${esc(title)}</span><span class="cx">QUICK MISSION</span></button>`).join("");
  $("chips").addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    $("req").value = QUICK[+b.dataset.i][2];
    run();
  });
  $("req").value = "";

  let busy = false;
  async function run() {
    const text = $("req").value.trim();
    if (!text || busy) { if (!text) $("req").focus(); return; }
    busy = true;
    $("run").classList.add("busy");
    $("run").textContent = "⏳ Agents working…";
    const s = pipeline.run(text);

    // animate the trace through the lane
    laneReset();
    $("trace").innerHTML = "";
    $("output").innerHTML = `<div class="empty"><div class="big">⚙️</div>Agents are on it…</div>`;
    let lastAgent = null;
    for (const step of s.trace) {
      if (step.agent !== lastAgent) { laneSet(step.agent); lastAgent = step.agent; }
      if (step.agent === "critic" && String(step.detail.verdict).startsWith("retry")) laneNodes.critic.classList.add("retry");
      const li = document.createElement("li");
      li.innerHTML = `<span class="agent">${esc(step.agent)}</span><div><div class="action">${esc(step.action)}</div><div class="detail">${esc(fmtDetail(step.detail)).replace(/\n/g, "<br>")}</div></div>`;
      $("trace").appendChild(li);
      if (!reduced) await sleep(230);
    }
    Object.values(laneNodes).forEach((n) => { n.classList.remove("active"); n.classList.add("done"); });
    $("laneFill").style.width = "84%";

    const [resLabel, resCls, xp] = RES[s.resolution];
    const route = s.resolution === "escalated" ? `Routed to <b>${esc(s.queue)}</b> as high priority.`
      : s.resolution === "agent_assist" ? `Draft queued for review by <b>${esc(s.queue)}</b>.`
      : `Owning queue: ${esc(s.queue)}.`;
    const meter = (label, v) => `<div class="meter">${label} <b>${Math.round(v * 100)}%</b><div class="track"><i data-w="${Math.min(100, v * 100)}"></i></div></div>`;
    $("output").innerHTML = `<div class="result">
      <div class="result-top"><span class="badge ${resCls}">${resLabel}</span><span class="score">+${xp} XP</span></div>
      <div class="badges">
        <span class="badge acc">${LABELS[s.category]}</span>
        <span class="badge ${s.priority === "high" ? "bad" : ""}">Priority: ${s.priority}</span>
        ${s.attempts ? `<span class="badge warn">🔁 ${s.attempts} retr${s.attempts > 1 ? "ies" : "y"}</span>` : ""}
        ${Object.keys(s.pii).length ? `<span class="badge warn">🛡️ ${Object.entries(s.pii).map(([k, v]) => `${k}×${v}`).join(", ")}</span>` : ""}
      </div>
      <div class="answer">${esc(s.draft).replace(/\[(\d+)\]/g, "<sup>[$1]</sup>")}</div>
      <div class="meters">${meter("Intent confidence", s.confidence)}${meter("Grounding", s.grounding)}${meter("Coverage", s.coverage)}</div>
      <div class="sources">Sources: ${s.sources.map(esc).join("; ") || "none"}<br>${route}</div>
    </div>`;
    requestAnimationFrame(() => document.querySelectorAll(".meter i").forEach((i) => (i.style.width = i.dataset.w + "%")));

    const max = Math.max(...s.retrieved.map((h) => h.score), 0.01);
    $("passages").innerHTML = s.retrieved.map((h, i) => `
      <div class="passage">
        <header><span>[${i + 1}] ${esc(h.chunk.title)} › ${esc(h.chunk.section)}</span><span class="muted">${h.score.toFixed(3)}</span></header>
        <div>${esc(h.chunk.text)}</div>
        <div class="bar"><i style="width:${(h.score / max) * 100}%"></i></div>
      </div>`).join("") || `<p class="muted">No passage matched.</p>`;

    // missions
    game.addXP(xp);
    game.unlock("boot");
    const piiKinds = Object.keys(s.pii).length;
    if (piiKinds) game.unlock("shield");
    if (piiKinds >= 3) game.unlock("fortknox");
    if (s.attempts > 0) game.unlock("retry");
    if (s.resolution === "escalated") game.unlock("redalert");
    if (s.resolution === "agent_assist") game.unlock("loop");
    if (s.draft.startsWith("I could not find")) game.unlock("stump");
    if (s.resolution === "auto_resolved") game.progress("coverage", s.category);
    if (!QUICK.some((q) => q[2] === text)) game.unlock("author");

    busy = false;
    $("run").classList.remove("busy");
    $("run").textContent = "▶ Deploy agents";
  }
  $("run").addEventListener("click", run);
  $("req").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); });

  // batch table
  $("batch").querySelector("tbody").innerHTML = runs.map((r) => {
    const [l, c] = RES[r.resolution];
    return `<tr><td>${esc(r.redacted)}</td><td><span class="badge acc">${LABELS[r.category]}</span> <span class="muted">${(r.category_confidence * 100).toFixed(0)}%</span></td><td class="num">${r.attempts}</td><td class="num">${(r.grounding * 100).toFixed(0)}%</td><td><span class="badge ${c}">${l}</span></td></tr>`;
  }).join("");

  // ROI boss level
  const a = { ...roi.assumptions };
  const mix = roi.mix;
  const SL = [
    ["monthly_volume", "Requests per month", 500, 20000, 500, (v) => v.toLocaleString("en-IN")],
    ["baseline_minutes", "Manual minutes per request", 3, 30, 1, (v) => v],
    ["assist_minutes", "Minutes when editing a draft", 1, 15, 1, (v) => v],
    ["hourly_cost", "Loaded cost per hour", 300, 3000, 50, (v) => "₹" + v.toLocaleString("en-IN")],
    ["build_cost", "One-off build cost", 100000, 3000000, 50000, (v) => "₹" + (v / 100000).toFixed(1) + " L"],
    ["run_cost_monthly", "Run cost per month", 0, 200000, 5000, (v) => "₹" + v.toLocaleString("en-IN")],
  ];
  $("sliders").innerHTML = `<p class="muted" style="margin-top:0;font-size:.88rem">Resolution mix from the batch: <b>${(mix.auto_resolved * 100).toFixed(0)}%</b> auto · <b>${(mix.agent_assist * 100).toFixed(0)}%</b> assist · <b>${(mix.escalated * 100).toFixed(0)}%</b> escalated</p>` +
    SL.map(([k, l, min, max, step]) => `<div class="slider"><label for="s-${k}">${l}<output id="o-${k}"></output></label><input type="range" id="s-${k}" min="${min}" max="${max}" step="${step}" value="${a[k]}"></div>`).join("");
  const inr = (v) => (Math.abs(v) >= 1e7 ? `₹${(v / 1e7).toFixed(2)} Cr` : Math.abs(v) >= 1e5 ? `₹${(v / 1e5).toFixed(1)} L` : "₹" + Math.round(v).toLocaleString("en-IN"));
  function updateRoi(user) {
    SL.forEach(([k, , , , , f]) => { a[k] = +$("s-" + k).value; $("o-" + k).textContent = f(a[k]); });
    const r = estimateRoi(mix, a);
    const g1 = r.roi >= 1000, g2 = r.payback !== null && r.payback < 1;
    $("roiGoals").innerHTML =
      `<div class="goal ${g1 ? "done" : ""}"><span class="gi">${g1 ? "✅" : "🎯"}</span><span>First-year ROI above <b>1,000%</b> · now ${r.roi.toFixed(0)}%</span></div>` +
      `<div class="goal ${g2 ? "done" : ""}"><span class="gi">${g2 ? "✅" : "🎯"}</span><span>Payback in under <b>1 month</b> · now ${r.payback ? r.payback.toFixed(1) + " mo" : "never"}</span></div>`;
    $("kpis").innerHTML = [
      [Math.round(r.hours).toLocaleString("en-IN"), "Hours saved per month", true],
      [`${a.baseline_minutes} → ${r.after.toFixed(1)}`, "Minutes per request"],
      [inr(r.net), "Net saving per month", true],
      [r.payback ? r.payback.toFixed(1) + " mo" : "n/a", "Payback period"],
      [r.roi.toFixed(0) + "%", "First-year ROI", true],
      [inr(r.net * 12), "Annual net saving"],
    ].map(([v, l, hot]) => `<div class="kpi ${hot ? "hot" : ""}"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
    if (user && g1) game.unlock("whale");
  }
  $("sliders").addEventListener("input", () => updateRoi(true));
  updateRoi(false);

  // scroll reveal
  const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && e.target.classList.add("in")), { threshold: 0.08 });
  document.querySelectorAll("main section").forEach((s) => { s.classList.add("reveal"); io.observe(s); });
}

main().catch((e) => {
  $("output").innerHTML = `<p class="badge bad">Failed to load demo data: ${esc(e.message)}</p>`;
});
