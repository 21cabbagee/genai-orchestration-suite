import { createPipeline, estimateRoi } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const LABELS = { expense: "Expense", it_access: "IT access", procurement: "Procurement", refund: "Refund", hr_leave: "HR / leave", data_privacy: "Data privacy" };
const RES = { auto_resolved: ["Auto-resolved", "ok"], agent_assist: ["Agent assist", "warn"], escalated: ["Escalated", "bad"] };

const EXAMPLES = [
  "How many quotes do we need to buy 20 monitors worth 4 lakh?",
  "URGENT: I lost my laptop at the airport this morning",
  "Customer raised a chargeback but is also asking for a refund on ORD-55120",
  "Can I work from home three days a week?",
  "Customer wants a copy of their data, call them on 9876543210 or email priya@example.com",
];

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

  // knowledge base list
  const docs = [...new Map(model.retrieval.chunks.map((c) => [c.title, c])).keys()];
  $("kb").innerHTML = docs.map((d) => `<li>${esc(d)}</li>`).join("");

  // example chips
  $("chips").innerHTML = EXAMPLES.map((e, i) => `<button data-i="${i}">${esc(e.length > 46 ? e.slice(0, 44) + "…" : e)}</button>`).join("");
  $("chips").addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    $("req").value = EXAMPLES[+b.dataset.i];
    run();
  });

  let timer = [];
  function run() {
    timer.forEach(clearTimeout);
    timer = [];
    const text = $("req").value.trim();
    if (!text) return;
    const s = pipeline.run(text);

    const [resLabel, resCls] = RES[s.resolution];
    $("badges").innerHTML = [
      `<span class="badge ${resCls}">${resLabel}</span>`,
      `<span class="badge">${LABELS[s.category]} · ${(s.confidence * 100).toFixed(0)}%</span>`,
      `<span class="badge ${s.priority === "high" ? "bad" : ""}">Priority: ${s.priority}</span>`,
      `<span class="badge">Grounding ${(s.grounding * 100).toFixed(0)}%</span>`,
      `<span class="badge">Coverage ${(s.coverage * 100).toFixed(0)}%</span>`,
      s.attempts ? `<span class="badge warn">${s.attempts} retr${s.attempts > 1 ? "ies" : "y"}</span>` : "",
      Object.keys(s.pii).length ? `<span class="badge warn">PII redacted: ${Object.entries(s.pii).map(([k, v]) => `${k}×${v}`).join(", ")}</span>` : "",
    ].join("");
    $("answer").innerHTML = esc(s.draft).replace(/\[(\d+)\]/g, "<sup>[$1]</sup>");
    const route = s.resolution === "escalated" ? `Routed to <b>${esc(s.queue)}</b> as high priority.`
      : s.resolution === "agent_assist" ? `Draft queued for review by <b>${esc(s.queue)}</b>.`
      : `Owning queue: ${esc(s.queue)}.`;
    $("sources").innerHTML = `Sources: ${s.sources.map(esc).join("; ")}<br>${route}`;
    $("result").classList.remove("hidden");

    $("trace").innerHTML = "";
    s.trace.forEach((step, i) => {
      timer.push(setTimeout(() => {
        const li = document.createElement("li");
        li.className = "show";
        li.innerHTML = `<span class="agent">${esc(step.agent)}</span><div><div class="action">${esc(step.action)}</div><div class="detail">${esc(fmtDetail(step.detail)).replace(/\n/g, "<br>")}</div></div>`;
        $("trace").appendChild(li);
      }, i * 110));
    });

    const max = Math.max(...s.retrieved.map((h) => h.score), 0.01);
    $("passages").innerHTML = s.retrieved.map((h, i) => `
      <div class="passage">
        <header><span>[${i + 1}] ${esc(h.chunk.title)} › ${esc(h.chunk.section)}</span><span class="muted">${h.score.toFixed(3)}</span></header>
        <div>${esc(h.chunk.text)}</div>
        <div class="bar"><i style="width:${(h.score / max) * 100}%"></i></div>
      </div>`).join("") || `<p class="muted">No passage matched.</p>`;
  }
  $("run").addEventListener("click", run);
  $("req").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); });
  run();

  // batch table
  $("batch").querySelector("tbody").innerHTML = runs.map((r) => {
    const [l, c] = RES[r.resolution];
    return `<tr><td>${esc(r.redacted)}</td><td>${LABELS[r.category]} <span class="muted">${(r.category_confidence * 100).toFixed(0)}%</span></td><td class="num">${r.attempts}</td><td class="num">${(r.grounding * 100).toFixed(0)}%</td><td><span class="badge ${c}">${l}</span></td></tr>`;
  }).join("");

  // ROI sliders
  const a = { ...roi.assumptions };
  const mix = roi.mix;
  const SL = [
    ["monthly_volume", "Requests per month", 500, 20000, 500, (v) => v.toLocaleString("en-IN")],
    ["baseline_minutes", "Manual minutes per request", 3, 30, 1, (v) => v],
    ["assist_minutes", "Minutes when editing a draft", 1, 15, 1, (v) => v],
    ["hourly_cost", "Loaded cost per hour (INR)", 300, 3000, 50, (v) => "₹" + v.toLocaleString("en-IN")],
    ["build_cost", "One-off build cost (INR)", 100000, 3000000, 50000, (v) => "₹" + (v / 100000).toFixed(1) + " L"],
    ["run_cost_monthly", "Run cost per month (INR)", 0, 200000, 5000, (v) => "₹" + v.toLocaleString("en-IN")],
  ];
  $("sliders").innerHTML = `<p class="muted" style="margin-top:0">Resolution mix from batch: ${(mix.auto_resolved * 100).toFixed(0)}% auto · ${(mix.agent_assist * 100).toFixed(0)}% assist · ${(mix.escalated * 100).toFixed(0)}% escalated</p>` +
    SL.map(([k, l, min, max, step]) => `<div class="slider"><label for="s-${k}">${l}<output id="o-${k}"></output></label><input type="range" id="s-${k}" min="${min}" max="${max}" step="${step}" value="${a[k]}"></div>`).join("");
  const inr = (v) => "₹" + Math.round(v).toLocaleString("en-IN");
  function updateRoi() {
    SL.forEach(([k, , , , , f]) => { a[k] = +$("s-" + k).value; $("o-" + k).textContent = f(a[k]); });
    const r = estimateRoi(mix, a);
    $("kpis").innerHTML = [
      [Math.round(r.hours).toLocaleString("en-IN"), "Hours saved per month"],
      [`${a.baseline_minutes} → ${r.after.toFixed(1)}`, "Minutes per request"],
      [inr(r.net), "Net saving per month"],
      [r.payback ? r.payback.toFixed(1) + " mo" : "n/a", "Payback period"],
      [r.roi.toFixed(0) + "%", "First-year ROI"],
      [inr(r.net * 12), "Annual net saving"],
    ].map(([v, l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  }
  $("sliders").addEventListener("input", updateRoi);
  updateRoi();
}

main().catch((e) => {
  document.querySelector("#demo").insertAdjacentHTML("beforeend", `<p class="badge bad">Failed to load demo data: ${esc(e.message)}</p>`);
});
