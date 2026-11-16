// Browser port of genai_suite (offline extractive backend). Mirrors the Python modules 1:1 and
// uses the classifier weights, TF-IDF vocabulary and chunks exported by scripts/build_demo.py.

export function createPipeline(model) {
  const R = model.retrieval;
  const C = model.classifier;
  const STOP = new Set(R.stopwords);

  // ---- text.py ----
  const tokenize = (t) => t.toLowerCase().match(/\b[A-Za-z_][A-Za-z0-9_]+\b/g) || [];
  function stem(w) {
    if (w.endsWith("s") && !w.endsWith("ss") && w.length > 4) w = w.slice(0, -1);
    for (const suf of ["ing", "ed"]) {
      if (w.endsWith(suf) && w.length - suf.length >= 4) { w = w.slice(0, -suf.length); break; }
    }
    if (w.endsWith("e") && w.length > 4) w = w.slice(0, -1);
    return w;
  }
  const contentTokens = (t) => tokenize(t).filter((w) => !STOP.has(w)).map(stem);
  const splitSentences = (t) => t.replace(/\n/g, " ").split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  const stripCitations = (t) => t.replace(/\s*\[\d+\]/g, "");

  // ---- TF-IDF (sklearn: sublinear_tf=True, norm=l2) ----
  function vectorizer(vocab, idf, analyzer) {
    const index = new Map(vocab.map((v, i) => [v, i]));
    return (text) => {
      const counts = new Map();
      for (const t of analyzer(text)) { const i = index.get(t); if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1); }
      const vec = new Map();
      let norm = 0;
      for (const [i, c] of counts) { const v = (1 + Math.log(c)) * idf[i]; vec.set(i, v); norm += v * v; }
      norm = Math.sqrt(norm) || 1;
      for (const [i, v] of vec) vec.set(i, v / norm);
      return vec;
    };
  }
  const dot = (a, b) => { let s = 0; for (const [i, v] of a) if (b.has(i)) s += v * b.get(i); return s; };

  const embed = vectorizer(R.vocab, R.idf, contentTokens);
  const chunkVecs = R.chunks.map((c) => embed(c.index_text));
  const termWeights = new Map(R.vocab.map((v, i) => [v, R.idf[i]]));
  const classify = (() => {
    const feat = vectorizer(C.vocab, C.idf, contentTokens);
    return (text) => {
      const x = feat(text);
      const logits = C.weights.map((row, k) => { let s = C.bias[k]; for (const [i, v] of x) s += row[i] * v; return s; });
      const m = Math.max(...logits);
      const ex = logits.map((l) => Math.exp(l - m));
      const z = ex.reduce((a, b) => a + b, 0);
      const probs = Object.fromEntries(C.labels.map((l, k) => [l, ex[k] / z]));
      const label = C.labels[logits.indexOf(m)];
      return { label, confidence: probs[label], probs };
    };
  })();

  // ---- vector_store.py ----
  function expandQuery(q) {
    const extra = (q.toLowerCase().match(/\w+/g) || []).filter((w) => w in R.synonyms).map((w) => R.synonyms[w]);
    return extra.length ? q + " " + extra.join(" ") : q;
  }
  function search(query, k, boostSource) {
    const q = embed(query);
    return R.chunks
      .map((c, i) => {
        let s = dot(chunkVecs[i], q);
        if (boostSource && s > 0 && c.source === boostSource) s += 0.1;
        return { chunk: c, score: s };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .filter((h) => h.score > 0);
  }

  // ---- processors ----
  function luhn(d) {
    let sum = 0;
    [...d].reverse().forEach((ch, i) => { let n = +ch; if (i % 2) { n *= 2; if (n > 9) n -= 9; } sum += n; });
    return sum % 10 === 0;
  }
  const PII = [
    ["EMAIL", /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g],
    ["CARD", /\b(?:\d[ -]?){12,15}\d\b/g],
    ["AADHAAR", /\b\d{4}[ -]\d{4}[ -]\d{4}\b/g],
    ["PAN", /\b[A-Z]{5}\d{4}[A-Z]\b/g],
    ["PHONE", /(?:\+91[ -]?)?\b[6-9]\d{9}\b|\+\d{1,3}[ -]\d{3}[ -]\d{3}[ -]\d{4}/g],
  ];
  function redactPii(text) {
    const found = {};
    for (const [label, re] of PII) {
      text = text.replace(re, (m) => {
        if (label === "CARD") { const d = m.replace(/\D/g, ""); if (d.length < 13 || !luhn(d)) return m; }
        found[label] = (found[label] || 0) + 1;
        return `[${label}]`;
      });
    }
    return { text, found };
  }
  function extractEntities(text) {
    const ents = {
      amounts: text.match(/(?:INR|Rs\.?|₹|USD|\$)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?(?:rupees|lakh|crore)\b/gi) || [],
      dates: text.match(/\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b|\b(?:today|tomorrow|yesterday|next week|last week)\b/gi) || [],
      ids: (text.match(/\b(?:INV|PO|ORD|TKT|EMP)-\d{3,}\b/gi) || []).map((s) => s.toUpperCase()),
      durations: text.match(/\b\d+\s(?:days?|weeks?|months?)\b/gi) || [],
    };
    return Object.fromEntries(Object.entries(ents).filter(([, v]) => v.length));
  }
  const URGENT = ["urgent", "asap", "immediately", "breach", "chargeback", "lost", "locked", "escalate", "blocked"];
  const inferPriority = (t) => (URGENT.some((w) => t.toLowerCase().includes(w)) ? "high" : "normal");

  function summarize(text, n = 2) {
    // sklearn TfidfVectorizer() defaults: raw tf, smooth idf, l2 norm; centroid cosine ranking.
    const sents = splitSentences(text);
    if (sents.length <= n) return sents.join(" ");
    const toks = sents.map((s) => s.toLowerCase().match(/\b\w\w+\b/g) || []);
    const df = new Map();
    toks.forEach((ts) => new Set(ts).forEach((t) => df.set(t, (df.get(t) || 0) + 1)));
    const N = sents.length;
    const vecs = toks.map((ts) => {
      const v = new Map();
      ts.forEach((t) => v.set(t, (v.get(t) || 0) + 1));
      let norm = 0;
      for (const [t, c] of v) { const w = c * (Math.log((1 + N) / (1 + df.get(t))) + 1); v.set(t, w); norm += w * w; }
      norm = Math.sqrt(norm) || 1;
      for (const [t, w] of v) v.set(t, w / norm);
      return v;
    });
    const centroid = new Map();
    vecs.forEach((v) => v.forEach((w, t) => centroid.set(t, (centroid.get(t) || 0) + w / N)));
    const cn = Math.sqrt([...centroid.values()].reduce((a, b) => a + b * b, 0)) || 1;
    const scores = vecs.map((v) => dot(v, centroid) / cn);
    const keep = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, n).map((x) => x[1]).sort((a, b) => a - b);
    return keep.map((i) => sents[i]).join(" ");
  }

  // ---- llm.py: LocalExtractiveBackend ----
  function answer(question, contexts, maxSentences = 3) {
    const q = new Set(contentTokens(expandQuery(question)));
    let cands = [];
    contexts.forEach(({ chunk, score }, idx) => {
      const rank = idx + 1;
      splitSentences(chunk.text).forEach((sentence, pos) => {
        const shared = [...new Set(contentTokens(sentence))].filter((t) => q.has(t));
        if (shared.length) {
          const weight = shared.reduce((a, t) => a + (termWeights.get(t) ?? 1), 0);
          cands.push({ key: weight * (1 + 2 * score), rank, pos, sentence, count: shared.length });
        }
      });
    });
    if (cands.some((c) => c.count >= 2)) cands = cands.filter((c) => c.count >= 2);
    if (!cands.length) return "I could not find this in the knowledge base.";
    cands.sort((a, b) => b.key - a.key || a.rank - b.rank || a.pos - b.pos);
    return cands.slice(0, maxSentences).sort((a, b) => a.rank - b.rank || a.pos - b.pos)
      .map((c) => `${c.sentence} [${c.rank}]`).join(" ");
  }

  // ---- critic ----
  function grounding(draft, passages) {
    const evidence = new Set(contentTokens(passages.join(" ")));
    const sents = splitSentences(stripCitations(draft)).filter((s) => contentTokens(s).length);
    if (!sents.length) return 0;
    const ok = sents.filter((s) => { const t = contentTokens(s); return t.filter((x) => evidence.has(x)).length / t.length >= 0.6; });
    return ok.length / sents.length;
  }
  function coverage(query, draft) {
    const q = new Set(contentTokens(query));
    if (!q.size) return 0;
    const d = new Set(contentTokens(stripCitations(draft)));
    return [...q].filter((t) => d.has(t)).length / q.size;
  }

  // ---- orchestrator.py ----
  const r3 = (x) => Math.round(x * 1000) / 1000;
  function run(request) {
    const trace = [];
    const log = (agent, action, detail) => trace.push({ agent, action, detail });
    const s = { request, attempts: 0 };

    // intake
    const red = redactPii(request);
    s.redacted = red.text; s.pii = red.found;
    log("intake", "redact_pii", { redacted: Object.keys(red.found).length ? red.found : "none found" });
    const words = s.redacted.split(/\s+/).filter(Boolean).length;
    s.summary = words > 60 ? summarize(s.redacted, 2) : s.redacted;
    if (words > 60) log("intake", "summarize", { from_words: words, to_words: s.summary.split(/\s+/).length });
    const cls = classify(s.summary);
    s.category = cls.label; s.confidence = cls.confidence; s.probs = cls.probs;
    log("intake", "classify", { category: s.category, confidence: r3(s.confidence) });
    s.entities = extractEntities(s.redacted); s.priority = inferPriority(s.redacted);
    log("intake", "extract", { entities: s.entities, priority: s.priority });

    // router
    s.query = s.summary.replace(/\[[A-Z]+\]/g, " ").replace(/\s+/g, " ").trim();
    s.k = s.confidence < 0.5 ? 4 : 3;
    log("router", "plan", { plan: ["research", "analyst", "critic", "responder"], k: s.k,
      reason: s.k === 4 ? "low classifier confidence, widening retrieval" : "confident intent" });

    const prior = s.confidence >= 0.5 ? model.category_source[s.category] : null;
    for (;;) {
      const query = s.attempts === 0 ? s.query : expandQuery(s.query);
      const k = s.k + s.attempts;
      s.retrieved = search(query, k, prior);
      s.topScore = s.retrieved.length ? s.retrieved[0].score : 0;
      log("research", "retrieve", { query, k, prior, hits: s.retrieved.map((h) => `${h.chunk.title} > ${h.chunk.section} (${h.score.toFixed(2)})`) });

      s.draft = answer(s.query, s.retrieved);
      log("analyst", "draft_answer", { backend: "local-extractive", chars: s.draft.length });

      s.grounding = r3(grounding(s.draft, s.retrieved.map((h) => h.chunk.text)));
      s.coverage = r3(coverage(query, s.draft));
      const weak = s.grounding < 0.7 || s.topScore < 0.18 || s.coverage < 0.25;
      const retry = weak && s.attempts < 2;
      log("critic", "review", { grounding: s.grounding, coverage: s.coverage, top_score: r3(s.topScore),
        verdict: retry ? "retry with expanded query" : weak ? "accept, low confidence" : "accept" });
      if (!retry) break;
      s.attempts += 1;
    }

    const confident = s.grounding >= 0.7 && s.topScore >= 0.18 && s.coverage >= 0.25 && s.confidence >= 0.45;
    s.resolution = s.priority === "high" ? "escalated" : confident ? "auto_resolved" : "agent_assist";
    s.queue = model.category_queue[s.category] || "General";
    s.sources = [...new Set(s.retrieved.slice(0, 3).map((h) => `${h.chunk.title} - ${h.chunk.section}`))].sort();
    log("responder", "respond", { resolution: s.resolution, queue: s.queue });
    s.trace = trace;
    return s;
  }

  return { run };
}

// ---- roi.py ----
export function estimateRoi(mix, a) {
  const after = (mix.auto_resolved || 0) * 0.5 + (mix.agent_assist || 0) * a.assist_minutes + (mix.escalated || 0) * a.escalated_minutes;
  const hours = (a.monthly_volume * (a.baseline_minutes - after)) / 60;
  const gross = hours * a.hourly_cost;
  const net = gross - a.run_cost_monthly;
  return {
    after, hours, gross, net,
    payback: net > 0 ? a.build_cost / net : null,
    roi: ((net * 12 - a.build_cost) / a.build_cost) * 100,
  };
}
