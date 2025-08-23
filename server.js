// server/server.js
// SmartListCore — Pack A (Stage 1.5)
// Recreated to match existing structure (helmet, rate-limit, node-fetch, LRU cache)
// and to add /analyze with userLanguages, forceCategory, subcategoryOnly.
// Date: 2025-08-23

import express from "express";
import helmet from "helmet";
import cors from "cors";
import rateLimit from "express-rate-limit";
import fetch from "node-fetch";

// ---------- App setup ----------
const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "512kb" }));
app.set("trust proxy", 1);

const limiter = rateLimit({ windowMs: 60 * 1000, max: 120 });
app.use(limiter);

// ---------- ENV ----------
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_MODEL   = process.env.OPENAI_MODEL || "gpt-5-nano";
const DEBUG_LOG      = process.env.DEBUG_LOG === "1";
const FALLBACK_CONFIDENCE = Number(process.env.FALLBACK_CONFIDENCE || "0.6");

// ---------- Tiny LRU cache ----------
const LRU_MAX = 5000;
const CACHE_TTL_MS = 7 * 24 * 3600 * 1000;
const CACHE = new Map(); // key -> { ts, value }

function cacheKey({ text, categories, subcategoriesByCat }) {
  return JSON.stringify({
    t: String(text || "").trim().toLowerCase(),
    c: (categories || []).map((c) => c.toLowerCase()).sort(),
    s: Object.fromEntries(
      Object.entries(subcategoriesByCat || {}).map(([k, v]) => [
        k.toLowerCase(),
        (v || []).map((x) => x.toLowerCase()).sort(),
      ])
    ),
  });
}
function cacheGet(key) {
  const hit = CACHE.get(key);
  if (!hit) return null;
  if (Date.now() - hit.ts > CACHE_TTL_MS) {
    CACHE.delete(key);
    return null;
  }
  return hit.value;
}
function cacheSet(key, value) {
  if (CACHE.size > LRU_MAX) {
    const first = CACHE.keys().next().value;
    if (first) CACHE.delete(first);
  }
  CACHE.set(key, { ts: Date.now(), value });
}

// ---------- Helpers ----------
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

function pickProvidedCategory(modelOut, provided) {
  if (!modelOut || !provided?.length) return null;
  const m = String(modelOut).toLowerCase();
  for (const p of provided) {
    if (String(p).toLowerCase() === m) return p; // keep provided casing
  }
  return null;
}

function validateSub(category, sub, subsByCat) {
  if (!sub || !category) return null;
  const list = subsByCat?.[category] || [];
  return list.find((s) => s.toLowerCase() === String(sub).toLowerCase()) || null;
}

function normLangs(arr) {
  if (!Array.isArray(arr)) return [];
  const out = [];
  const seen = new Set();
  for (const a of arr) {
    const v = String(a || "").trim().toLowerCase();
    if (v && !seen.has(v)) { out.push(v); seen.add(v); }
    if (out.length >= 5) break;
  }
  return out;
}

// ---------- Classifier instruction (rules-only) ----------
const CLASSIFIER_INSTRUCTION = `
You classify a short user note into exactly ONE of the provided categories.

CATEGORIES (intended meaning)
- Reminders: explicit time/date present (“at 4:30”, “tomorrow”, weekdays, ISO times).
- To-do: task phrasing (buy, pick up, call, schedule, renew, subscribe) with NO explicit time.
- Groceries: food/ingredients/edibles or common household consumables (eggs, sugar, cilantro, detergent).
- Movies: film-related with media signals (movie/film/trailer/watch + title/year/actor/director).
- Shows: TV/series/anime/cartoon with signals (show/series/episode/season).
- Other: everything else.

AMBIGUITY POLICY (CRITICAL)
If the text is short and could be multiple things (e.g., a single proper noun with no context),
do NOT guess. Use "category":"Other" AND set "suggestedNewCategory" to a concise type (e.g., "Sports", "Cars", "Books", "Shopping") inferred from general knowledge.

SUBCATEGORIES
Return a subcategory ONLY if it exists in the provided mapping for the chosen category; else null.

CONFIDENCE
Return a value 0..1. Penalize short/ambiguous inputs; increase only when explicit signals exist.

OUTPUT
Return STRICT JSON (no prose, no markdown):
{
  "category": string,             // must be one of the provided categories
  "subcategory": string|null,     // must be from subcategoriesByCat[category] or null
  "confidence": number,           // 0..1
  "reason": string,
  "suggestedNewCategory": string|null
}
`;

// ---------- Gemini primary ----------
async function classifyWithGemini({ text, categories, subcategoriesByCat }) {
  if (!GEMINI_API_KEY) throw new Error("Missing GEMINI_API_KEY");

  const url =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=" +
    GEMINI_API_KEY;

  const cats = Array.isArray(categories) ? categories : [];
  const subs = subcategoriesByCat && typeof subcategoriesByCat === "object" ? subcategoriesByCat : {};

  // Ambiguity nudge for short proper nouns with no obvious signals
  const trimmed = String(text || "").trim();
  const isShort  = trimmed.split(/\s+/).length <= 2;
  const looksProper = /^[A-Z][a-z]+$/.test(trimmed);
  const hasSignals = /\b(movie|film|show|series|episode|season|watch|buy|order|call|email|tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}:\d{2}|am|pm|kg|g|oz|lb|lbs|dozen)\b/i.test(trimmed);

  const ambiguityHint = (isShort && looksProper && !hasSignals)
    ? "The input looks like a short proper noun without clear signals. Do not guess; use category 'Other' and propose a concise suggestedNewCategory if a generic type is obvious (e.g., Sports, Cars, Books)."
    : "";

  const userText =
    CLASSIFIER_INSTRUCTION +
    (ambiguityHint ? ("\n\nAMBIGUITY HINT:\n" + ambiguityHint) : "") +
    "\n\nPROVIDED CATEGORIES:\n" + JSON.stringify(cats) +
    "\n\nSUBCATEGORIES_BY_CATEGORY:\n" + JSON.stringify(subs) +
    "\n\nTEXT:\n" + text;

  const payload = {
    contents: [{ role: "user", parts: [{ text: userText }] }],
    generationConfig: { response_mime_type: "application/json" },
  };

  const ac = new AbortController();
  const to = setTimeout(() => ac.abort(), 8000); // 8s guard
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: ac.signal,
    });
    const j = await r.json();
    const textOut = j?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
    const out = JSON.parse(textOut || "{}");

    const picked = pickProvidedCategory(out.category, cats);
    return {
      provider: "gemini",
      category: picked ?? "Other",
      subcategory: validateSub(picked ?? "Other", out.subcategory, subs),
      confidence: clamp(out.confidence ?? FALLBACK_CONFIDENCE, 0, 1),
      reason: out.reason || "",
      suggestedNewCategory: out.suggestedNewCategory || null,
    };
  } finally {
    clearTimeout(to);
  }
}

// ---------- OpenAI fallback ----------
async function classifyWithOpenAI({ text, categories, subcategoriesByCat }) {
  if (!OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");

  const url = "https://api.openai.com/v1/chat/completions";
  const cats = Array.isArray(categories) ? categories : [];
  const subs = subcategoriesByCat && typeof subcategoriesByCat === "object" ? subcategoriesByCat : {};

  const userText =
    CLASSIFIER_INSTRUCTION +
    "\n\nPROVIDED CATEGORIES:\n" + JSON.stringify(cats) +
    "\n\nSUBCATEGORIES_BY_CATEGORY:\n" + JSON.stringify(subs) +
    "\n\nTEXT:\n" + text;

  const payload = {
    model: OPENAI_MODEL,
    messages: [
      { role: "system", content: "You are a JSON-only classifier. Output valid JSON object only." },
      { role: "user", content: userText }
    ],
    response_format: { type: "json_object" }
  };

  const ac = new AbortController();
  const to = setTimeout(() => ac.abort(), 8000);
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify(payload),
      signal: ac.signal,
    });
    const j = await r.json();
    const textOut = j?.choices?.[0]?.message?.content?.trim();
    const out = JSON.parse(textOut || "{}");

    const picked = pickProvidedCategory(out.category, cats);
    return {
      provider: "openai",
      category: picked ?? "Other",
      subcategory: validateSub(picked ?? "Other", out.subcategory, subs),
      confidence: clamp(out.confidence ?? FALLBACK_CONFIDENCE, 0, 1),
      reason: out.reason || "",
      suggestedNewCategory: out.suggestedNewCategory || null,
    };
  } finally {
    clearTimeout(to);
  }
}

// ---------- Router with fallback logic ----------
async function routeClassify({ text, categories, subcategoriesByCat }) {
  // Primary: Gemini
  if (GEMINI_API_KEY) {
    try {
      return await classifyWithGemini({ text, categories, subcategoriesByCat });
    } catch (e) {
      if (DEBUG_LOG) console.warn("[route] Gemini failed, falling back to OpenAI:", String(e));
    }
  }
  // Fallback: OpenAI
  if (OPENAI_API_KEY) {
    return await classifyWithOpenAI({ text, categories, subcategoriesByCat });
  }
  // Final fallback: minimal heuristic so the proxy still works during local dev
  const trimmed = String(text || "").trim();
  const hasTime = /\b(\d{1,2}:\d{2}\s*(am|pm)?)\b/i.test(trimmed) || /\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i.test(trimmed);
  const hasDo = /\b(buy|pick up|call|schedule|renew|subscribe|email|text|order)\b/i.test(trimmed);
  let cat = "Other";
  if (hasTime) cat = "Reminders";
  else if (hasDo) cat = "To-do";
  const picked = pickProvidedCategory(cat, categories) ?? "Other";
  return {
    provider: "heuristic",
    category: picked,
    subcategory: null,
    confidence: FALLBACK_CONFIDENCE,
    reason: "heuristic",
    suggestedNewCategory: null
  };
}

// ---------- Endpoints ----------
app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    primary: GEMINI_API_KEY ? "gemini" : (OPENAI_API_KEY ? "openai" : "none"),
    fallback: OPENAI_API_KEY ? "openai" : "none",
    cacheSize: CACHE.size
  });
});

app.post("/classify", async (req, res) => {
  const t0 = Date.now();
  try {
    const { text, categories, subcategoriesByCat } = req.body || {};
    if (!text || !Array.isArray(categories)) {
      return res.status(400).json({ error: "text and categories required" });
    }
    const key = cacheKey({ text, categories, subcategoriesByCat });
    const cached = cacheGet(key);
    if (cached) return res.json({ ...cached, cached: true, ms: 0 });

    if (DEBUG_LOG) console.log(`[classify] start text="${String(text).slice(0, 80)}"`);
    const out = await routeClassify({ text, categories, subcategoriesByCat });
    cacheSet(key, out);
    const ms = Date.now() - t0;
    if (DEBUG_LOG) console.log(`[classify] done ${ms}ms via ${out.provider} -> ${out.category}/${out.subcategory ?? "—"} conf=${out.confidence}`);
    res.json({ ...out, cached: false, ms });
  } catch (e) {
    console.error("[classify] error", e);
    res.status(502).json({ error: "Upstream error", detail: String(e) });
  }
});

app.post("/classify-batch", async (req, res) => {
  const t0 = Date.now();
  try {
    const { items, categories, subcategoriesByCat } = req.body || {};
    if (!Array.isArray(items) || !Array.isArray(categories)) {
      return res.status(400).json({ error: "items[] and categories[] required" });
    }
    if (DEBUG_LOG) console.log(`[batch] start n=${items.length}`);
    const results = await Promise.all(
      items.map(async (text) => {
        const key = cacheKey({ text, categories, subcategoriesByCat });
        const cached = cacheGet(key);
        if (cached) return { ...cached, cached: true, ms: 0 };
        const out = await routeClassify({ text, categories, subcategoriesByCat });
        cacheSet(key, out);
        return { ...out, cached: false, ms: 0 };
      })
    );
    const ms = Date.now() - t0;
    if (DEBUG_LOG) console.log(`[batch] done ${ms}ms`);
    res.json({ results, ms });
  } catch (e) {
    console.error("[batch] error", e);
    res.status(502).json({ error: "Upstream error", detail: String(e) });
  }
});

// ---- NEW: /analyze (Stage 1.5) ----
// Per-line only. No overall category. Enforces subcategoryOnly when set.
// Accepts EITHER top-level fields OR a nested { context } for compatibility with iOS client.
app.post("/analyze", async (req, res) => {
  try {
    // Allow both shapes: { lines, context:{...} } OR legacy { text, categories, ... }
    const body = req.body || {};
    const lines = Array.isArray(body.lines)
      ? body.lines
      : String(body.text || "").split(/\r?\n/).map(s => s.replace(/^\s*-\s*\[\s*\]\s*/,"").replace(/^\s*-\s*/,"").trim()).filter(Boolean);

    const ctx = body.context && typeof body.context === "object" ? body.context : body;

    let { currentCategory, categories, subcategoriesByCat, userLanguages, hintsByCategory, forceCategory, subcategoryOnly } = ctx;
    if (!Array.isArray(categories)) {
      return res.status(400).json({ ok: false, error: "categories[] required" });
    }
    subcategoriesByCat = subcategoriesByCat || {};
    hintsByCategory = hintsByCategory || {};
    userLanguages = normLangs(userLanguages || []);
    const lockCategory = subcategoryOnly ? (forceCategory || currentCategory || null) : null;

    // Build instruction for model providers (if configured)
    const INSTR = `Return ONLY JSON with this exact shape:
{
  "results": [
    { "raw": string, "category": (string|null), "subcategory": (string|null),
      "suggestedNewCategory": (string|null), "suggestedNewSubcategory": (string|null),
      "isReminder": boolean, "reminderDateTime": (string|null),
      "quantity": (number|null), "confidence": number }
  ]
}

Rules:
- Process each line independently; ignore empty lines and checkboxes ("- [ ] " prefix) or "- " prefix.
- ${subcategoryOnly ? "NEVER change the category; it MUST remain " + JSON.stringify(lockCategory) + "." : "Choose category from provided CATEGORIES if subcategoryOnly is false."}
- When subcategoryOnly=true, set suggestedNewCategory=null.
- Suggest an existing subcategory only if it appears in SUBCATEGORIES_BY_CATEGORY[category]; otherwise consider "suggestedNewSubcategory" from the text tokens.
- Quantities imply physical items; bias towards groceries/physical categories when not locked.
- Reminder phrasing allowed without explicit time; if date/time is present, set reminderDateTime and increase confidence.
- Use HINTS_BY_CATEGORY only to disambiguate; never force.
- Include a normalized "confidence" 0..1.
- Do not include any other fields.

CATEGORIES:
${JSON.stringify(categories)}

SUBCATEGORIES_BY_CATEGORY:
${JSON.stringify(subcategoriesByCat)}

HINTS_BY_CATEGORY:
${JSON.stringify(hintsByCategory)}

USER_LANGUAGES (for parsing cues; do not hardcode defaults):
${JSON.stringify(userLanguages)}

LOCKED_CATEGORY (optional):
${JSON.stringify(lockCategory)}

LINES:
${JSON.stringify(lines)}
`.trim();

    async function callProvider() {
      if (GEMINI_API_KEY) {
        const url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=" + GEMINI_API_KEY;
        const payload = {
          contents: [{ role: "user", parts: [{ text: INSTR }] }],
          generationConfig: { response_mime_type: "application/json", maxOutputTokens: 1024 }
        };
        const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const j = await r.json();
        const textOut = j?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "{}";
        return JSON.parse(textOut);
      }
      if (OPENAI_API_KEY) {
        const url = "https://api.openai.com/v1/chat/completions";
        const payload = {
          model: OPENAI_MODEL,
          messages: [
            { role: "system", content: "Return JSON only." },
            { role: "user", content: INSTR }
          ],
          response_format: { type: "json_object" }
        };
        const r = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json", "Authorization": `Bearer ${OPENAI_API_KEY}` },
          body: JSON.stringify(payload)
        });
        const j = await r.json();
        const textOut = j?.choices?.[0]?.message?.content?.trim() || "{}";
        return JSON.parse(textOut);
      }
      return null;
    }

    function heuristicOne(line) {
      const raw = String(line || "");
      const hasTime = /\b(\d{1,2}(:\d{2})?\s*(am|pm)?)\b/i.test(raw) || /\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\\d{4}-\\d{2}-\\d{2})\b/i.test(raw);
      const isReminder = /\b(remind|follow\s*up|pay|renew|call|email|text|schedule|book|register|submit)\b/i.test(raw) || hasTime;
      const qty = (() => {
        const m = raw.match(/(?:^|\\s)(\\d+(?:\\.\\d+)?)\\s*(?:x|×)\\s*[^\\s]/i)
          || raw.match(/(?:^|\\s)[^\\s]+\\s*(?:x|×)\\s*(\\d+(?:\\.\\d+)?)/i)
          || raw.match(/(?:^|\\s)(\\d+(?:\\.\\d+)?)\\s*(?:pcs?|lb|lbs|pounds?|kg|g|oz|pack|packs|bag|bags)\\b/i)
          || raw.match(/(?:^|\\s)(\\d+(?:\\.\\d+)?)(?=\\s+[a-zA-Z])/i);
        return m ? Number(m[1] || m[2] || 1) : null;
      })();
      let category = null;
      if (subcategoryOnly) {
        category = lockCategory;
      } else if (qty) {
        // try bias to groceries-like category name
        const gi = (categories || []).find((c) => /grocery|grocer|shopping|store|market|food|pantry|cart/i.test(String(c)));
        category = gi || null;
      }
      // simple subcategory nudge
      const subs = (subcategoryOnly ? (subcategoriesByCat[lockCategory] || []) : (subcategoriesByCat[category] || []));
      let sub = null;
      for (const s of subs) {
        const n = String(s).toLowerCase();
        if (n && raw.toLowerCase().includes(n)) { sub = s; break; }
      }
      const conf = clamp((subcategoryOnly ? 0.65 : 0.55) + (qty ? 0.1 : 0) + (isReminder ? 0.1 : 0), 0, 0.95);
      return {
        raw,
        category,
        subcategory: sub,
        suggestedNewCategory: subcategoryOnly ? null : null,
        suggestedNewSubcategory: sub ? null : null,
        isReminder: !!isReminder,
        reminderDateTime: null,
        quantity: qty,
        confidence: conf
      };
    }

    let providerOut = await callProvider();
    let results = [];
    if (providerOut && Array.isArray(providerOut.results)) {
      results = providerOut.results.map((r) => {
        // Enforce locks and normalize
        const cat = subcategoryOnly ? (lockCategory ?? null) : (r.category ?? null);
        return {
          raw: String(r.raw || ""),
          category: cat,
          subcategory: r.subcategory ?? null,
          suggestedNewCategory: subcategoryOnly ? null : (r.suggestedNewCategory ?? null),
          suggestedNewSubcategory: r.suggestedNewSubcategory ?? null,
          isReminder: !!r.isReminder,
          reminderDateTime: r.reminderDateTime ?? null,
          quantity: typeof r.quantity === "number" ? r.quantity : null,
          confidence: typeof r.confidence === "number" ? clamp(r.confidence, 0, 1) : FALLBACK_CONFIDENCE
        };
      });
    } else {
      // Heuristic path per line
      results = lines.map(heuristicOne);
    }

    // Final guard: validate subcategories against provided mapping
    results = results.map((r) => {
      if (!r) return r;
      const cat = r.category;
      if (cat && r.subcategory) {
        const valid = validateSub(cat, r.subcategory, subcategoriesByCat);
        r.subcategory = valid || null;
      }
      if (subcategoryOnly) {
        r.category = lockCategory ?? null;
        r.suggestedNewCategory = null;
      }
      return r;
    });

    res.json({ ok: true, results });
  } catch (e) {
    console.error("[/analyze] error", e);
    res.status(502).json({ ok: false, error: "Upstream error", detail: String(e?.message || e) });
  }
});

// ---------- Start ----------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("proxy up on :" + PORT));
