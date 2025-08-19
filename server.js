// server.js
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

const limiter = rateLimit({ windowMs: 60 * 1000, max: 120 });
app.use(limiter);

// ---------- ENV ----------
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-5-nano";
const DEBUG_LOG = process.env.DEBUG_LOG === "1";
const FALLBACK_CONFIDENCE = Number(process.env.FALLBACK_CONFIDENCE || "0.6");

// JWT_* present in your env, but intentionally **unused** (no client token required)

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
    if (String(p).toLowerCase() === m) return p; // use provided casing
  }
  return null;
}

function validateSub(category, sub, subsByCat) {
  if (!sub || !category) return null;
  const list = subsByCat?.[category] || [];
  return list.find((s) => s.toLowerCase() === String(sub).toLowerCase()) || null;
}

const CLASSIFIER_INSTRUCTION = `
You are a short-text classifier. Return ONLY valid JSON:
{"category": string, "subcategory": string|null, "confidence": number (0..1), "reason": string, "suggestedNewCategory": string|null}

DECISION RULES (apply in order):
1) If the text is a TV series, season, or episode title → category "Shows".
2) If the text is a film/movie title → category "Movies".
3) If the text is an ACTION without time/date (buy, watch, call, renew, subscribe, order) → "To-do".
4) If the text includes a time or a date (e.g., "4:30", "tomorrow", "Friday", "next week") → "Reminders".
5) If the text is a food/ingredient/grocery item → "Groceries".
6) If none of the above fit → "Other".

IMPORTANT:
- Prefer "Reminders" ONLY when there is explicit time/date; do NOT place media titles or ingredients in "Reminders".
- "Groceries" is ONLY for foods/ingredients or household consumables (cilantro, green chillies, eggs, detergent), not generic words like "friends".
- "subcategory" must be one from subcategoriesByCat[category] if appropriate (case-insensitive); otherwise null.
- Output category must match one of the provided categories exactly (case-insensitive match to pick, but output must use provided casing).
- If a clearly better category is missing (e.g., "Tennis" → "Sports"), set "suggestedNewCategory" to that single word; else null.

FEW-SHOTS:
- "How I Met Your Mother" → {"category":"Shows","subcategory":null}
- "Friends" → {"category":"Shows","subcategory":null}
- "Rocky" → {"category":"Movies","subcategory":null}
- "Rambo" → {"category":"Movies","subcategory":null}
- "green chillies" → {"category":"Groceries","subcategory":null}
- "cilantro 2" → {"category":"Groceries","subcategory":null}
- "pick up Rishi at 4:30" → {"category":"Reminders","subcategory":null}
- "watch Chinese drama" → {"category":"To-do","subcategory":null}
- "Chinese drama subscription" → {"category":"To-do","subcategory":null}
`;

// ---------- Gemini ----------
async function classifyWithGemini({ text, categories, subcategoriesByCat }) {
  if (!GEMINI_API_KEY) throw new Error("Missing GEMINI_API_KEY");

  const url =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=" +
    GEMINI_API_KEY;

  const cats = Array.isArray(categories) ? categories : [];
  const subs = subcategoriesByCat && typeof subcategoriesByCat === "object" ? subcategoriesByCat : {};

  const userText =
    CLASSIFIER_INSTRUCTION +
    "\n\nCATEGORIES:\n" +
    JSON.stringify(cats) +
    "\n\nSUBCATEGORIES_BY_CATEGORY:\n" +
    JSON.stringify(subs) +
    "\n\nTEXT:\n" +
    text;

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
    "\n\nCATEGORIES:\n" +
    JSON.stringify(cats) +
    "\n\nSUBCATEGORIES_BY_CATEGORY:\n" +
    JSON.stringify(subs) +
    "\n\nTEXT:\n" +
    text;

  const payload = {
    model: OPENAI_MODEL,
    messages: [
      { role: "system", content: "You are a JSON-only classifier. Output valid JSON object only." },
      { role: "user", content: userText }
    ],
    response_format: { type: "json_object" } // JSON mode
    // NOTE: do not set temperature; some small models only accept default
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
  throw new Error("No provider configured (set GEMINI_API_KEY or OPENAI_API_KEY)");
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

// ---------- Start ----------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("proxy up on :" + PORT));
