// server.js
import express from "express";
import rateLimit from "express-rate-limit";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

// Fix trust proxy to stop 'X-Forwarded-For' validation errors in Render/Heroku
app.set("trust proxy", 1);

// Rate limiter
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 60,
});
app.use(limiter);

// Keys from env
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

// ====== Helpers ======
async function callGemini(prompt) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_API_KEY}`;
  const payload = {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: {
      response_mime_type: "application/json",
      maxOutputTokens: 400,
    },
  };
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const j = await r.json();
  const textOut = j?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "{}";
  return JSON.parse(textOut);
}

async function callOpenAI(prompt) {
  const url = "https://api.openai.com/v1/chat/completions";
  const payload = {
    model: OPENAI_MODEL,
    messages: [
      { role: "system", content: "Return JSON only." },
      { role: "user", content: prompt },
    ],
    response_format: { type: "json_object" },
  };
  const r = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify(payload),
  });
  const j = await r.json();
  const textOut = j?.choices?.[0]?.message?.content?.trim() || "{}";
  return JSON.parse(textOut);
}

async function runModel(prompt) {
  if (GEMINI_API_KEY) return callGemini(prompt);
  if (OPENAI_API_KEY) return callOpenAI(prompt);
  throw new Error("No provider configured");
}

// ====== Routes ======

// Classify single
app.post("/classify", async (req, res) => {
  try {
    const { text, categories, subcategoriesByCat, hintsByCategory } = req.body;
    if (!text || !categories) return res.status(400).json({ error: "Missing input" });

    const prompt = `Classify this text strictly into categories: ${JSON.stringify(categories)}. Text: ${text}`;
    const out = await runModel(prompt);
    res.json(out);
  } catch (e) {
    console.error("classify error", e);
    res.status(502).json({ error: "Upstream error" });
  }
});

// Classify batch
app.post("/classify-batch", async (req, res) => {
  try {
    const { items, categories } = req.body;
    if (!Array.isArray(items) || !categories) return res.status(400).json({ error: "Missing input" });

    const results = await Promise.all(items.map(async (t) => {
      const prompt = `Classify this text strictly into categories: ${JSON.stringify(categories)}. Text: ${t}`;
      const out = await runModel(prompt);
      return { text: t, ...out };
    }));
    res.json({ items: results });
  } catch (e) {
    console.error("batch error", e);
    res.status(502).json({ error: "Upstream error" });
  }
});

// Analyze multi-line (per-line classification, no overall category)
app.post("/analyze", async (req, res) => {
  try {
    const { text, categories, subcategoriesByCat, hintsByCategory } = req.body;
    if (typeof text !== "string" || !Array.isArray(categories)) {
      return res.status(400).json({ error: "text and categories required" });
    }

    const INSTR = `Return ONLY JSON with schema:
{
  "items": [
    { "text": string, "category": string, "subcategory": string|null, "confidence": number,
      "suggestedNewCategory": string|null, "suggestedNewSubcategory": string|null }
  ]
}
Rules:
- Each line must be classified independently.
- "category" must be from ${JSON.stringify(categories)} (casing preserved). If no fit, set category="Other".
- If better category missing, suggestNewCategory with a single word (Food, Shopping, etc).
- Subcategory only if exists in SUBCATEGORIES_BY_CATEGORY and confidence >= 0.8.

TEXT:
${text}`;

    const out = await runModel(INSTR);
    const items = Array.isArray(out.items) ? out.items : [];
    res.json({ items });
  } catch (e) {
    console.error("analyze error", e);
    res.status(502).json({ error: "Upstream error", detail: String(e) });
  }
});

// Start
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Server running on", PORT));
