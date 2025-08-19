import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import jwt from "jsonwebtoken";
import dotenv from "dotenv";
import fetch from "node-fetch";

dotenv.config();

const {
  PORT = 3000,
  NODE_ENV = "development",
  JWT_SECRET,
  GEMINI_API_KEY,
  OPENAI_API_KEY,
  ALLOW_ORIGINS = "*"
} = process.env;

if (!JWT_SECRET) {
  console.warn("[WARN] JWT_SECRET is not set. Set it in Render env vars.");
}

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(helmet());
app.use(cors({ origin: (ALLOW_ORIGINS === "*" ? true : ALLOW_ORIGINS.split(",")) }));

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 120,
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

// ---- auth helpers ----
function signToken(deviceId) {
  const payload = { did: deviceId };
  return jwt.sign(payload, JWT_SECRET || "dev-secret", { expiresIn: "30m" });
}

function auth(req, res, next) {
  const authz = req.headers.authorization || "";
  const token = authz.startsWith("Bearer ") ? authz.slice(7) : null;
  if (!token) return res.status(401).json({ error: "Missing token" });
  try {
    const dec = jwt.verify(token, JWT_SECRET || "dev-secret");
    req.user = dec;
    next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid token" });
  }
}

// ---- routes ----
app.get("/health", (_req, res) => {
  res.json({ ok: true, provider: (GEMINI_API_KEY ? "gemini" : (OPENAI_API_KEY ? "openai" : "none")) });
});

app.post("/session", (req, res) => {
  const { deviceId } = req.body || {};
  if (!deviceId) return res.status(400).json({ error: "deviceId required" });
  const token = signToken(deviceId);
  res.json({ token, expiresInSec: 30 * 60 });
});

// Core classify (single)
app.post("/classify", auth, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCat, provider = "gemini" } = req.body || {};
    if (!text || !Array.isArray(categories)) return res.status(400).json({ error: "text and categories required" });

    const result = await routeClassify({ text, categories, subcategoriesByCat, provider });
    res.json(result);
  } catch (e) {
    console.error("[/classify] error", e);
    res.status(502).json({ error: "Upstream error", detail: String(e) });
  }
});

// Batch classify
app.post("/classify-batch", auth, async (req, res) => {
  try {
    const { items, categories, subcategoriesByCat, provider = "gemini" } = req.body || {};
    if (!Array.isArray(items) || !Array.isArray(categories)) {
      return res.status(400).json({ error: "items[] and categories[] required" });
    }
    const results = await Promise.all(items.map(text => routeClassify({ text, categories, subcategoriesByCat, provider })));
    res.json({ results });
  } catch (e) {
    console.error("[/classify-batch] error", e);
    res.status(502).json({ error: "Upstream error", detail: String(e) });
  }
});

// ---- router ----
async function routeClassify({ text, categories, subcategoriesByCat, provider }) {
  if (provider === "gemini") {
    if (!GEMINI_API_KEY) throw new Error("Missing GEMINI_API_KEY");
    return await classifyWithGemini(text, categories, subcategoriesByCat);
  } else {
    if (!OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");
    return await classifyWithOpenAI(text, categories, subcategoriesByCat);
  }
}

// ---- Gemini ----
async function classifyWithGemini(text, categories, subcats) {
  const model = "gemini-2.5-flash-lite";
  const sys = `You are a categorizer. Only respond with strict JSON.
Return:
{
  "category": "<one of categories array, or a new suggestion>",
  "subcategory": "<optional string or null>",
  "confidence": <0..1 number>,
  "teach": {
    "positiveTokens": ["optional", "keywords"],
    "negativeTokens": ["optional", "keywords"],
    "patterns": ["optional regex or phrase hints"]
  }
}`;
  const prompt = {
    role: "user",
    parts: [{
      text:
`SYSTEM:
${sys}

CATEGORIES: ${JSON.stringify(categories)}
SUBCATS: ${JSON.stringify(subcats || {})}

TEXT: ${text}

Rules:
- If none fits well, propose a new category name (short, singular), but still pick the best among current categories as the 'category' value if you must. Use confidence to reflect uncertainty.
- If a subcategory is obvious (e.g., store name for groceries), include it.
- Keep JSON valid. No extra commentary.`
    }]
  };

  const body = {
    contents: [prompt],
    generationConfig: { temperature: 0.4, responseMimeType: "application/json" }
  };

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!r.ok) {
    const textErr = await r.text();
    throw new Error(`Gemini ${r.status}: ${textErr}`);
  }
  const data = await r.json();
  const raw = data?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
  // Best-effort parse
  let parsed;
  try { parsed = JSON.parse(raw); } catch {
    parsed = { category: "Other", subcategory: null, confidence: 0.5 };
  }
  // Normalize
  return {
    category: parsed.category || "Other",
    subcategory: (parsed.subcategory === "" ? null : parsed.subcategory) ?? null,
    confidence: typeof parsed.confidence === "number" ? parsed.confidence : 0.6,
    teach: parsed.teach || { positiveTokens: [], negativeTokens: [], patterns: [] }
  };
}

// ---- OpenAI fallback (optional) ----
async function classifyWithOpenAI(text, categories, subcats) {
  const url = "https://api.openai.com/v1/chat/completions";
  const system = `You are a categorizer. Reply in JSON: {"category":"","subcategory":null,"confidence":0.0,"teach":{"positiveTokens":[],"negativeTokens":[],"patterns":[]}}`;
  const messages = [
    { role: "system", content: system },
    { role: "user", content: `Categories: ${JSON.stringify(categories)}; Subcats: ${JSON.stringify(subcats||{})}; Text: ${text}` }
  ];
  const body = {
    model: "gpt-5-mini",
    temperature: 0.4,
    messages
  };
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify(body)
  });
  if (!r.ok) {
    const err = await r.text();
    throw new Error(`OpenAI ${r.status}: ${err}`);
  }
  const data = await r.json();
  const textOut = data?.choices?.[0]?.message?.content || "{}";
  let parsed;
  try { parsed = JSON.parse(textOut); } catch {
    parsed = { category: "Other", subcategory: null, confidence: 0.5, teach: {positiveTokens:[],negativeTokens:[],patterns:[]} };
  }
  return {
    category: parsed.category || "Other",
    subcategory: (parsed.subcategory === "" ? null : parsed.subcategory) ?? null,
    confidence: typeof parsed.confidence === "number" ? parsed.confidence : 0.6,
    teach: parsed.teach || { positiveTokens: [], negativeTokens: [], patterns: [] }
  };
}

app.listen(PORT, () => {
  console.log(`[proxy] listening on ${PORT} (${NODE_ENV})`);
});
