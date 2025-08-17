// server.js
import "dotenv/config.js";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import fetch from "node-fetch";
import jwt from "jsonwebtoken";
import { z } from "zod";

// ===== Env =====
const {
  PORT = 3000,
  OPENAI_API_KEY = "",
  GEMINI_API_KEY = "",
  OPENAI_MODEL = "gpt-5-nano",
  FALLBACK_CONFIDENCE = "0.6",
  JWT_SECRET = "",
  JWT_TTL_SECONDS = "3600"
} = process.env;

const FALLBACK_C = Math.max(0, Math.min(1, Number(FALLBACK_CONFIDENCE)));
const JWT_TTL = Math.max(60, Number(JWT_TTL_SECONDS) || 3600);

// ===== App =====
const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// Basic rate limit (tune as you like)
app.use(
  "/classify",
  rateLimit({
    windowMs: 60_000,
    max: 60, // 60 req/min per IP
    standardHeaders: true,
    legacyHeaders: false
  })
);

// ===== Schemas =====
const classifyBodySchema = z.object({
  text: z.string().min(1),
  categories: z.array(z.string()).min(1),
  subcategoriesByCat: z.record(z.array(z.string())).default({}),
  provider: z.enum(["gemini", "openai"]).optional().default("gemini"),
  allowFallback: z.boolean().optional().default(true)
});

// ===== Helpers =====
function clampResult(res, categories, subByCat) {
  const catsLower = categories.map((c) => c.toLowerCase());
  let cat = String(res.category || "").toLowerCase().trim();
  if (!catsLower.includes(cat)) cat = catsLower.includes("todo") ? "todo" : catsLower[0];

  const category = categories.find((c) => c.toLowerCase() === cat) || categories[0] || "todo";

  let sub = res.subcategory ?? null;
  if (sub) {
    const allowedSubs = (subByCat[category] || []).map((s) => s.toLowerCase().trim());
    if (!allowedSubs.includes(String(sub).toLowerCase().trim())) sub = null;
  }

  const confidence =
    typeof res.confidence === "number" && res.confidence >= 0 && res.confidence <= 1
      ? res.confidence
      : 0.7;

  return { category, subcategory: sub, confidence, rationale: res.rationale || "" };
}

function buildClassificationPrompt(text, categories, subByCat) {
  return `
You are a strict JSON classifier. Decide {category, subcategory, confidence}.

Allowed categories: [${categories.join(", ")}]
Allowed subcategories by category: ${JSON.stringify(subByCat)}

Rules:
- Output ONLY JSON: {"category": "...", "subcategory": null|"...", "confidence": 0..1, "rationale": "..."}
- Category MUST be from the allowed list (case-insensitive).
- Subcategory MUST be from the allowed list for the chosen category; if unclear, use null.
- Prefer "movies" for film titles (Moana, Thor, Harry Potter, Deadpool & Wolverine).
- Prefer "shows" for TV series (Bridgerton).
- Prefer "groceries" for foods/ingredients (carrot, bread).
- Use "todo" only for explicit actions/tasks.

Input: "${text}"
`.trim();
}

// ===== Providers =====
async function callGemini({ text, categories, subByCat }) {
  if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY missing");
  const url =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent";
  const prompt = buildClassificationPrompt(text, categories, subByCat);

  const r = await fetch(url, {
    method: "POST",
    headers: { "x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.2 }
    })
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`Gemini upstream ${r.status}: ${detail}`);
  }
  const data = await r.json();
  const textOut = data?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
  const s = textOut.indexOf("{"),
    e = textOut.lastIndexOf("}");
  const jsonText = s !== -1 && e !== -1 ? textOut.slice(s, e + 1) : textOut;
  return JSON.parse(jsonText);
}

async function callOpenAI({ text, categories, subByCat }) {
  if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY missing");
  const url = "https://api.openai.com/v1/chat/completions";
  const promptSystem = buildClassificationPrompt("{INPUT}", categories, subByCat)
    .replace('Input: "{INPUT}"', "")
    .trim();

  const body = {
    model: OPENAI_MODEL,
    messages: [
      { role: "system", content: promptSystem },
      { role: "user", content: `Input: "${text}"` }
    ]
  };
  // Omit temperature for some nano variants to avoid "unsupported_value"
  if (!OPENAI_MODEL.toLowerCase().includes("gpt-5-nano")) body.temperature = 0.2;

  const r = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`OpenAI upstream ${r.status}: ${detail}`);
  }
  const data = await r.json();
  const textOut = data?.choices?.[0]?.message?.content || "{}";
  const s = textOut.indexOf("{"),
    e = textOut.lastIndexOf("}");
  const jsonText = s !== -1 && e !== -1 ? textOut.slice(s, e + 1) : textOut;
  return JSON.parse(jsonText);
}

// ===== JWT auth =====
function requireJWT(req, res, next) {
  const auth = req.headers.authorization || "";
  const [, token] = auth.split(" ");
  if (!token) return res.status(401).json({ error: "Missing token" });
  if (!JWT_SECRET) return res.status(500).json({ error: "JWT secret not configured" });

  try {
    const payload = jwt.verify(token, JWT_SECRET);
    req.user = payload;
    next();
  } catch {
    return res.status(401).json({ error: "Invalid token" });
  }
}

// ===== Routes =====
app.post("/session", (req, res) => {
  const { deviceId } = req.body || {};
  if (!deviceId) return res.status(400).json({ error: "deviceId required" });
  if (!JWT_SECRET) return res.status(500).json({ error: "JWT secret not configured" });

  const now = Math.floor(Date.now() / 1000);
  const exp = now + JWT_TTL;
  const token = jwt.sign({ sub: deviceId, iat: now, exp }, JWT_SECRET, { algorithm: "HS256" });
  res.json({ token, expiresInSec: JWT_TTL });
});

app.post("/classify", requireJWT, async (req, res) => {
  // Validate input
  const parse = classifyBodySchema.safeParse(req.body);
  if (!parse.success) {
    return res.status(400).json({ error: "Invalid body", detail: parse.error.flatten() });
  }
  const { text, categories, subcategoriesByCat, provider, allowFallback } = parse.data;

  const job = { text, categories, subByCat: subcategoriesByCat };

  try {
    let primary = provider === "openai" ? callOpenAI : callGemini;
    let secondary = provider === "openai" ? callGemini : callOpenAI;

    let raw = await primary(job);
    let result = clampResult(raw, categories, subcategoriesByCat);

    if (allowFallback && result.confidence < FALLBACK_C) {
      try {
        const altRaw = await secondary(job);
        const alt = clampResult(altRaw, categories, subcategoriesByCat);
        if (alt.confidence > result.confidence) result = alt;
      } catch { /* ignore */ }
    }

    res.json(result);
  } catch (err) {
    if (allowFallback) {
      try {
        const altRaw = await (provider === "openai" ? callGemini(job) : callOpenAI(job));
        const alt = clampResult(altRaw, categories, subcategoriesByCat);
        return res.json(alt);
      } catch (err2) {
        return res.status(502).json({ error: "All providers failed", detail: String(err2) });
      }
    }
    return res.status(502).json({ error: "Provider failed", detail: String(err) });
  }
});

app.get("/health", (_req, res) => res.json({ ok: true }));

app.listen(Number(PORT), () => console.log("Proxy listening on", PORT));
