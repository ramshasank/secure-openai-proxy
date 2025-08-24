import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import LRUCache from 'lru-cache';
import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';

// Optional morgan (won't crash if not installed)
let morgan = null;
try {
  const mod = await import('morgan');
  morgan = mod.default || mod;
} catch (e) {
  // no-op: fallback logger below
}

const app = express();
const PORT = process.env.PORT || 7845;
const APP_KEY = process.env.APP_KEY || '';

app.use(express.json({ limit: '1mb' }));
app.use(cors());
app.use(helmet());
if (morgan) {
  app.use(morgan('tiny'));
} else {
  // tiny fallback logger
  app.use((req, _res, next) => {
    console.log(`${req.method} ${req.url}`);
    next();
  });
}

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 120,
});
app.use(limiter);

function requireAppKey(req, res, next) {
  if (!APP_KEY) return next(); // allow local dev if not set
  const key = req.get('X-App-Key');
  if (key && key === APP_KEY) return next();
  return res.status(401).json({ error: 'Unauthorized' });
}

// Cache for normalized inputs
const cache = new LRUCache({
  max: 2000,
  ttl: 5 * 60 * 1000,
});

// Providers
const geminiKey = process.env.GEMINI_API_KEY;
const genAI = geminiKey ? new GoogleGenerativeAI(geminiKey) : null;
const openaiKey = process.env.OPENAI_API_KEY;
const openai = openaiKey ? new OpenAI({ apiKey: openaiKey }) : null;
const openaiModel = process.env.OPENAI_MODEL || 'gpt-4o-mini';

function normalizeKey(s) {
  return String(s || '').trim().toLowerCase();
}

function capHints(hintsByCategory, maxPerCat = 100) {
  const out = {};
  for (const [cat, arr] of Object.entries(hintsByCategory || {})) {
    if (Array.isArray(arr)) out[cat] = arr.slice(0, maxPerCat);
  }
  return out;
}

// Prompts (Stage-1 spec)
function buildClassifyPrompt(payload) {
  const { text, categories, subcategoriesByCategory = {}, hintsByCategory = {}, languages = [] } = payload;
  return `/classify
You are a deterministic short-text classifier.
Return ONLY a single JSON object with this schema:
{
  "category": string,
  "subcategory": string|null,
  "confidence": number,
  "reason": string,
  "suggestedNewCategory": string|null,
  "suggestedNewSubcategory": string|null,
  "alternativeCategory": string|null
}

CONTEXT:
- USER_PREFERRED_LANGUAGES: ${JSON.stringify(languages)}
- CATEGORIES: ${JSON.stringify(categories)}
- SUBCATEGORIES_BY_CATEGORY: ${JSON.stringify(subcategoriesByCategory)}
- HINTS_BY_CATEGORY: ${JSON.stringify(capHints(hintsByCategory, 100))}

INTERPRETATION RULES
1) Normalize across languages and transliterations.
2) Reminders if time/date or 'remind me' phrasing (even w/o exact time → moderate confidence).
3) Quantities → physical items; ingredients → Groceries; prepared dishes → Other+suggestedNewCategory="Food".
4) Action verbs bias To-do unless explicit reminder time.
5) Media: movies vs shows; if 'watch <title>' → To-do with alternativeCategory.
6) App feedback/dev tasks → App.
7) If no confident fit → Other with single-word suggestedNewCategory.
8) Subcategory only if provided under chosen category AND confidence ≥ 0.8.

CONSTRAINTS:
- category/alternativeCategory must be from CATEGORIES (casing preserved).
- subcategory must be valid under chosen category if present; else null.
- Output JSON only.

TEXT TO CLASSIFY:
${text}`;
}

function buildAnalyzePrompt(payload) {
  const { text, categories, subcategoriesByCategory = {}, hintsByCategory = {}, languages = [] } = payload;
  return `/analyze
You are a deterministic multi-line text classifier.
Return ONLY a single JSON object with this schema:
{ "items": [ { "text": string, "category": string, "subcategory": string|null, "confidence": number, "reason": string, "suggestedNewCategory": string|null, "suggestedNewSubcategory": string|null } ] }

CONTEXT:
- USER_PREFERRED_LANGUAGES: ${JSON.stringify(languages)}
- CATEGORIES: ${JSON.stringify(categories)}
- SUBCATEGORIES_BY_CATEGORY: ${JSON.stringify(subcategoriesByCategory)}
- HINTS_BY_CATEGORY: ${JSON.stringify(capHints(hintsByCategory, 100))}

RULES:
- Preserve order; classify each non-empty line independently.
- Same interpretation rules as /classify.
- Constraints: category must be from CATEGORIES; subcategory must be valid if present.

TEXT TO ANALYZE:
${text}`;
}

function safeJsonParse(s) {
  try {
    if (typeof s !== 'string') return null;
    const t = s.trim();
    if (!t) return null;
    return JSON.parse(t);
  } catch {
    return null;
  }
}

async function callLLM({ prompt }) {
  // Gemini first
  if (genAI) {
    try {
      const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
      const r = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: prompt }]}],
        generationConfig: { responseMimeType: 'application/json' }
      });
      const text = r?.response?.text?.() ?? r?.response?.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
      const parsed = safeJsonParse(text);
      if (parsed) return parsed;
    } catch {}
  }
  // OpenAI fallback
  if (openai) {
    try {
      const r = await openai.chat.completions.create({
        model: openaiModel,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' },
        temperature: 0
      });
      const text = r?.choices?.[0]?.message?.content || '';
      const parsed = safeJsonParse(text);
      if (parsed) return parsed;
    } catch {}
  }
  throw new Error('All providers failed');
}

function sanitizeSingle(obj, categories, subcatsByCat) {
  const catSet = new Set(Array.isArray(categories) ? categories : []);
  const out = {
    category: 'Other',
    subcategory: null,
    confidence: 0,
    reason: '',
    suggestedNewCategory: null,
    suggestedNewSubcategory: null,
    alternativeCategory: null,
  };
  if (!obj || typeof obj !== 'object') return out;
  let { category, subcategory, confidence, reason, suggestedNewCategory, suggestedNewSubcategory, alternativeCategory } = obj;
  if (typeof category === 'string' && catSet.has(category)) out.category = category;
  if (typeof confidence === 'number') out.confidence = Math.max(0, Math.min(1, confidence));
  if (typeof reason === 'string') out.reason = reason.slice(0, 280);
  if (typeof suggestedNewCategory === 'string' && suggestedNewCategory.trim()) out.suggestedNewCategory = suggestedNewCategory.trim().split(/\s+/)[0];
  if (typeof suggestedNewSubcategory === 'string' && suggestedNewSubcategory.trim()) out.suggestedNewSubcategory = suggestedNewSubcategory.trim().split(/\s+/)[0];
  if (typeof alternativeCategory === 'string' && catSet.has(alternativeCategory) && alternativeCategory !== out.category) out.alternativeCategory = alternativeCategory;

  const subs = subcatsByCat?.[out.category];
  if (Array.isArray(subs) && typeof subcategory === 'string' && subs.includes(subcategory) && out.confidence >= 0.8) {
    out.subcategory = subcategory;
  } else {
    out.subcategory = null;
  }
  return out;
}

function splitLines(text) {
  return String(text || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
}

app.get('/health', (_req, res) => res.json({ ok: true }));

app.post('/classify', requireAppKey, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCategory, hintsByCategory, languages } = req.body || {};
    const raw = String(text || '');
    const key = 'classify:' + normalizeKey(raw) + '|' + JSON.stringify(categories || []);
    if (cache.has(key)) return res.json(cache.get(key));

    const prompt = buildClassifyPrompt({ text: raw, categories, subcategoriesByCategory, hintsByCategory, languages });
    const result = await callLLM({ prompt });
    const clean = sanitizeSingle(result, categories, subcategoriesByCategory);
    cache.set(key, clean);
    res.json(clean);
  } catch (e) {
    res.status(500).json({ error: 'classification_failed' });
  }
});

app.post('/analyze', requireAppKey, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCategory, hintsByCategory, languages } = req.body || {};
    const lines = splitLines(text);
    if (!lines.length) return res.json({ items: [] });

    const key = 'analyze:' + normalizeKey(lines.join('|')) + '|' + JSON.stringify(categories || []);
    if (cache.has(key)) return res.json(cache.get(key));

    const prompt = buildAnalyzePrompt({ text: lines.join('\\n'), categories, subcategoriesByCategory, hintsByCategory, languages });
    const result = await callLLM({ prompt });
    let items = Array.isArray(result?.items) ? result.items : [];
    items = lines.map((line, idx) => {
      const r = items[idx] || {};
      const clean = sanitizeSingle(r, categories, subcategoriesByCategory);
      clean.text = line;
      delete clean.alternativeCategory;
      return clean;
    });
    const out = { items };
    cache.set(key, out);
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: 'analyze_failed' });
  }
});

app.listen(PORT, () => {
  console.log(`smartnotes-proxy listening on ${PORT}`);
});
