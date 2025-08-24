import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';

/** Simple TTL cache without external deps */
class TTLCache {
  constructor(ttlMs = 300000, max = 2000) { // 5 min, cap size
    this.ttl = ttlMs;
    this.max = max;
    this.map = new Map();
  }
  _now() { return Date.now(); }
  get(k) {
    const e = this.map.get(k);
    if (!e) return undefined;
    if (this._now() - e.t > this.ttl) { this.map.delete(k); return undefined; }
    return e.v;
  }
  set(k, v) {
    if (this.map.size >= this.max) {
      // drop oldest
      const first = this.map.keys().next().value;
      if (first !== undefined) this.map.delete(first);
    }
    this.map.set(k, { v, t: this._now() });
  }
  has(k) { return this.get(k) !== undefined; }
}

const app = express();
const PORT = process.env.PORT || 7845;
const APP_KEY = process.env.APP_KEY || '';

app.use(express.json({ limit: '1mb' }));
app.use(cors());
app.use(helmet());

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 120,
});
app.use(limiter);

function requireAppKey(req, res, next) {
  if (!APP_KEY) return next(); // allow local dev if key not set
  const key = req.get('X-App-Key');
  if (key && key === APP_KEY) return next();
  return res.status(401).json({ error: 'Unauthorized' });
}

const cache = new TTLCache(5 * 60 * 1000, 2000);

// Providers
const geminiKey = process.env.GEMINI_API_KEY;
const genAI = geminiKey ? new GoogleGenerativeAI(geminiKey) : null;
const openaiKey = process.env.OPENAI_API_KEY;
const openai = openaiKey ? new OpenAI({ apiKey: openaiKey }) : null;
const openaiModel = process.env.OPENAI_MODEL || 'gpt-4o-mini';

const normalizeKey = s => String(s || '').trim().toLowerCase();

function capHints(hintsByCategory, maxPerCat = 100) {
  const out = {};
  for (const [cat, arr] of Object.entries(hintsByCategory || {})) {
    if (Array.isArray(arr)) out[cat] = arr.slice(0, maxPerCat);
  }
  return out;
}

function buildClassifyPrompt(payload) {
  const { text, categories, subcategoriesByCategory = {}, hintsByCategory = {}, languages = [] } = payload;
  return `/classify
You are a deterministic short-text classifier.
Return ONLY a single JSON object with this schema:
{"category":string,"subcategory":string|null,"confidence":number,"reason":string,"suggestedNewCategory":string|null,"suggestedNewSubcategory":string|null,"alternativeCategory":string|null}
CONTEXT:
- USER_PREFERRED_LANGUAGES: ${JSON.stringify(languages)}
- CATEGORIES: ${JSON.stringify(categories)}
- SUBCATEGORIES_BY_CATEGORY: ${JSON.stringify(subcategoriesByCategory)}
- HINTS_BY_CATEGORY: ${JSON.stringify(capHints(hintsByCategory, 100))}
INTERPRETATION RULES: (same as spec: reminders > todo verbs; quantities→groceries; prepared dish→Other+Food; media rules; App, etc.)
CONSTRAINTS: category/alternativeCategory ∈ CATEGORIES; subcategory valid; JSON only.
TEXT TO CLASSIFY:
${text}`;
}

function buildAnalyzePrompt(payload) {
  const { text, categories, subcategoriesByCategory = {}, hintsByCategory = {}, languages = [] } = payload;
  return `/analyze
You are a deterministic multi-line text classifier.
Return ONLY a single JSON object with this schema:{"items":[{"text":string,"category":string,"subcategory":string|null,"confidence":number,"reason":string,"suggestedNewCategory":string|null,"suggestedNewSubcategory":string|null}]}
CONTEXT/LANGUAGES/CATEGORIES/SUBCATEGORIES/HINTS same as /classify. Apply same rules.
TEXT TO ANALYZE:
${text}`;
}

function safeJsonParse(s) {
  try {
    if (typeof s !== 'string') return null;
    const t = s.trim();
    if (!t) return null;
    return JSON.parse(t);
  } catch { return null; }
}

async function callLLM({ prompt }) {
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
    category: 'Other', subcategory: null, confidence: 0, reason: '',
    suggestedNewCategory: null, suggestedNewSubcategory: null, alternativeCategory: null
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
  if (Array.isArray(subs) && typeof subcategory === 'string' && subs.includes(subcategory) && out.confidence >= 0.8) out.subcategory = subcategory;
  return out;
}

const splitLines = text => String(text || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);

app.get('/health', (_req, res) => res.json({ ok: true }));

app.post('/classify', requireAppKey, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCategory, hintsByCategory, languages } = req.body || {};
    const raw = String(text || '');
    const key = 'c:' + normalizeKey(raw) + '|' + JSON.stringify(categories || []);
    const hit = cache.get(key);
    if (hit) return res.json(hit);
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
    const key = 'a:' + normalizeKey(lines.join('|')) + '|' + JSON.stringify(categories || []);
    const hit = cache.get(key);
    if (hit) return res.json(hit);
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
  console.log(`smartnotes-proxy (minimal) on ${PORT}`);
});
