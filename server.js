import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import LRUCache from 'lru-cache';
import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';

const app = express();
const PORT = process.env.PORT || 7845;
const APP_KEY = process.env.APP_KEY || '';

app.use(express.json({ limit: '1mb' }));
app.use(cors());
app.use(helmet());
app.use(morgan('tiny'));

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 120, // 120 req/min per IP
});
app.use(limiter);

function requireAppKey(req, res, next) {
  if (!APP_KEY) return next(); // allow in dev if key not set
  const key = req.get('X-App-Key');
  if (key && key === APP_KEY) return next();
  return res.status(401).json({ error: 'Unauthorized' });
}

// Simple normalized-text -> JSON LRU
const cache = new LRUCache({
  max: 2000,
  ttl: 5 * 60 * 1000, // 5 min
});

// Clients
const geminiKey = process.env.GEMINI_API_KEY;
const genAI = geminiKey ? new GoogleGenerativeAI(geminiKey) : null;
const openaiKey = process.env.OPENAI_API_KEY;
const openai = openaiKey ? new OpenAI({ apiKey: openaiKey }) : null;
const openaiModel = process.env.OPENAI_MODEL || 'gpt-4o-mini';

// Prompt templates (from user's finalized spec)
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

DECISION ORDER (strict):
1) Normalize spelling/transliteration/synonyms across USER_PREFERRED_LANGUAGES before classifying.
2) If TEXT is empty/whitespace → category="Other", confidence ≤ 0.5.
3) Reminders have priority:
   - If explicit time/date/weekday/relative date OR reminder phrasing ("remind me","set a reminder","reminder to") → category="Reminders".
   - If reminder phrasing without a concrete time, choose "Reminders" with moderate confidence (0.6–0.75).
4) Quantities → physical items:
   - If TEXT contains quantity markers ("x2","2kg","3 packs","2", etc.), treat as PHYSICAL ITEM.
   - Do NOT classify as "Movies"/"Shows" unless explicit media cues exist ("season","episode","trailer","movie","series","watch").
   - If ingredient/consumable → "Groceries".
   - If prepared dish/cuisine item → "Other", suggestedNewCategory="Food".
5) Media:
   - Clear film title → "Movies"
   - Clear episodic/series → "Shows"
6) Actions (no reminder phrasing/time):
   - Action verbs ("buy","get","renew","call","watch","pickup","order") bias "To-do".
   - If both a task verb and a media title (e.g., "watch Moana") → category="To-do", alternativeCategory="Movies" (or "Shows").
7) App/feature/bug/UX/dev tasks without explicit time → "App".
8) If no confident fit within CATEGORIES, choose "Other" and set suggestedNewCategory to the best single word (e.g., "Food","Shopping","Cosmetics","Sports","Finance").
9) Subcategories:
   - Only output a subcategory if it exists under the chosen category AND confidence ≥ 0.8. Otherwise null.

HINT USAGE:
- Treat HINTS_BY_CATEGORY as user-learned examples to improve accuracy; never echo hints in the reason.
- If TEXT exactly or nearly matches a hint for a category, increase confidence accordingly.

CONSTRAINTS:
- "category" MUST be one of CATEGORIES using provided casing; if no fit, use "Other".
- "alternativeCategory" (if present) MUST also be from CATEGORIES and different from "category".
- "subcategory" MUST be valid under the chosen category if used; else null.
- Output JSON only. No extra text, no markdown, no trailing commas.

TEXT TO CLASSIFY:
${text}`;
}

function buildAnalyzePrompt(payload) {
  const { text, categories, subcategoriesByCategory = {}, hintsByCategory = {}, languages = [] } = payload;

  return `/analyze
You are a deterministic multi-line text classifier.

Return ONLY a single JSON object with this schema:
{
  "items": [
    {
      "text": string,
      "category": string,
      "subcategory": string|null,
      "confidence": number,
      "reason": string,
      "suggestedNewCategory": string|null,
      "suggestedNewSubcategory": string|null
    }
  ]
}

CONTEXT:
- USER_PREFERRED_LANGUAGES: ${JSON.stringify(languages)}
- CATEGORIES: ${JSON.stringify(categories)}
- SUBCATEGORIES_BY_CATEGORY: ${JSON.stringify(subcategoriesByCategory)}
- HINTS_BY_CATEGORY: ${JSON.stringify(capHints(hintsByCategory, 100))}

RULES:
1) Preserve input order. Each non-empty line is classified independently. Do not merge or invent items. Ignore empty lines.
2) Normalize spelling/transliteration/synonyms across USER_PREFERRED_LANGUAGES before classifying.
3) Reminders have priority:
   - Explicit time/date/weekday/relative date OR reminder phrasing ("remind me", "set a reminder", "reminder to") → "Reminders".
   - Without concrete time but with reminder phrasing → "Reminders" with confidence 0.6–0.75.
4) Quantities → physical items:
   - If a line contains quantity markers ("x2","2","2kg","3 packs"), treat as PHYSICAL ITEM.
   - Do NOT classify as "Movies"/"Shows" unless explicit media cues exist ("season","episode","trailer","movie","series","watch").
   - Ingredient/consumable → "Groceries".
   - Prepared dish/cuisine item → "Other", suggestedNewCategory="Food".
5) Media:
   - Film titles → "Movies"
   - Episodic/series → "Shows"
   - If phrased as an action (e.g., "watch Moana trailer") it may also be "To-do"; choose the stronger intent.
6) Actions (no reminder phrasing/time):
   - Verbs ("buy","get","renew","call","watch","pickup","order") bias "To-do".
7) App/feature/bug/UX/dev tasks without explicit time → "App".
8) If no confident fit, choose "Other" and set suggestedNewCategory to the best single word (e.g., "Food","Shopping","Cosmetics","Sports","Finance").
9) Subcategories:
   - Only output a subcategory if it exists under the chosen category AND confidence ≥ 0.8. Otherwise null.

HINT USAGE:
- Treat HINTS_BY_CATEGORY as user-learned examples; never echo hints in reasons.
- If a line exactly or nearly matches a hint, increase confidence accordingly.

CONSTRAINTS:
- "category" MUST be from CATEGORIES using provided casing; if no fit, use "Other".
- Output JSON only. No extra text, no markdown, no trailing commas.

TEXT TO ANALYZE:
${text}`;
}

// helper to cap hints per category
function capHints(hintsByCategory, maxPerCat = 100) {
  const out = {};
  for (const [cat, arr] of Object.entries(hintsByCategory || {})) {
    if (Array.isArray(arr)) {
      out[cat] = arr.slice(0, maxPerCat);
    }
  }
  return out;
}

function normalizeKey(s) {
  return String(s || '').trim().toLowerCase();
}

// Strict JSON parse with minimal cleanup
function safeJsonParse(s) {
  try {
    if (typeof s !== 'string') return null;
    const trimmed = s.trim();
    if (!trimmed) return null;
    return JSON.parse(trimmed);
  } catch (e) {
    return null;
  }
}

// Try Gemini then OpenAI
async function callLLM({ prompt }) {
  // 1) Gemini
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
    } catch (e) {}
  }
  // 2) OpenAI fallback
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
    } catch (e) {}
  }
  throw new Error('All providers failed');
}

// Validate category/subcategory and clamp confidence
function sanitizeSingle(obj, categories, subcatsByCat) {
  const CATS = Array.isArray(categories) ? categories : [];
  const catSet = new Set(CATS);
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

  // Subcategory validity
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

// Routes
app.get('/health', (req, res) => res.json({ ok: true }));

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
    return res.json(clean);
  } catch (e) {
    return res.status(500).json({ error: 'classification_failed' });
  }
});

app.post('/analyze', requireAppKey, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCategory, hintsByCategory, languages } = req.body || {};
    const lines = splitLines(text);
    if (!lines.length) return res.json({ items: [] });

    const key = 'analyze:' + normalizeKey(lines.join('|')) + '|' + JSON.stringify(categories || []);
    if (cache.has(key)) return res.json(cache.get(key));

    const prompt = buildAnalyzePrompt({ text: lines.join('\n'), categories, subcategoriesByCategory, hintsByCategory, languages });
    const result = await callLLM({ prompt });
    let items = Array.isArray(result?.items) ? result.items : [];
    // sanitize each
    items = lines.map((line, idx) => {
      const r = items[idx] || {};
      const clean = sanitizeSingle(r, categories, subcategoriesByCategory);
      clean.text = line;
      // /analyze schema doesn't include alternativeCategory
      delete clean.alternativeCategory;
      return clean;
    });
    const out = { items };
    cache.set(key, out);
    return res.json(out);
  } catch (e) {
    return res.status(500).json({ error: 'analyze_failed' });
  }
});

app.listen(PORT, () => {
  console.log(`secure-openai-proxy-stage1 listening on ${PORT}`);
});
