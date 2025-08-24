import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';

const app = express();
const PORT = process.env.PORT || 7845;
const APP_KEY = process.env.APP_KEY || '';

app.use(express.json({ limit: '1mb' }));
app.use(cors());
app.use(helmet());

const limiter = rateLimit({ windowMs: 60 * 1000, max: 120 });
app.use(limiter);

function requireAppKey(req, res, next) {
  if (!APP_KEY) return next();
  const key = req.get('X-App-Key');
  if (key && key === APP_KEY) return next();
  return res.status(401).json({ error: 'Unauthorized' });
}

// Providers (Gemini first, OpenAI fallback)
const geminiKey = process.env.GEMINI_API_KEY;
const genAI = geminiKey ? new GoogleGenerativeAI(geminiKey) : null;
const openaiKey = process.env.OPENAI_API_KEY;
const openai = openaiKey ? new OpenAI({ apiKey: openaiKey }) : null;
const openaiModel = process.env.OPENAI_MODEL || 'gpt-4o-mini';

// Util
const splitLines = (text) => String(text || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
const capHints = (h, max = 100) => {
  const out = {};
  for (const [k,v] of Object.entries(h || {})) if (Array.isArray(v)) out[k] = v.slice(0, max);
  return out;
};
const safeParse = (s) => { try { return JSON.parse(String(s||'').trim()); } catch { return null; } };

function buildClassifyPrompt({ text, categories, subcategoriesByCategory = {}, hintsByCategory = {}, languages = [] }) {
  return `/classify
You are a deterministic short-text classifier.   
Return ONLY a single JSON object with this schema: 
{  
  "category": string,                    // must be one of CATEGORIES (case-insensitive match, but output must use provided casing) 
  "subcategory": string|null,            // only choose from SUBCATEGORIES_BY_CATEGORY[category] if present AND highly confident; else null 
  "confidence": number,                  // 0.0..1.0 
  "reason": string,                      // one sentence, why you chose it 
  "suggestedNewCategory": string|null,   // single word if a clearly better category is missing from CATEGORIES; else null 
  "suggestedNewSubcategory": string|null, // single word if a clearly better subcategory is missing; else null 
  "alternativeCategory": string|null     // optional: if text plausibly fits TWO categories, give the second best choice 
} 

CONTEXT: 
- USER_PREFERRED_LANGUAGES: ${JSON.stringify(languages)} 
- CATEGORIES: ${JSON.stringify(categories)} 
- SUBCATEGORIES_BY_CATEGORY: ${JSON.stringify(subcategoriesByCategory)}          // include only if user has subcategories 
- HINTS_BY_CATEGORY: ${JSON.stringify(capHints(hintsByCategory))}                // optional user-learned phrases; can be empty 

INTERPRETATION RULES (language-aware, dynamic) 
1) Normalize spelling, transliteration, and synonyms across USER_PREFERRED_LANGUAGES before classifying. 
2) If text contains QUANTITY markers ("x2","2kg","3 packs"), treat as a PHYSICAL ITEM: 
   - Do NOT classify as "Movies" or "Shows" unless explicit media cues exist ("season","episode","trailer","movie","series","watch"). 
   - If it looks like an ingredient/consumable (cilantro, potato, onions) → category="Groceries". 
   - If it looks like a prepared dish or cuisine item → category="Other", suggestedNewCategory="Food". 
3) Reminders:
   - Choose "Reminders" if there is an explicit time/date expression (times, weekdays, relative dates), 
     OR if the text contains explicit reminder phrasing such as "remind me", "set a reminder", "reminder to". 
   - If there is reminder phrasing but no concrete time/date, still choose "Reminders" with moderate confidence.
4) Action verbs ("buy","get","renew","call","watch","pickup","order") bias towards "To-do"   
   (unless a concrete time/date is present → "Reminders"). 
5) Media: 
   - If the text is clearly a movie/film title → "Movies" 
   - If clearly episodic/series → "Shows" 
6) If both a task intent (verb) AND a media title are present (e.g. "watch Moana"): 
   - category = "To-do" 
   - alternativeCategory = "Movies" (or "Shows" if series) 
   - reason should note both interpretations. 
7) App/feature/bug/UX feedback or dev tasks (without explicit time) → "App" 
8) If no confident match within CATEGORIES, use "Other" and set suggestedNewCategory to the best single word (e.g. "Food","Shopping","Cosmetics","Sports","Finance"). 
9) Only output a subcategory if it exists under the chosen category AND confidence ≥ 0.8. 

CONSTRAINTS: 
- "category" MUST be from CATEGORIES (use provided casing). If no fit, choose "Other". 
- "alternativeCategory" MUST also be from CATEGORIES if present; else null. 
- "subcategory" MUST be valid under the chosen category if used; else null. 
- Output JSON only. No markdown or extra text. 

TEXT TO CLASSIFY: 
${text}`;
}

function buildAnalyzePrompt({ text, categories, subcategoriesByCategory = {}, hintsByCategory = {}, languages = [] }) {
  return `/analyze

You are a deterministic multi-line text classifier. 
Return ONLY a single JSON object with this schema:
{
  "items": [
    {
      "text": string,
      "category": string,                 // must be one of CATEGORIES (case-insensitive match, but output must use provided casing)
      "subcategory": string|null,         // only choose from SUBCATEGORIES_BY_CATEGORY[category] if present AND highly confident; else null
      "confidence": number,               // 0.0..1.0
      "reason": string,                   // one sentence, why you chose it
      "suggestedNewCategory": string|null,   // single word if a clearly better category is missing from CATEGORIES; else null
      "suggestedNewSubcategory": string|null // single word if a clearly better subcategory is missing; else null
    }
  ]
}

CONTEXT:
- USER_PREFERRED_LANGUAGES: ${JSON.stringify(languages)}
- CATEGORIES: ${JSON.stringify(categories)}
- SUBCATEGORIES_BY_CATEGORY: ${JSON.stringify(subcategoriesByCategory)}      // include only if user has subcategories
- HINTS_BY_CATEGORY: ${JSON.stringify(capHints(hintsByCategory))}            // optional user-learned phrases; can be empty

INTERPRETATION RULES (language-aware, dynamic)
1) Normalize spelling, transliteration, and synonyms across USER_PREFERRED_LANGUAGES before classifying.

2) Quantities:
   - If text contains QUANTITY markers ("x2","2","2kg","3 packs"), treat as a PHYSICAL ITEM.
   - Do NOT classify as "Movies" or "Shows" unless explicit media cues exist ("season","episode","trailer","movie","series","watch").
   - If it looks like an ingredient/consumable (produce, staples) → category="Groceries".
   - If it looks like a prepared dish or cuisine item → category="Other", suggestedNewCategory="Food".

3) Reminders:
   - Choose "Reminders" if there is an explicit time/date expression (times, weekdays, relative dates like "tomorrow", "next week", "at 6pm", "on Monday").
   - OR if the text contains explicit reminder phrasing such as "remind me", "set a reminder", "reminder to".
   - If there is reminder phrasing but no concrete time/date, still choose "Reminders" with moderate confidence (0.6–0.75).

4) Action verbs ("buy","get","renew","call","watch","pickup","order") bias towards "To-do"
   (unless a concrete time/date or reminder phrasing is present → then "Reminders").

5) Media:
   - Movie/film titles → "Movies"
   - Episodic/series titles → "Shows"
   - If phrased as an action (e.g., "watch Moana trailer") it may also be "To-do"; choose the stronger intent.

6) App/feature/bug/UX feedback or development tasks (without explicit time) → "App".

7) If no confident match within CATEGORIES, use "Other" and set a single-word suggestedNewCategory that best fits 
   (e.g., "Food","Shopping","Cosmetics","Sports","Finance"). 
   - Prefer "Food" for prepared dishes/cuisine.
   - Prefer "Shopping" for non-food tangible items if "Groceries" doesn’t apply.

CONSTRAINTS:
- Category MUST be from CATEGORIES (use provided casing). If no fit, choose "Other".
- Only output a subcategory if it already exists under the chosen category AND confidence ≥ 0.8; otherwise null.
- Output JSON only. No markdown or extra text.

TEXT TO CLASSIFY:

${text}`;
}

async function callLLM({ prompt }) {
  // Gemini primary
  if (genAI) {
    try {
      const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
      const r = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: prompt }]}],
        generationConfig: { responseMimeType: 'application/json' }
      });
      const t = r?.response?.text?.() ?? r?.response?.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
      const parsed = safeParse(t);
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
      const t = r?.choices?.[0]?.message?.content || '';
      const parsed = safeParse(t);
      if (parsed) return parsed;
    } catch {}
  }
  throw new Error('LLM providers failed');
}

function sanitizeSingle(obj, categories, subcats) {
  const valid = new Set(Array.isArray(categories) ? categories : []);
  const out = {
    category: 'Other', subcategory: null, confidence: 0, reason: '',
    suggestedNewCategory: null, suggestedNewSubcategory: null, alternativeCategory: null
  };
  if (!obj || typeof obj !== 'object') return out;
  const { category, subcategory, confidence, reason, suggestedNewCategory, suggestedNewSubcategory, alternativeCategory } = obj;
  if (typeof category === 'string' && valid.has(category)) out.category = category;
  if (typeof confidence === 'number') out.confidence = Math.max(0, Math.min(1, confidence));
  if (typeof reason === 'string') out.reason = reason.slice(0, 280);
  if (typeof suggestedNewCategory === 'string' && suggestedNewCategory.trim()) out.suggestedNewCategory = suggestedNewCategory.trim().split(/\s+/)[0];
  if (typeof suggestedNewSubcategory === 'string' && suggestedNewSubcategory.trim()) out.suggestedNewSubcategory = suggestedNewSubcategory.trim().split(/\s+/)[0];
  if (typeof alternativeCategory === 'string' && valid.has(alternativeCategory) && alternativeCategory !== out.category) out.alternativeCategory = alternativeCategory;

  const subs = subcats?.[out.category];
  if (Array.isArray(subs) && typeof subcategory === 'string' && subs.includes(subcategory) && out.confidence >= 0.8) out.subcategory = subcategory;
  return out;
}

app.get('/health', (_req, res) => res.json({ ok: true }));

app.post('/classify', requireAppKey, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCategory, hintsByCategory, languages } = req.body || {};
    const prompt = buildClassifyPrompt({ text, categories, subcategoriesByCategory, hintsByCategory, languages });
    const result = await callLLM({ prompt });
    const clean = sanitizeSingle(result, categories, subcategoriesByCategory);
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
    const prompt = buildAnalyzePrompt({ text: lines.join('\n'), categories, subcategoriesByCategory, hintsByCategory, languages });
    const result = await callLLM({ prompt });
    let items = Array.isArray(result?.items) ? result.items : [];
    items = lines.map((line, i) => {
      const r = items[i] || {};
      const clean = sanitizeSingle(r, categories, subcategoriesByCategory);
      return { text: line, ...clean, alternativeCategory: undefined };
    });
    res.json({ items });
  } catch (e) {
    res.status(500).json({ error: 'analyze_failed' });
  }
});

app.listen(PORT, () => console.log(`smartnotes-proxy (stage-1 prompts) on ${PORT}`));
