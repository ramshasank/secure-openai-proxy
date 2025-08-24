// server.js — SmartNotes proxy (exact prompts + hardened + Rule #2 post-fix)
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';

/* ---------- Prompt builders (inline) ---------- */
const pretty = (obj) => JSON.stringify(obj ?? {}, null, 2);

function buildClassifyPrompt({ text, languages = [], categories = [], subcats = {}, hints = {} }) {
  return `
/classify

You are a deterministic short-text classifier.     
Return ONLY a single JSON object with this schema:    
{     
  "category": string,                     // must be one of CATEGORIES (case-insensitive match, but output must use provided casing)    
  "subcategory": string|null,             // only choose from SUBCATEGORIES_BY_CATEGORY[category] if present AND highly confident; else null    
  "confidence": number,                   // 0.0..1.0    
  "reason": string,                       // one sentence, why you chose it    
  "suggestedNewCategory": string|null,    // single word if a clearly better category is missing from CATEGORIES; else null    
  "suggestedNewSubcategory": string|null, // single word if a clearly better subcategory is missing; else null    
  "alternativeCategory": string|null      // optional: if text plausibly fits TWO categories, give the second best choice    
}    

CONTEXT:    
- USER_PREFERRED_LANGUAGES: ${pretty(languages)}    
- CATEGORIES: ${pretty(categories)}    
- SUBCATEGORIES_BY_CATEGORY: ${pretty(subcats)}          // include only if user has subcategories    
- HINTS_BY_CATEGORY: ${pretty(hints)}                    // optional user-learned phrases; can be empty    

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
${text ?? ''}
`.trim();
}

function buildAnalyzePrompt({ lines, languages = [], categories = [], subcats = {}, hints = {} }) {
  const joined = (lines ?? []).join("\n");
  return `
/analyze:

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
- USER_PREFERRED_LANGUAGES: ${pretty(languages)}  
- CATEGORIES: ${pretty(categories)}  
- SUBCATEGORIES_BY_CATEGORY: ${pretty(subcats)}      // include only if user has subcategories  
- HINTS_BY_CATEGORY: ${pretty(hints)}                // optional user-learned phrases; can be empty  

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
- Category MUST be from CATEGORIES (use provided casing).  
- Only output a subcategory if it already exists under the chosen category AND confidence ≥ 0.8; otherwise null.  
- Output JSON only. No markdown or extra text.  

TEXT TO CLASSIFY:
${joined}
`.trim();
}

/* ---------- Mappers ---------- */
function tryParseJSON(s) {
  if (!s) return null;
  try { return JSON.parse(s); } catch {}
  const i = s.indexOf('{'); const j = s.lastIndexOf('}');
  if (i >= 0 && j > i) { try { return JSON.parse(s.slice(i, j + 1)); } catch {} }
  return null;
}

function mapClassify(modelText) {
  const parsed = tryParseJSON(modelText);
  if (!parsed || typeof parsed !== 'object') {
    return { provider:'model', category:'Other', subcategory:null, confidence:0.1, reason:'Fallback: unparseable model output', suggestedNewCategory:null, suggestedNewSubcategory:null, alternativeCategory:null, __raw:modelText };
  }
  return {
    provider:'model',
    category: typeof parsed.category === 'string' ? parsed.category : 'Other',
    subcategory: parsed.subcategory ?? null,
    confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.5,
    reason: typeof parsed.reason === 'string' ? parsed.reason : '',
    suggestedNewCategory: parsed.suggestedNewCategory ?? null,
    suggestedNewSubcategory: parsed.suggestedNewSubcategory ?? null,
    alternativeCategory: parsed.alternativeCategory ?? null,
    __raw: modelText
  };
}

function mapAnalyze(modelText, lines) {
  const parsed = tryParseJSON(modelText);
  if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.items)) {
    return { provider:'model', items: lines.map(t => ({ text:t, category:'Other', subcategory:null, confidence:0.1, reason:'Fallback: unparseable model output', suggestedNewCategory:null, suggestedNewSubcategory:null })), __raw:modelText };
  }
  const out = parsed.items.map((it, idx) => ({
    text: lines[idx] ?? it.text ?? '',
    category: typeof it.category === 'string' ? it.category : 'Other',
    subcategory: it.subcategory ?? null,
    confidence: typeof it.confidence === 'number' ? it.confidence : 0.5,
    reason: typeof it.reason === 'string' ? it.reason : '',
    suggestedNewCategory: it.suggestedNewCategory ?? null,
    suggestedNewSubcategory: it.suggestedNewSubcategory ?? null
  }));
  return { provider:'model', items: out, __raw:modelText };
}

/* ---------- App ---------- */
const app = express();
const PORT = process.env.PORT || 7845;
const APP_KEY = process.env.APP_KEY || '';
const DEBUG_AI = process.env.DEBUG_AI === '1';
const trunc = (s, n = 2000) => (typeof s === 'string' && s.length > n ? s.slice(0, n) + '…' : s);

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

// Providers (Gemini primary, OpenAI fallback)
const geminiKey = process.env.GEMINI_API_KEY;
const genAI = geminiKey ? new GoogleGenerativeAI(geminiKey) : null;
const openaiKey = process.env.OPENAI_API_KEY;
const openai = openaiKey ? new OpenAI({ apiKey: openaiKey }) : null;
const openaiModel = process.env.OPENAI_MODEL || 'gpt-4o-mini';

// Utils
const splitLines = (text) => String(text || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
const capHints = (h, max = 100) => {
  const out = {};
  for (const [k,v] of Object.entries(h || {})) if (Array.isArray(v)) out[k] = v.slice(0, max);
  return out;
};
const safeParse = (s) => { try { return JSON.parse(String(s||'').trim()); } catch { return null; } };

async function callLLM({ prompt }) {
  // Gemini primary (2.5 Flash-Lite)
  if (genAI) {
    try {
      const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });
      const r = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: prompt }]}],
        generationConfig: { responseMimeType: 'application/json' }
      });
      const t = r?.response?.text?.() ?? r?.response?.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
      const parsed = safeParse(t);
      if (parsed) return { provider: 'gemini', raw: t, parsed };
    } catch (e) {
      if (DEBUG_AI) console.error('[gemini] error:', e?.message || e);
    }
  }
  // OpenAI fallback (GPT-5 nano)
  if (openai) {
    try {
      const r = await openai.chat.completions.create({
        model: 'gpt-5-nano',
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' },
        temperature: 0
      });
      const t = r?.choices?.[0]?.message?.content || '';
      const parsed = safeParse(t);
      if (parsed) return { provider: 'openai', raw: t, parsed };
    } catch (e) {
      if (DEBUG_AI) console.error('[openai] error:', e?.message || e);
    }
  }
  throw new Error('LLM providers failed');
}
function sanitizeSingle(obj, categories, subcats) {
  const valid = new Set(Array.isArray(categories) ? categories : []);
  const out = {
    provider: obj?.provider || 'model',
    category: 'Other', subcategory: null, confidence: 0, reason: '',
    suggestedNewCategory: null, suggestedNewSubcategory: null, alternativeCategory: null
  };
  if (!obj || typeof obj !== 'object') return out;
  const {
    category, subcategory, confidence, reason,
    suggestedNewCategory, suggestedNewSubcategory, alternativeCategory
  } = obj;

  if (typeof category === 'string' && valid.has(category)) out.category = category;
  if (typeof confidence === 'number') out.confidence = Math.max(0, Math.min(1, confidence));
  if (typeof reason === 'string') out.reason = reason.slice(0, 280);
  if (typeof suggestedNewCategory === 'string' && suggestedNewCategory.trim()) {
    out.suggestedNewCategory = suggestedNewCategory.trim().split(/\s+/)[0];
  }
  if (typeof suggestedNewSubcategory === 'string' && suggestedNewSubcategory.trim()) {
    out.suggestedNewSubcategory = suggestedNewSubcategory.trim().split(/\s+/)[0];
  }
  if (typeof alternativeCategory === 'string' && valid.has(alternativeCategory) && alternativeCategory !== out.category) {
    out.alternativeCategory = alternativeCategory;
  }

  const subs = subcats?.[out.category];
  if (Array.isArray(subs) && typeof subcategory === 'string' && subs.includes(subcategory) && out.confidence >= 0.8) {
    out.subcategory = subcategory;
  }
  return out;
}

/* ---------- Routes ---------- */
app.get('/health', (_req, res) => res.json({ ok: true }));

app.post('/classify', requireAppKey, async (req, res) => {
  try {
    const body = req.body || {};
    if (DEBUG_AI) console.log('[classify] body:', JSON.stringify(body).slice(0, 500));

    const text = typeof body.text === 'string' ? body.text : String(body.text ?? '').trim();

    let categories = Array.isArray(body.categories) && body.categories.length
      ? body.categories
      : ["To-do","Reminders","Groceries","Movies","Shows","App","Other"];

    let languages = Array.isArray(body.languages) && body.languages.length
      ? body.languages
      : ["en","es","hi","zh","ko","it","vi","fr","te","ta","mr","bn","gu","pa","ur"];

    const subcategoriesByCategory = body.subcategoriesByCategory || {};
    const hintsByCategory = body.hintsByCategory || {};

    const prompt = buildClassifyPrompt({
      text,
      languages,
      categories,
      subcats: subcategoriesByCategory,
      hints: capHints(hintsByCategory)
    });

    const { provider, raw, parsed } = await callLLM({ prompt });
    const mapped = mapClassify(JSON.stringify(parsed));
    mapped.provider = provider;

    const clean = sanitizeSingle(mapped, categories, subcategoriesByCategory);

    // POST-RULE (Rule #2): ingredient/produce → Groceries
    if (clean.category === 'Other' &&
        (clean.suggestedNewCategory || '').toLowerCase() === 'food') {
      clean.category = 'Groceries';
      clean.suggestedNewCategory = null;
      clean.reason = (clean.reason ? clean.reason + ' ' : '') + '(rule 2: ingredient → Groceries)';
    }

    if (req.query.debug === '1') {
      clean.__debug = { prompt: trunc(prompt, 8000), raw: trunc(raw, 4000) };
    }
    if (DEBUG_AI) {
      console.log('[classify] cats:', categories);
      console.log('[classify] langs:', languages);
      console.log('[classify] out:', clean);
    }
    res.json(clean);
  } catch (e) {
    if (DEBUG_AI) console.error('[classify] error:', e?.message || e);
    res.status(500).json({ error: 'classification_failed' });
  }
});

app.post('/analyze', requireAppKey, async (req, res) => {
  try {
    const body = req.body || {};
    if (DEBUG_AI) console.log('[analyze] body:', JSON.stringify(body).slice(0, 500));

    const lines = splitLines(body.text);

    let categories = Array.isArray(body.categories) && body.categories.length
      ? body.categories
      : ["To-do","Reminders","Groceries","Movies","Shows","App","Other"];

    let languages = Array.isArray(body.languages) && body.languages.length
      ? body.languages
      : ["en","es","hi","zh","ko","it","vi","fr","te","ta","mr","bn","gu","pa","ur"];

    const subcategoriesByCategory = body.subcategoriesByCategory || {};
    const hintsByCategory = body.hintsByCategory || {};

    if (!lines.length) return res.json({ items: [] });

    const prompt = buildAnalyzePrompt({
      lines,
      languages,
      categories,
      subcats: subcategoriesByCategory,
      hints: capHints(hintsByCategory)
    });

    const { provider, raw, parsed } = await callLLM({ prompt });
    const mapped = mapAnalyze(JSON.stringify(parsed), lines);
    const items = mapped.items.map(it => {
      const x = sanitizeSingle({ ...it, provider }, categories, subcategoriesByCategory);
      // POST-RULE (Rule #2): ingredient/produce → Groceries
      if (x.category === 'Other' &&
          (x.suggestedNewCategory || '').toLowerCase() === 'food') {
        x.category = 'Groceries';
        x.suggestedNewCategory = null;
        x.reason = (x.reason ? x.reason + ' ' : '') + '(rule 2: ingredient → Groceries)';
      }
      return x;
    });

    const payload = { items };
    if (req.query.debug === '1') payload.__debug = { prompt: trunc(prompt, 8000), raw: trunc(raw, 4000) };
    if (DEBUG_AI) {
      console.log('[analyze] cats:', categories);
      console.log('[analyze] langs:', languages);
      console.log('[analyze] items:', items.length);
    }
    res.json(payload);
  } catch (e) {
    if (DEBUG_AI) console.error('[analyze] error:', e?.message || e);
    res.status(500).json({ error: 'analyze_failed' });
  }
});

app.listen(PORT, () => console.log(`smartnotes-proxy (exact prompts, hardened) on ${PORT}`));
