import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';
import { buildClassifyPrompt, buildAnalyzePrompt } from './server/prompts_exact.js';
import { mapClassify, mapAnalyze } from './server/mapper_strict.js';

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

// Providers (Gemini first, OpenAI fallback)
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
      if (parsed) return { provider: 'gemini', raw: t, parsed };
    } catch (e) {
      if (DEBUG_AI) console.error('[gemini] error:', e?.message || e);
    }
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

app.get('/health', (_req, res) => res.json({ ok: true }));

app.post('/classify', requireAppKey, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCategory, hintsByCategory, languages } = req.body || {};
    const prompt = buildClassifyPrompt({
      text,
      languages,
      categories,
      subcats: subcategoriesByCategory,
      hints: capHints(hintsByCategory)
    });

    const { provider, raw, parsed } = await callLLM({ prompt });
    const mapped = mapClassify(JSON.stringify(parsed)); // strict parse of model JSON
    mapped.provider = provider;

    const clean = sanitizeSingle(mapped, categories, subcategoriesByCategory);

    // Optional debug echo (?debug=1)
    if (req.query.debug === '1') {
      clean.__debug = { prompt: trunc(prompt, 8000), raw: trunc(raw, 4000) };
    }
    if (DEBUG_AI) {
      console.log('[classify] prompt:', trunc(prompt, 1200));
      console.log('[classify] mapped:', clean);
    }
    res.json(clean);
  } catch (e) {
    if (DEBUG_AI) console.error('[classify] error:', e?.message || e);
    res.status(500).json({ error: 'classification_failed' });
  }
});

app.post('/analyze', requireAppKey, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCategory, hintsByCategory, languages } = req.body || {};
    const lines = splitLines(text);
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
    // batch schema intentionally omits alternativeCategory

    const items = mapped.items.map(it => sanitizeSingle({ ...it, provider }, categories, subcategoriesByCategory));

    const payload = { items };
    if (req.query.debug === '1') {
      payload.__debug = { prompt: trunc(prompt, 8000), raw: trunc(raw, 4000) };
    }
    if (DEBUG_AI) {
      console.log('[analyze] prompt:', trunc(prompt, 1200));
      console.log('[analyze] items:', items.length);
    }
    res.json(payload);
  } catch (e) {
    if (DEBUG_AI) console.error('[analyze] error:', e?.message || e);
    res.status(500).json({ error: 'analyze_failed' });
  }
});

app.listen(PORT, () => console.log(`smartnotes-proxy (exact prompts) on ${PORT}`));
