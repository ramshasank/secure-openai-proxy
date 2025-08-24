// mapper_strict.js (ESM)

function tryParseJSON(s) {
  if (!s) return null;
  try { return JSON.parse(s); } catch {}
  const i = s.indexOf('{'); const j = s.lastIndexOf('}');
  if (i >= 0 && j > i) { try { return JSON.parse(s.slice(i, j + 1)); } catch {} }
  return null;
}

export function mapClassify(modelText) {
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

export function mapAnalyze(modelText, lines) {
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
  return { provider:'model', items: out, __raw: modelText };
}
