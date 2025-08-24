// server/prompts_exact.js
// Builds EXACT prompts per your provided spec (no examples, no tweaks).

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
${text}
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
- Category MUST be from CATEGORIES (use provided casing). If no fit, choose "Other".  
- Only output a subcategory if it already exists under the chosen category AND confidence ≥ 0.8; otherwise null.  
- Output JSON only. No markdown or extra text.  

TEXT TO CLASSIFY:
${joined}
`.trim();
}

module.exports = { buildClassifyPrompt, buildAnalyzePrompt };
