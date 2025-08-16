import 'dotenv/config'
import express from 'express'
import helmet from 'helmet'
import cors from 'cors'
import rateLimit from 'express-rate-limit'
import jwt from 'jsonwebtoken'
import { z } from 'zod'
import fetch from 'node-fetch'

const {
  OPENAI_API_KEY,
  OPENAI_MODEL = 'gpt-5-nano',
  JWT_SECRET,
  PORT = 3000,
  ALLOW_ORIGINS = ''
} = process.env

if (!OPENAI_API_KEY) throw new Error('Missing OPENAI_API_KEY')
if (!JWT_SECRET) throw new Error('Missing JWT_SECRET')

const app = express()

app.use(helmet())
app.use(express.json({ limit: '32kb' }))

// CORS allowlist (comma-separated origins). Mobile apps don't need CORS.
const allow = ALLOW_ORIGINS.split(',').map(s => s.trim()).filter(Boolean)
app.use(cors({
  origin: (origin, cb) => {
    if (!origin) return cb(null, true) // native apps / curl
    if (allow.length === 0 || allow.includes(origin)) return cb(null, true)
    cb(new Error('Not allowed by CORS'))
  }
}))

// basic rate limit
app.use(rateLimit({
  windowMs: 60_000,
  max: 300,
  standardHeaders: true,
  legacyHeaders: false
}))

// --- JWT helpers ---
function issueJWT(subject, minutes = 10) {
  const now = Math.floor(Date.now() / 1000)
  const exp = now + minutes * 60
  return jwt.sign(
    { sub: subject, scope: 'classify', iat: now, exp },
    JWT_SECRET,
    { algorithm: 'HS256', issuer: 'secure-proxy' }
  )
}
function verifyJWT(req, res, next) {
  try {
    const hdr = req.headers.authorization || ''
    const token = hdr.startsWith('Bearer ') ? hdr.slice(7) : null
    if (!token) return res.status(401).json({ error: 'Missing token' })
    const payload = jwt.verify(token, JWT_SECRET, { algorithms: ['HS256'], issuer: 'secure-proxy' })
    req.user = payload
    next()
  } catch {
    return res.status(401).json({ error: 'Invalid token' })
  }
}

// --- Schemas ---
const SessionBody = z.object({ deviceId: z.string().max(128).optional() })
const ClassifyBody = z.object({
  text: z.string().min(1).max(2000),
  categories: z.array(z.string().min(1)).min(1).max(200),
  subcategoriesByCat: z.record(z.array(z.string()))
})

// --- Routes ---
app.get('/health', (req, res) => res.json({ ok: true }))

app.post('/session', (req, res) => {
  try {
    SessionBody.parse(req.body || {})
    const token = issueJWT(req.body?.deviceId || 'ios')
    return res.json({ token, expiresInSec: 600 })
  } catch {
    return res.status(400).json({ error: 'Bad request' })
  }
})

app.post('/classify', verifyJWT, async (req, res) => {
  try {
    const { text, categories, subcategoriesByCat } = ClassifyBody.parse(req.body || {})

    const system = `You are a strict JSON-only classifier. Choose one category from the list and optionally one subcategory for that category. If none fits, use null. Return JSON only with keys: category, subcategory, confidence, rationale.`
    const user = `Text: "${text}"\nCategories: ${JSON.stringify(categories)}\nSubcategoriesByCategory: ${JSON.stringify(subcategoriesByCat)}`

    const body = {
      model: OPENAI_MODEL,              // e.g. 'gpt-5-nano'
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user }
      ]
    }

    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })

    if (!r.ok) {
      const t = await r.text()
      return res.status(502).json({ error: 'Upstream error', detail: t })
    }

    const data = await r.json()
    const content = data?.choices?.[0]?.message?.content || '{}'
    // Return only what the client needs; never expose upstream secrets.
    return res.json({ classification: content })
  } catch (e) {
    if (e?.issues) return res.status(400).json({ error: 'Bad request', detail: e.issues })
    return res.status(500).json({ error: 'Server error' })
  }
})

app.listen(PORT, () => console.log(`Secure proxy running on :${PORT}`))
