# Secure OpenAI Proxy (Render-ready)

This proxy hides your OpenAI API key behind a simple API the iOS app can call.

## Endpoints
- `GET /health` → `{ ok: true }`
- `POST /session` → returns a short-lived JWT (`{ token, expiresInSec }`)
- `POST /classify` (requires `Authorization: Bearer <jwt>`) → calls OpenAI Chat Completions and returns only the classification JSON string

## Local dev
```bash
cp .env.example .env
# fill OPENAI_API_KEY and JWT_SECRET (local dev only — do NOT commit .env)

npm install
npm run dev
# http://localhost:3000/health
```

## Deploy on Render
1) Push this folder to GitHub (without `.env`).
2) Create a **Web Service** on Render (Node).
   - Build: `npm install`
   - Start: `npm start`
3) In Render → **Settings → Environment**, add:
   - `OPENAI_API_KEY` = your real key (sk-...)
   - `JWT_SECRET` = long random string
   - *(optional)* `OPENAI_MODEL` = `gpt-5-nano`
   - *(optional)* `ALLOW_ORIGINS` = your web origin(s)
4) Deploy and verify `/health`.

## Using from iOS
- Configure `Proxy_Base_URL` in your app’s Info.plist (e.g., `https://<your-service>.onrender.com/`).
- Use the iOS client to call `/session` and `/classify` (the app never sees the OpenAI key).
