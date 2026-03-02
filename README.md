# VibeCheck (Vercel Native)

This repository now runs as a Vercel-native Python web app (Flask) without Streamlit runtime.

## App Routes

- `/` Home
- `/about`
- `/legal/privacy`
- `/legal/terms`
- `/legal/accessibility`
- `/legal/beta-disclaimer`
- `/app` Auth + main app workflows
- `/delete-garments` Garment deletion workflow

## Required Environment Variables

Set these in Vercel Project Settings -> Environment Variables:

- `MONGO_URI`
- `MONGO_DB` (default: `Wardrobe_db`)
- `APP_SECRET_KEY`
- `COOKIE_SECURE` (`1` in production)

You can copy `.env.example` for local setup.

## Local Run

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
python api/index.py
```

## Deploy on Vercel

1. Push this repository to GitHub.
2. Import it into Vercel as a project.
3. Keep root directory as repository root.
4. Vercel uses `vercel.json` and deploys `api/index.py`.
5. Add the environment variables above.
6. Add your custom domain in Vercel -> Project -> Domains.

Every `git push` to the connected branch triggers a new Vercel deployment.
