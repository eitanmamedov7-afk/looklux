# LookLux (Vercel Native)

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
- `LOOKLUX_INFERENCE_URL` (optional, only required for image extraction/upload flows when running on Vercel)
- `LOOKLUX_INFERENCE_EXTRACT_PATH` (optional, default: `/extract-parts`)
- `LOOKLUX_INFERENCE_SINGLE_PATH` (optional, default: `/single-garment`)
- `LOOKLUX_INFERENCE_BEARER_TOKEN` (optional, for secured remote inference APIs)
- `LOOKLUX_INFERENCE_AUTH_HEADER` + `LOOKLUX_INFERENCE_AUTH_VALUE` (optional custom auth header pair)

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

Note:
- `vercel.json` now sets a non-secret default `LOOKLUX_INFERENCE_URL` with failover URLs for runtime reliability.
- If you want a different inference host, set `LOOKLUX_INFERENCE_URL` in Vercel Project Settings and redeploy.
- You can provide multiple URLs separated by commas (the app will auto-fallback to the next URL).
- You can disable code-level default fallback by setting `LOOKLUX_ENABLE_DEFAULT_INFERENCE_URL=0`.

## Note on ML dependencies in Vercel

Vercel Python functions have strict storage limits. To keep deployment size under the limit:

- Scoring uses lightweight NumPy MLP weights from `work/model_out/mlp_numpy.npz`.
- Heavy local ML extractors (Torch/Torchvision/Transformers parser stack) are not bundled by default.
- Upload/extraction flows require either:
  - local heavy deps in non-Vercel environments, or
  - a remote inference service configured via `LOOKLUX_INFERENCE_URL`.

The remote service must expose JSON endpoints for outfit extraction and single-garment processing.
By default, LookLux calls `/extract-parts` and `/single-garment`, and can also auto-detect `/api/extract-parts` and `/api/single-garment`.

## Included Inference Server

This repo includes a separate heavy-ML inference service under `inference_service/`.

- Deploy it on Render (or any full Python host), not on Vercel.
- Use:
  - Build: `pip install -r inference_service/requirements.txt`
  - Start: `python inference_service/app.py`
- Then set `LOOKLUX_INFERENCE_URL` in Vercel to that deployed service URL.
