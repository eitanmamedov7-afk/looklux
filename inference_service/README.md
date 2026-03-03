# LookLux Inference Service (Render)

This service runs the heavy ML upload inference outside Vercel.

## Endpoints

- `GET /health`
- `POST /extract-parts`
- `POST /single-garment`

## Request/Response

### `POST /extract-parts`

Request:

```json
{
  "image_b64": "<base64 image bytes or data URI>"
}
```

Response:

```json
{
  "parts": {
    "shirt": { "image_b64": "data:image/png;base64,...", "embedding": [0.0] },
    "pants": { "image_b64": "data:image/png;base64,...", "embedding": [0.0] },
    "shoes": { "image_b64": "data:image/png;base64,...", "embedding": [0.0] }
  }
}
```

### `POST /single-garment`

Request:

```json
{
  "image_b64": "<base64 image bytes or data URI>",
  "customer_id": "optional"
}
```

Response:

```json
{
  "part_guess": "shirt",
  "embedding": [0.0],
  "image_b64": "data:image/png;base64,..."
}
```

## Deploy on Render

- Root Directory: repository root (or this folder if you prefer a split service repo)
- Build Command:
  - `pip install -r inference_service/requirements.txt`
- Start Command:
  - `python inference_service/app.py`

## Wire to Vercel

Set in Vercel project env:

- `LOOKLUX_INFERENCE_URL=https://<your-render-inference-service>.onrender.com`
- Optional:
  - `LOOKLUX_INFERENCE_EXTRACT_PATH=/extract-parts`
  - `LOOKLUX_INFERENCE_SINGLE_PATH=/single-garment`
