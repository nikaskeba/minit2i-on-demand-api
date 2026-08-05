# MiniT2I On-Demand API

A Dockerized HTTP API for [MiniT2I](https://github.com/Hope7Happiness/minit2i-torch) that loads the model only while generating an image.

The API server itself stays lightweight. Every generation runs in a short-lived subprocess that:

1. loads MiniT2I onto the GPU;
2. generates a 512×512 PNG;
3. exits and releases the model, CUDA context, and VRAM.

Downloaded Hugging Face weights remain cached on disk, so subsequent requests do not download the model again.

## Features

- Direct PNG response from `POST /generate`
- OpenAI-shaped base64 response from `POST /v1/images/generations`
- MiniT2I-B/16 (`b16`) and MiniT2I-L/16 (`l16`)
- Configurable steps, guidance scale, and seed
- Optional bearer-token or `X-API-Key` authentication
- Single-generation concurrency guard
- Persistent model download cache
- NVIDIA GPU support through Docker Compose
- Interactive OpenAPI documentation

## Requirements

- Windows or Linux with Docker and Docker Compose
- NVIDIA GPU
- NVIDIA Container Toolkit / Docker Desktop GPU support
- Enough disk space for the container image and model cache

This image uses PyTorch 2.11.0 with CUDA 12.8. It was validated on an NVIDIA GeForce RTX 5090 with 32 GB VRAM.

## Quick start

Clone the repository and start the service:

```bash
git clone <YOUR_REPOSITORY_URL>
cd minit2i-api
docker compose up -d --build
```

Check the service:

```bash
curl http://localhost:8092/health
```

Open the interactive API documentation at [http://localhost:8092/docs](http://localhost:8092/docs).

## Generate an image

### curl

```bash
curl -X POST http://localhost:8092/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A tiny red panda reading beside a rainy window, cinematic light",
    "model": "b16",
    "steps": 100,
    "seed": 42
  }' \
  --output output.png
```

### PowerShell

```powershell
$body = @{
  prompt = "A tiny red panda reading beside a rainy window, cinematic light"
  model  = "b16"
  steps  = 100
  seed   = 42
} | ConvertTo-Json

Invoke-WebRequest `
  -Uri http://localhost:8092/generate `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  -OutFile output.png
```

The response is a PNG. Useful metadata is returned in the headers:

- `X-MiniT2I-Model`
- `X-MiniT2I-Seed`
- `X-MiniT2I-Guidance`
- `X-MiniT2I-Seconds`

## Request fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | string | required | Image description, up to 4096 characters |
| `model` | `b16` or `l16` | `b16` | MiniT2I model variant |
| `steps` | integer | `100` | Inference steps, from 1 to 200 |
| `guidance_scale` | number | model default | Defaults to 2.5 for `b16` and 6.0 for `l16` |
| `seed` | integer | random | Reuse a seed for deterministic output |

Both variants generate 512×512 images. `b16` is smaller and faster; `l16` is larger.

## OpenAI-shaped endpoint

`POST /v1/images/generations` accepts the same JSON request and returns the image in `data[0].b64_json`:

```json
{
  "created": 1785960000,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAA..."
    }
  ],
  "model": "b16",
  "seed": 42,
  "guidance_scale": 2.5,
  "generation_seconds": 15.336
}
```

## Authentication

Authentication is disabled by default. Before exposing the API outside a trusted local network, create a `.env` file:

```dotenv
MINIT2I_API_KEY=replace-with-a-long-random-secret
```

Restart the service:

```bash
docker compose up -d --force-recreate
```

Send the key as either:

```text
Authorization: Bearer replace-with-a-long-random-secret
```

or:

```text
X-API-Key: replace-with-a-long-random-secret
```

## Operations

View status:

```bash
docker compose ps
docker compose logs -f minit2i-api
```

Stop the API:

```bash
docker compose down
```

Rebuild after code changes:

```bash
docker compose up -d --build
```

The model cache is stored in `./cache`. Removing that directory forces the weights to be downloaded again.

## Endpoints

| Method | Path | Result |
| --- | --- | --- |
| `GET` | `/health` | Service, GPU, and worker status |
| `POST` | `/generate` | Raw `image/png` response |
| `POST` | `/v1/images/generations` | JSON with base64 PNG |
| `POST` | `/unload` | Confirms no worker is active |
| `GET` | `/docs` | Swagger UI |

Only one generation can run at a time. A concurrent request receives HTTP `429`.

## How unloading works

The Uvicorn API process does not import PyTorch or initialize CUDA. When a request arrives, it starts `api.worker` as a child process. The worker loads the selected model, creates the image, saves the PNG, and exits. Process termination guarantees that CUDA allocations are released rather than leaving a framework cache in VRAM.

The first request is slower because it downloads the model and FLAN-T5 encoder. Later requests use the persistent Hugging Face cache but still reload the weights into GPU memory for each generation.

## Project structure

```text
api/
  app.py               Lightweight FastAPI server
  worker.py            Short-lived PyTorch inference worker
diffusers/
  pipeline.py          MiniT2I Diffusers pipeline
  mmdit.py             MiniT2I model implementation
Dockerfile
docker-compose.yml
requirements-api.txt
```

## Upstream project and model

This project packages the official [Hope7Happiness/minit2i-torch](https://github.com/Hope7Happiness/minit2i-torch) PyTorch implementation as an on-demand API. Model weights are downloaded from [MiniT2I/MiniT2I](https://huggingface.co/MiniT2I/MiniT2I).

MiniT2I is a direct-RGB text-to-image model conditioned on FLAN-T5-Large text embeddings.

## License

The upstream MiniT2I PyTorch implementation is released under the MIT License. See [LICENSE](LICENSE).
