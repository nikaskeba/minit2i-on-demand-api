import asyncio
import base64
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field


API_KEY = os.getenv("MINIT2I_API_KEY", "")
DEVICE = os.getenv("MINIT2I_DEVICE", "cuda")
MODEL_ID = os.getenv("MINIT2I_MODEL_ID", "MiniT2I/MiniT2I")
GENERATION_TIMEOUT = int(os.getenv("MINIT2I_GENERATION_TIMEOUT", "1800"))

generation_lock = asyncio.Lock()

app = FastAPI(
    title="MiniT2I On-Demand API",
    version="1.0.0",
    description="Runs MiniT2I in a short-lived process so no model or CUDA context remains loaded.",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    model: Literal["b16", "l16"] = "b16"
    steps: int = Field(default=100, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    if not API_KEY:
        return
    bearer = authorization.removeprefix("Bearer ") if authorization else None
    supplied = x_api_key or bearer
    if supplied is None or not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _generate_in_worker(request: GenerateRequest):
    fd, output_path = tempfile.mkstemp(suffix=".png", prefix="minit2i-")
    os.close(fd)
    payload = request.model_dump()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "api.worker", "--output", output_path],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=GENERATION_TIMEOUT,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Inference worker failed").strip()
            raise RuntimeError(detail[-4000:])
        metadata = json.loads(completed.stdout.strip().splitlines()[-1])
        image_bytes = Path(output_path).read_bytes()
        return image_bytes, metadata
    finally:
        Path(output_path).unlink(missing_ok=True)


async def _run_generation(request: GenerateRequest):
    if generation_lock.locked():
        raise HTTPException(status_code=429, detail="A generation is already running; retry later")
    async with generation_lock:
        try:
            return await asyncio.to_thread(_generate_in_worker, request)
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Generation timed out") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": generation_lock.locked(),
        "busy": generation_lock.locked(),
        "cuda_device_present": Path("/dev/nvidiactl").exists() or Path("/dev/dxg").exists(),
        "device": DEVICE,
        "model_id": MODEL_ID,
        "load_mode": "per-request subprocess",
    }


@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(request: GenerateRequest):
    image_bytes, metadata = await _run_generation(request)
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "X-MiniT2I-Model": request.model,
            "X-MiniT2I-Seed": str(metadata["seed"]),
            "X-MiniT2I-Guidance": str(metadata["guidance_scale"]),
            "X-MiniT2I-Seconds": str(metadata["generation_seconds"]),
        },
    )


@app.post("/v1/images/generations", dependencies=[Depends(require_api_key)])
async def openai_compatible_generate(request: GenerateRequest):
    image_bytes, metadata = await _run_generation(request)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(image_bytes).decode("ascii")}],
        "model": request.model,
        **metadata,
    }


@app.post("/unload", dependencies=[Depends(require_api_key)])
def unload():
    if generation_lock.locked():
        raise HTTPException(status_code=409, detail="A generation worker is currently running")
    return {"status": "unloaded"}
