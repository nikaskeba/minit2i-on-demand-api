import argparse
import importlib.util
import json
import os
import secrets
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, T5EncoderModel


MODEL_ID = os.getenv("MINIT2I_MODEL_ID", "MiniT2I/MiniT2I")
DEVICE = os.getenv("MINIT2I_DEVICE", "cuda")
CACHE_DIR = os.getenv("HF_HOME", "/cache/huggingface")
SOURCE_ROOT = Path(os.getenv("MINIT2I_SOURCE_ROOT", "/app"))


def load_pipeline_module():
    # Import the installed package first, then load this project's pipeline file.
    import diffusers as _installed_diffusers  # noqa: F401

    source_dir = SOURCE_ROOT / "diffusers"
    sys.path.insert(0, str(source_dir))
    spec = importlib.util.spec_from_file_location("minit2i_project_pipeline", source_dir / "pipeline.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the MiniT2I pipeline source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_pipeline(module, model: str):
    model_dir = "minit2i-b-16" if model == "b16" else "minit2i-l-16"
    root = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=CACHE_DIR,
            allow_patterns=[f"{model_dir}/transformer/*", f"{model_dir}/scheduler/*"],
        )
    )
    transformer = module.MiniT2IMMJiTModel.from_pretrained(
        root / model_dir / "transformer", torch_dtype=torch.bfloat16
    )
    scheduler = module.MiniT2IFlowMatchScheduler.from_pretrained(root / model_dir / "scheduler")
    text_encoder_name = transformer.mmjit_config.llm
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, cache_dir=CACHE_DIR)
    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_name, torch_dtype=torch.float32, cache_dir=CACHE_DIR
    )
    pipeline = module.MiniT2ITextToImagePipeline(
        transformer=transformer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_encoder_name=text_encoder_name,
    )
    return pipeline.to(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.load(sys.stdin)

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available inside the container")

    model = request["model"]
    seed = request.get("seed")
    if seed is None:
        seed = secrets.randbelow(2**63)
    guidance = request.get("guidance_scale")
    if guidance is None:
        guidance = 2.5 if model == "b16" else 6.0

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    started = time.perf_counter()
    module = load_pipeline_module()
    pipeline = build_pipeline(module, model)
    result = pipeline(
        request["prompt"],
        guidance_scale=guidance,
        num_inference_steps=request["steps"],
        generator=generator,
        progress=False,
    )
    result.images[0].save(args.output, format="PNG")
    print(
        json.dumps(
            {
                "seed": seed,
                "guidance_scale": guidance,
                "generation_seconds": round(time.perf_counter() - started, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
