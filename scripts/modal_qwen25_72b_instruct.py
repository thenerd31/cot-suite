"""Modal app hosting Qwen2.5-72B-Instruct on 4×H100 for Stage 3.

Stage 3 substitute for Llama-3.1-70B-Instruct (gated; manual review
queue 2026-04-23). Non-thinking-mode baseline at 70B-class scale.

# Version pins:
#   vllm==0.19.1
#   Qwen/Qwen2.5-72B-Instruct — non-thinking-mode reasoning
#
# 72B BF16 weights ≈ 145 GB; 4×H100 (320 GB total HBM) fits with KV
# cache headroom for MAX_MODEL_LEN=32768. tensor_parallel_size=4.
# Non-thinking-mode semantics identical to Qwen2.5-7B-Instruct
# (reasoning = content = full response).
"""

from __future__ import annotations

import modal

VLLM_VERSION = "0.19.1"
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
MAX_MODEL_LEN = 32768
TP_SIZE = 4

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "transformers>=4.51,<5",
        "huggingface_hub[hf_transfer]>=0.24",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/hf",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        },
    )
)

app = modal.App("cotdiv-qwen25-72b-instruct")

hf_cache = modal.Volume.from_name("cotdiv-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=f"H100:{TP_SIZE}",
    volumes={"/cache": hf_cache},
    timeout=90 * 60,
    scaledown_window=5 * 60,
    secrets=[modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])],
)
class Qwen25_72BServer:
    """4×H100 Qwen2.5-72B-Instruct async engine (no thinking mode)."""

    @modal.enter()
    def load(self) -> None:
        import os

        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.tokenizers import get_tokenizer

        os.environ.setdefault("HF_HOME", "/cache/hf")
        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            dtype="bfloat16",
            tensor_parallel_size=TP_SIZE,
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=False,
            enforce_eager=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = get_tokenizer(MODEL_ID, trust_remote_code=False)

    @modal.method()
    async def generate(
        self,
        question: str,
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 32768,
        seed: int | None = 0,
    ) -> dict:
        import time
        import uuid

        from vllm import SamplingParams

        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            seed=seed,
        )

        request_id = f"cotdiv-{uuid.uuid4().hex[:8]}"
        start = time.monotonic()
        final_output = None
        async for output in self.engine.generate(prompt, sampling, request_id):
            final_output = output
        wall = time.monotonic() - start

        assert final_output is not None
        completion = final_output.outputs[0]
        text = completion.text

        return {
            "reasoning": text,
            "content": text,
            "raw_text": text,
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(completion.token_ids),
            "thinking_tokens": 0,
            "finish_reason": completion.finish_reason,
            "wall_clock_s": wall,
            "model_id": MODEL_ID,
            "vllm_version": VLLM_VERSION,
            "sampling": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "seed": seed,
            },
        }
