"""Modal app hosting Llama-3.1-8B-Instruct on a single H100 for Stage 3.

Stage 3.2a: the non-Qwen sanity check for the non-thinking PHR pattern.
If Llama-3.1-8B shows the same ~20% PHR rate and >1.0 legibility-coverage
gap that Qwen2.5-7B showed (Checkpoint 1, 2026-04-24), the non-thinking
monitorability pattern is not Qwen-family-specific — it's a property of
the training regime across families.

# Version pins:
#   vllm==0.19.1
#   meta-llama/Llama-3.1-8B-Instruct — HF-gated, requires token with
#   "gated-repo read" permission + Meta license acceptance.
#
# Non-thinking contract matches Qwen2.5-7B-Instruct:
#   reasoning = content = raw_text (full response)
#   thinking_tokens = 0
# No <think> tag parsing; the entire assistant emission is treated as
# the CoT + answer trajectory by the autorater and PHR detector.
"""

from __future__ import annotations

import modal

VLLM_VERSION = "0.19.1"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_MODEL_LEN = 32768

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

app = modal.App("cotdiv-llama31-8b-instruct")

hf_cache = modal.Volume.from_name("cotdiv-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/cache": hf_cache},
    timeout=60 * 60,
    scaledown_window=5 * 60,
    secrets=[modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])],
)
class Llama31_8BServer:
    """Single-GPU Llama-3.1-8B-Instruct async engine (no thinking mode).

    Contract identical to Qwen25Server / Qwen3Server for driver compat:
        result = server.generate.remote(question="...")
        # {reasoning, content, raw_text, prompt_tokens, completion_tokens,
        #  thinking_tokens=0, finish_reason, wall_clock_s, model_id}
    """

    @modal.enter()
    def load(self) -> None:
        import os

        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.tokenizers import get_tokenizer

        os.environ.setdefault("HF_HOME", "/cache/hf")
        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            dtype="bfloat16",
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
        """One Llama-3.1-Instruct generation — non-thinking path."""
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

        assert final_output is not None, "AsyncLLMEngine produced no output"
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
