"""Modal app hosting Qwen3-14B on a single H100 for Stage 1 of the
2510.23966 reproduction pipeline.

# Version pins (locked 2026-04-20)
#   vllm==0.19.1        — PyPI latest stable (pypi.org/pypi/vllm/json, 2026-04-18)
#   Qwen/Qwen3-14B      — HF model card, card requires vllm>=0.8.5; we pin higher
#                         for Qwen3 reasoning-parser refinements landed in 0.17+
#
# Qwen3-14B recommended inference settings (thinking mode, verbatim from HF
# model card 2026-04-20):
#   temperature=0.6, top_p=0.95, top_k=20, min_p=0
#   max_new_tokens=32768 default; 38912 for complex problems
#   "DO NOT use greedy decoding"
#
# vLLM reasoning-parser flag (from model card):
#   --enable-reasoning --reasoning-parser deepseek_r1
# We use AsyncLLMEngine directly and parse <think>…</think> tags in Python,
# which avoids binding to a specific parser flag name across vLLM versions.
#
# Stage 1 framing: this is NOT a faithful reproduction of the paper's
# Qwen3-235B number — the paper's Table 1 is pooled across four datasets
# and tests a different checkpoint. Stage 1 is pipeline validation + a
# scale-down data point.
"""

from __future__ import annotations

import modal

VLLM_VERSION = "0.19.1"
MODEL_ID = "Qwen/Qwen3-14B"
MAX_MODEL_LEN = 40960  # prompt + thinking + answer headroom

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

app = modal.App("cotdiv-qwen3-14b")

hf_cache = modal.Volume.from_name("cotdiv-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/cache": hf_cache},
    timeout=60 * 60,  # kill any container stuck >1h
    scaledown_window=5 * 60,  # keep warm 5min for tight iterations
    secrets=[modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])],
)
class Qwen3Server:
    """Single-GPU Qwen3-14B async engine with thinking-mode parsing.

    Usage (from a driver):
        server = Qwen3Server()
        result = server.generate.remote(question="...")
        # result: {reasoning, content, prompt_tokens, completion_tokens,
        #          thinking_tokens, finish_reason, wall_clock_s}
    """

    @modal.enter()
    def load(self) -> None:
        """Cold-start: instantiate AsyncLLMEngine."""
        import os

        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.tokenizers import get_tokenizer  # moved from vllm.transformers_utils.tokenizer in 0.19

        os.environ.setdefault("HF_HOME", "/cache/hf")

        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=False,
            enforce_eager=False,
            # `disable_log_requests=True` removed — kwarg dropped in vllm 0.19.x
            # (was throwing TypeError on container init, 2026-04-20).
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
        """One Qwen3 generation in thinking mode.

        Defaults are the HF-card-recommended values for thinking mode.
        Returns a dict with reasoning, content, token counts, and latency.
        """
        import time
        import uuid

        from vllm import SamplingParams

        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
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
        reasoning, content = _split_thinking(text)

        reasoning_token_ids = self.tokenizer.encode(reasoning, add_special_tokens=False)

        return {
            "reasoning": reasoning,
            "content": content,
            "raw_text": text,
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(completion.token_ids),
            "thinking_tokens": len(reasoning_token_ids),
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


def _split_thinking(text: str) -> tuple[str, str]:
    """Split Qwen3's `<think>…</think>…final` output into (reasoning, content).

    Qwen3 with ``enable_thinking=True`` emits reasoning inside a single
    ``<think>…</think>`` block followed by the final answer. If the tags
    are absent — e.g. the model skipped thinking or was truncated mid-tag —
    we return (empty reasoning, full text) so the caller can detect and
    handle it.
    """
    open_tag = "<think>"
    close_tag = "</think>"
    open_idx = text.find(open_tag)
    close_idx = text.find(close_tag, open_idx + len(open_tag)) if open_idx != -1 else -1
    if open_idx == -1 or close_idx == -1:
        return "", text
    reasoning = text[open_idx + len(open_tag) : close_idx].strip()
    content = text[close_idx + len(close_tag) :].strip()
    return reasoning, content
