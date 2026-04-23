"""Modal app hosting Qwen3-32B on a single H200 for Stage 2 scaling (32B variant).

Sibling file to ``scripts/modal_qwen3_14b.py`` and ``scripts/modal_qwen3_8b.py``.
Differences vs the 14B/8B siblings (AUDIT-visible):

    MODEL_ID          = "Qwen/Qwen3-32B"      (vs Qwen/Qwen3-14B, Qwen/Qwen3-8B)
    modal.App(...)    = "cotdiv-qwen3-32b"    (vs 14b / 8b)
    gpu               = "H200"                (vs "H100" for 14B and 8B) ★

★ GPU divergence from the 14B/8B sibling pattern — intentional.
Qwen3-32B in bfloat16 is ~64 GB of weights alone; adding vLLM KV cache
(~10-15 GB at our max_model_len=40960) plus framework overhead would leave
≤5 GB of margin on an 80 GB H100, which is tight enough that first-run
OOM is plausible. H200 has 141 GB, gives ~70 GB of headroom. Modal prices
H200 at ~$4.50/hr vs H100 at ~$3.95/hr — 14% price premium for 17× the
memory margin. Also avoids tensor-parallel complexity (2×H100 would work
but adds NCCL / engine-config surface area).

Everything else identical to the 14B/8B siblings: same vLLM pin (0.19.1),
same Modal image, same Volume (shared hf-cache), same Secret (hf-token),
same MAX_MODEL_LEN (40960), same class name ``Qwen3Server`` so the
driver's ``--modal-app`` flag works uniformly across variants.

# Version pins (locked 2026-04-20, carried forward)
#   vllm==0.19.1        — PyPI latest stable (pypi.org/pypi/vllm/json, 2026-04-18)
#   Qwen/Qwen3-32B      — HF hybrid-mode model (public, same model-card
#                         defaults as 14B/8B per the Qwen3 family spec)
#
# Qwen3-32B recommended inference settings (thinking mode, verbatim from
# HF model card 2026-04-23):
#   temperature=0.6, top_p=0.95, top_k=20, min_p=0
#   max_new_tokens=32768 default; 38912 for complex problems
#   "DO NOT use greedy decoding"
#
# Stage 2 framing: same-family larger-scale data point for the scaling
# table. Not a paper reproduction — the paper tests Qwen3-235B only.
"""

from __future__ import annotations

import modal

VLLM_VERSION = "0.19.1"
MODEL_ID = "Qwen/Qwen3-32B"
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

app = modal.App("cotdiv-qwen3-32b")

hf_cache = modal.Volume.from_name("cotdiv-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="H200",  # divergence vs 14B/8B — see module docstring for rationale
    volumes={"/cache": hf_cache},
    timeout=60 * 60,  # kill any container stuck >1h
    scaledown_window=5 * 60,  # keep warm 5min for tight iterations
    secrets=[modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])],
)
class Qwen3Server:
    """Single-GPU Qwen3-32B async engine with thinking-mode parsing.

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
        from vllm.tokenizers import (
            get_tokenizer,  # moved from vllm.transformers_utils.tokenizer in 0.19
        )

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
