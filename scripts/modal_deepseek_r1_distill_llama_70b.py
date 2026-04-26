"""Modal app hosting DeepSeek-R1-Distill-Llama-70B on 4×H100 for Stage 3.

Cross-architecture R1-distill: same RL post-training methodology as
DeepSeek-R1-Distill-Qwen-14B, different base (Llama 3.x instead of
Qwen). Isolates the effect of the R1 distillation recipe across
base architectures.

# Version pins:
#   vllm==0.19.1
#   deepseek-ai/DeepSeek-R1-Distill-Llama-70B
#
# 70B BF16 ≈ 140 GB; 4×H100 (320 GB HBM) fits with KV cache headroom.
# tensor_parallel_size=4. Same <think>-tag reasoning contract as the
# 14B Distill — the R1 distill recipe is consistent across base
# models.
"""

from __future__ import annotations

import modal

VLLM_VERSION = "0.19.1"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
MAX_MODEL_LEN = 40960
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

app = modal.App("cotdiv-ds-r1-distill-llama-70b")

hf_cache = modal.Volume.from_name("cotdiv-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=f"H100:{TP_SIZE}",
    volumes={"/cache": hf_cache},
    timeout=120 * 60,
    scaledown_window=5 * 60,
    secrets=[modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])],
)
class DeepSeekR1DistillLlama70BServer:
    """4×H100 DeepSeek-R1-Distill-Llama-70B with native <think>-tag reasoning."""

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
    """Split DeepSeek-R1-Distill output into (reasoning, content).

    DeepSeek-R1-Distill chat template injects ``<think>`` into the
    PROMPT (not the assistant output), so the model only emits
    ``</think>`` (closing tag) followed by the final answer. The
    original two-tag splitter returned (reasoning="", content=full)
    on every R1-Distill row, which broke the PHR detector with
    "empty reasoning trace" errors. Fix shipped 2026-04-25 in
    scripts/fix_deepseek_split.py and ported here so future
    inference runs split correctly inline.

    Behavior:
      - both ``<think>`` and ``</think>`` present → split on both
        (Qwen3-Thinking compatibility).
      - only ``</think>`` present (DeepSeek-R1 pattern) → reasoning
        is everything before ``</think>``, content is everything
        after.
      - neither tag → return (empty, full text) so the caller can
        detect missing thinking output.
    """
    open_tag = "<think>"
    close_tag = "</think>"
    close_idx = text.find(close_tag)
    if close_idx == -1:
        return "", text
    open_idx = text.find(open_tag)
    if open_idx == -1 or open_idx >= close_idx:
        # DeepSeek-R1 pattern: closing tag only, or open tag came after
        # close (unusual, treat as no opening tag).
        reasoning = text[:close_idx].strip()
    else:
        reasoning = text[open_idx + len(open_tag) : close_idx].strip()
    content = text[close_idx + len(close_tag) :].strip()
    return reasoning, content
