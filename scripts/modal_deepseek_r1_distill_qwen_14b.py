"""Modal app hosting DeepSeek-R1-Distill-Qwen-14B on 2×H100 for Stage 3.

Controlled comparison for Qwen3-Thinking-14B: same Qwen base, different
RL post-training (DeepSeek R1 distillation from DeepSeek-R1 → Qwen
14B). The cleanest scientific isolation in the Stage 3 study.

# Version pins:
#   vllm==0.19.1
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
#
# DeepSeek-R1-Distill models emit <think>...</think> blocks natively —
# no `enable_thinking` flag required in the chat template. Same parsing
# contract as Qwen3Server: split on <think>...</think> to extract
# reasoning, remainder is the final answer content.
#
# Tensor parallel: 2×H100 = 160 GB HBM. 14B BF16 ≈ 28 GB weights,
# ample room for 40960-token KV cache. TP=2 buys throughput under
# the longer thinking traces DeepSeek-R1 distill produces.
"""

from __future__ import annotations

import modal

VLLM_VERSION = "0.19.1"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MAX_MODEL_LEN = 40960
TP_SIZE = 2

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

app = modal.App("cotdiv-ds-r1-distill-qwen-14b")

hf_cache = modal.Volume.from_name("cotdiv-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=f"H100:{TP_SIZE}",
    volumes={"/cache": hf_cache},
    timeout=90 * 60,
    scaledown_window=5 * 60,
    secrets=[modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])],
)
class DeepSeekR1DistillQwen14BServer:
    """2×H100 DeepSeek-R1-Distill-Qwen-14B with native <think>-tag reasoning."""

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
        # DeepSeek-R1-Distill chat template doesn't take an enable_thinking
        # kwarg — the model was trained to emit <think>...</think> natively.
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
    """Split `<think>…</think>…final` output into (reasoning, content).

    Identical to Qwen3Server's parser. DeepSeek-R1-Distill emits the
    tags natively; if absent (rare truncation case) we return
    (empty, full text) so the caller can detect it.
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
