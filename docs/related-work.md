# Related work

cot-suite v0.1 sits in a fast-moving literature on chain-of-thought monitorability and faithfulness. This page is the short version. The [full related-work doc on GitHub](https://github.com/thenerd31/cot-suite/blob/main/docs/related_work.md) carries the full eight-paper treatment + lit-sweep methodology + competing-bundle check.

## Charter document

- **Korbak et al. — arXiv 2507.11473 — "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety."** A 41-author position paper spanning Anthropic, OpenAI, DeepMind, METR, Apollo, MATS. Recommendation (1): "develop effective evaluations of CoT monitorability." cot-suite is a direct operationalization of that recommendation.

## Closest measurement-side prior work

- **Meek et al. — arXiv 2510.27378.** 4 frontier sibling pairs on BBH/GPQA/MMLU. Cue-injection from Chen + their own verbosity construct. cot-suite extends below their 32B floor with the open-weight 7B-72B scaling table.
- **Chua & Evans — arXiv 2501.08156.** Cleanest thinking-vs-non-thinking head-to-head on MMLU. **Direct checkpoint overlap on Qwen2.5-72B-Instruct, DeepSeek-R1-Distill-Qwen-14B, and a Llama-70B distilled on R1 CoTs** (paper does not specify the canonical HF identifier; our run is on the canonical `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` regardless). cot-suite extends to GPQA-Diamond and adds the within-family scale ladder.
- **Guan et al. (OpenAI) — arXiv 2512.18311.** Defines g-mean² monitorability metric and a three-archetype taxonomy (intervention / process / outcome-property). cot-suite plans to adopt g-mean² in v0.1.1, pending a binarization-scheme decision (see [roadmap](roadmap.md)).
- **Lewis-Lim et al. — arXiv 2508.19827.** Confidence-trajectory PHR analysis on soft-reasoning + GPQA-Main. Direct checkpoint overlap on Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Qwen3-32B (3 of our 8 models). Their confidence-trajectory PHR signal is on the v0.2 wishlist.

## Recent work surfaced by the literature sweep

- **Yueh-Han Chen et al. — arXiv 2603.05706 — CoT-Control.** Different threat model — *intentional* CoT control (steganographic / adversarial) vs cot-suite's structural-faithfulness focus. Anthropic + UK AISI + MATS authorship overlap with Korbak. Direct model overlap on Qwen3-14B / 32B.
- **MonitorBench — arXiv 2603.28590.** Stand-alone 1,514-instance benchmark with novel "direct concealment" + "monitor-aware evasion" stress-test paradigm. Verified non-competing — not on PyPI, doesn't bundle prior methods. cot-suite is the **first PyPI-installable bundle of Lanham 2307.13702, Turpin 2305.04388, Chen 2505.05410, Arcuschin 2503.08679, and Emmons & Zimmermann 2510.23966 under a unified CLI**; MonitorBench is a different category (single benchmark, no PyPI).
- **Baker et al. — arXiv 2503.11926.** RL-induced CoT obfuscation; the "Goodharting CoT monitors" risk Korbak warns about.
- **METR — Fine-tuning experiments on CoT controllability (April 1, 2026).** SFT lifts CoT-Control compliance 2.9% → 8.8% average across GPT-OSS-{20B,120B} + Qwen-3-{8B,32B}. Validates CoT-Control as the right intentional-controllability benchmark.

## Adjacent but out of v0.1 scope

- **McGuinness, Serrano, Bailey, Emmons — arXiv 2512.11949.** "Neural Chameleons" — activation-level steganography. Conceptual follow-up to Emmons & Zimmermann 2510.23966 (the autorater paper cot-suite ports). Highest-priority candidate for a future activation-level scorer in v0.2 (gated on the optional `cot-suite[activations]` extra).
- **Schoen et al. — arXiv 2509.15541.** Apollo + OpenAI anti-scheming training. Closed frontier only; cited as downstream-safety motivation, no numeric comparison.
- **Lindner et al. — arXiv 2512.22154.** UK AISI deployment-monitoring framework for production CoT monitoring. cot-suite is the kind of *measurement* tooling such a deployment framework consumes.

## v0.1 positioning

cot-suite is **not** a competing benchmark or a novel methodology. The framing throughout: **replication + tooling + open-weight scaling**, not new science. The five wrapped methodologies are Lanham, Turpin, Chen, Arcuschin, Emmons-Zimmermann; cot-suite wires them into one PyPI package and runs them consistently across 8 open-weight models on GPQA-Diamond.

The tooling-layer contribution distinct from any individual prior paper's evaluation: **cot-suite v0.1 is the first PyPI-installable bundle of Lanham 2307.13702, Turpin 2305.04388, Chen 2505.05410, Arcuschin 2503.08679, and Emmons & Zimmermann 2510.23966 under a unified CLI.** This claim is verified by 13 candidate-package PyPI-direct probes (all 404), a GitHub topic-search for "cot faithfulness evaluation" (7 single-paper companion repos with 0-2 stars, none bundling), and a comprehensive-benchmark check (MonitorBench, OpenAI's monitorability-evals, Yueh-Han Chen's CoT-Control all distinct categories). See the [full related-work doc](https://github.com/thenerd31/cot-suite/blob/main/docs/related_work.md) for the complete competing-bundle methodology.
