# On saturated CoT-monitorability axes, low judge-κ is a base-rate artifact, not disagreement

![Cohen's κ collapses to ~0.2 on saturated legibility while raw agreement and Gwet's AC2 stay ~0.97; on coverage κ tracks AC2.](results/kappa_degeneracy.png)

On saturated chain-of-thought monitorability axes — frontier-model legibility, where almost every CoT scores near the top of the rubric — Cohen's κ between LLM judges collapses to ~0.2 and reads as "the judges barely agree," yet raw agreement is ~0.97 and the prevalence-robust Gwet AC2 is ~0.96: the judges agree on ~98% of items. The low κ is a base-rate artifact, the Feinstein-Cicchetti prevalence paradox, not disagreement.

## The statistic is textbook; the application is the point

The mechanism is old. When one rating category dominates, κ's chance-correction term inflates and the coefficient falls toward zero even at near-perfect observed agreement (Feinstein & Cicchetti, 1990). Gwet's AC2 (Gwet, 2008) is a standard prevalence-robust alternative. Neither is new.

What is worth showing is where it bites. Cross-judge κ is increasingly used to decide whether an LLM-judged monitorability metric is reliable, and frontier-model legibility — the "high monitorability by default" regime those metrics are built to measure — is exactly the saturated case where κ is least trustworthy. A recent classifier-sensitivity result (Young, 2603.20172) reads low cross-classifier κ as construct divergence between classifiers, without separating the saturation artifact from genuine disagreement. That separation is the open question; this finding shows the artifact is real and large on the saturated axis.

## Both directions, honestly

This is not "LLM judges are unreliable." On coverage, the axis with real score variance (dominant category 52–80%, not saturated), κ runs 0.52–0.71, sits close to AC2, and the model rankings are judge-stable. Where there is spread, κ tracks agreement; it breaks only on the saturated axis. The takeaway is procedural: report a prevalence-robust coefficient and a saturation flag alongside κ before concluding a monitorability metric is judge-sensitive.

The data is a $0 re-analysis of committed scores: three judges (Claude Haiku 4.5, Claude Sonnet 4.6, Gemini 2.5 Pro) scoring 773 GPQA-Diamond chain-of-thoughts on Emmons & Zimmermann's legibility/coverage rubric. No new model calls.

## What's in the repo

[cot-suite](https://github.com/thenerd31/cot-suite) reports κ, Gwet AC2, raw agreement, and a per-judge saturation flag together for every faithfulness scorer, across at least two classifiers, and the re-analysis above reproduces from committed data at $0. The repo also carries two against-release reproductions of prior monitorability results: Turpin et al. 2023, reproduced cell-for-cell within 0.08pp; and Arcuschin et al. 2025 (ChainScope IPHR), an integer-exact metric replay of the released dataframe, partial across 4 of the 7 headline cells (the other 3 headline and 4 non-headline cells are blocked by oversampling the release does not let us reconstruct).

Full methodology and the per-pair table: [`docs/cross_judge_degeneracy.md`](docs/cross_judge_degeneracy.md).

## Status and limitations

This is a finding, not a finished paper. It rests on three judges and one rubric on GPQA-family CoTs; a gpt-4o legibility pass is still pending. The decomposition that would formally split a low κ into prevalence-artifact versus directional-disagreement (McNemar plus a bias/prevalence index), and the generalization to a binary K=2 metric at adequate N, are in progress, not done. Stated as it stands: on the saturated axis the κ collapse is a base-rate artifact; the formal artifact-versus-disagreement decomposition is the next step.

## Thread version

1/ "LLM judges disagree on CoT monitorability" is becoming a stylized fact, usually shown with a low Cohen's κ. On saturated axes that low κ is mostly a base-rate artifact. Here is the check that tells them apart. 🧵

2/ Known stats first: when one score dominates, κ's chance correction inflates and κ tanks even at near-perfect agreement (the Feinstein-Cicchetti prevalence paradox, 1990). Gwet's AC2 (2008) is the prevalence-robust fix. Not my invention; the application is the point.

3/ $0 re-analysis of committed data: 3 judges (Haiku 4.5, Sonnet 4.6, Gemini 2.5 Pro) score 773 GPQA chain-of-thoughts on Emmons & Zimmermann's legibility + coverage rubric. On legibility, κ = 0.19–0.34 but p_o = 0.97–0.99 and AC2 = 0.96–0.99. They agree on ~98% of items.

4/ Why it matters: frontier-model legibility is the "high monitorability by default" regime these metrics target, and it is near-ceiling, which is exactly where κ is least trustworthy. A judge substitute that looks unreliable (κ = 0.19) can be a faithful one (p_o = 0.97).

5/ Both directions: on coverage (real variance, dominant category 52–80%) κ = 0.52–0.71, close to AC2, rankings judge-stable. So this is not "judges are unreliable." It is "check saturation before concluding judge-sensitivity."

6/ A recent classifier-sensitivity result (Young, 2603.20172) reads low cross-classifier κ as construct divergence, without separating the saturation artifact from real disagreement. The decomposition that proves the split (McNemar + prevalence index) and a K=2 generalization are in progress, not done.

7/ Tooling and the $0 re-analysis are in cot-suite, with two against-release reproductions (Turpin ±0.08pp; Arcuschin IPHR integer-exact, partial). github.com/thenerd31/cot-suite
