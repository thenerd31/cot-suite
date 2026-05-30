# ChainScope IPHR — vendored data provenance

Input data for the **B4a Arcuschin IPHR against-release reproduction**
(`scripts/validate_b4_iphr_reproduction.py`, metric `src/cotsuite/tests/iphr.py`).

## Source

- **Repo:** [jettjaniak/chainscope](https://github.com/jettjaniak/chainscope) — MIT
  (see [`LICENSES/MIT-chainscope.txt`](../../LICENSES/MIT-chainscope.txt)).
- **Pinned commit:** `bb128ac0f22dd60ab3d876a78e1c6c22fff7e830` (2026-03-30).
- **Source file:** `chainscope/data/df-wm-non-ambiguous-hard-2.pkl.gz`
  - SHA-256: `80c23e2af268e36ba651585103a2dcfbd376789f1d21add01dfeaacbd3b059cc`
  - This is the per-question dataframe produced by `scripts/iphr/make_df.py`
    (columns include `p_correct`, `p_yes`, `total_count`, `prop_id`, `comparison`,
    `answer`, `x_name`, `y_name`, `qid`).

## Vendored file

- `df_wm_non_ambiguous_hard2_9models.csv.gz`
  - SHA-256: `ad7fd888c6710a34a91a8e17bd1597e9bc1d2569cf7b7bd087794b26f5de7c1d`
  - 87,012 rows (9 models × 9,668 questions), 10 columns.
  - **Extraction recipe** (values unchanged — only column/row subset + CSV format):
    ```python
    import pandas as pd
    NINE = {"openai/gpt-4o-mini","anthropic/claude-3.5-haiku","google/gemini-pro-1.5",
            "qwen/qwq-32b","meta-llama/Llama-3.1-70B","google/gemini-2.5-flash-preview",
            "meta-llama/Llama-3.3-70B-Instruct","deepseek/deepseek-chat",
            "anthropic/claude-3.7-sonnet_1k"}
    COLS = ["model_id","qid","prop_id","comparison","answer",
            "p_correct","p_yes","total_count","x_name","y_name"]
    df = pd.read_pickle("df-wm-non-ambiguous-hard-2.pkl.gz")
    df[df.model_id.isin(NINE)][COLS].to_csv(
        "df_wm_non_ambiguous_hard2_9models.csv.gz", index=False, compression="gzip")
    ```

## Why only 9 models

Reproduction is **integer-exact (±0)** for these 9 non-oversampled models. The
other 7 models in the paper were run with adaptive two-pass oversampling
(`--unfaithful-only -n 100`: pass-1 `n=10` candidate selection → pass-2 `n=100`
re-evaluation); the pass-1 candidate-selection state is **not** in the released
aggregate df, so their flagging is not reconstructable from released artifacts.
See [`AUDIT.md`](../../AUDIT.md) and the `iphr.py` module docstring.

## Reference (ChainScope-computed) per-model unfaithful-pair counts

From `notebooks/plots_for_writeup.py:813` (the Fig-2 numerators), denominator
`n_pairs = 4892`:

| model | count | rate | paper-headline |
|---|---|---|---|
| gpt-4o-mini | 660 | 13.49% | 13% |
| claude-3.5-haiku | 363 | 7.42% | 7% |
| gemini-pro-1.5 | 320 | 6.54% | — |
| qwq-32b | 220 | 4.50% | — (excluded from headline) |
| Llama-3.1-70B | 159 | 3.25% | — |
| gemini-2.5-flash-preview | 106 | 2.17% | 2.17% |
| Llama-3.3-70B-Instruct | 102 | 2.08% | — |
| deepseek-chat | 60 | 1.23% | — |
| claude-3.7-sonnet_1k | 2 | 0.04% | 0.04% |
