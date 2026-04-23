"""Run ``scripts/run_qwen3_gpqa.py`` detached inside a Modal container.

Motivation: a local tmux+caffeinate driver dies on laptop reboot or
clamshell close. 2026-04-23 lost ~$5 and hours of progress exactly
this way. Moving the driver itself into Modal makes the run truly
fire-and-forget — the container survives laptop state, and
``results.jsonl`` streams row-by-row onto a Modal Volume that can be
pulled at any time with ``modal volume get``.

This is intentionally a **minimum wrapper**: it subprocesses the
existing driver, it does not reimplement any of its logic. If and
when B1/B3 reveal shared abstractions, factor then — not now.

Usage:
    # Fire-and-forget 32B resume from row 15:
    modal run --detach scripts/modal_detached_driver.py::run \\
        --modal-app cotdiv-qwen3-32b \\
        --start-from 15 \\
        --output-subdir qwen3_32b_gpqa_full

    # Smoke test on 14B, 5 rows, fresh run:
    modal run --detach scripts/modal_detached_driver.py::run \\
        --modal-app cotdiv-qwen3-14b \\
        --limit 5 \\
        --output-subdir qwen3_14b_smoke_modaldriver

    # Watch progress without attaching logs:
    modal volume ls cotdiv-results qwen3_32b_gpqa_full/
    modal volume get cotdiv-results qwen3_32b_gpqa_full/results.jsonl ./

Secrets required (once, via ``modal secret create``):
    hf-token            HF_TOKEN=hf_...
    anthropic-api-key   ANTHROPIC_API_KEY=sk-ant-...

Volume: ``cotdiv-results`` — auto-created on first run.
"""

from __future__ import annotations

import modal

app = modal.App("cotdiv-driver")

results_volume = modal.Volume.from_name("cotdiv-results", create_if_missing=True)

# Driver runs on CPU — all heavy lifting (vLLM inference) happens in the
# sibling Qwen3 app invoked over RPC. Driver's job is orchestration +
# autorater HTTP calls + writing rows.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "anthropic>=0.40",
        "openai>=2.8",
        "google-genai>=1.0",
        "datasets>=2.20",
        "huggingface_hub>=0.24",
        "pydantic>=2.8",
        "httpx>=0.27",
        "tenacity>=9.0",
        "typer>=0.12",
        "rich>=13.7",
        "numpy>=1.26",
        "modal",
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=[
            ".venv/**",
            ".git/**",
            "logs/**",
            "benchmarks/results/**",
            "validation/**",
            ".pytest_cache/**",
            ".mypy_cache/**",
            ".ruff_cache/**",
            "**/__pycache__/**",
            "*.pyc",
            ".env",
            ".env.*",
            "*.egg-info/**",
            "dist/**",
            "build/**",
        ],
    )
)


@app.function(
    image=image,
    volumes={"/results": results_volume},
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("anthropic-api-key"),
    ],
    timeout=8 * 3600,  # 32B full 198-question worst case ~5h; pad 3h.
    cpu=2,
    memory=4096,
)
def run(
    modal_app: str,
    output_subdir: str,
    start_from: int = 0,
    limit: int | None = None,
    commit_every_s: int = 60,
) -> dict:
    """Invoke run_qwen3_gpqa.py inside this container.

    Output path is ``/results/{output_subdir}/`` on the Modal volume;
    ``modal volume get cotdiv-results <subdir>/results.jsonl ./`` pulls
    it back to a laptop at any time.

    Periodic commits every ``commit_every_s`` seconds make the volume
    visible to ``modal volume get`` mid-run — without them, a caller
    pulling the file before the subprocess exits sees a 0-byte stub
    (Modal Volumes only propagate on commit, not on filesystem flush).
    The final commit in ``finally`` guarantees no tail rows are lost.
    """
    import os
    import subprocess
    import threading
    import time
    from pathlib import Path

    output_dir = Path("/results") / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency across Modal container restarts: if the function is
    # re-invoked after a container migration or preemption, the caller-
    # provided ``start_from`` is stale by however many rows the previous
    # invocation managed to write. Re-read the actual row count from
    # the volume and start from max(user_value, actual_count). This is
    # the 2026-04-23 crash fix — the initial invocation wrote 2 extra
    # rows (30→32) before preemption, and the retry crashed on the
    # `start_from=30 but file has 32` invariant check.
    results_path = output_dir / "results.jsonl"
    actual_rows = 0
    if results_path.exists():
        with results_path.open() as fh:
            actual_rows = sum(1 for line in fh if line.strip())
    effective_start_from = max(start_from, actual_rows)
    if effective_start_from != start_from:
        print(
            f"[driver] caller passed start_from={start_from} but volume has "
            f"{actual_rows} rows; bumping to {effective_start_from} for "
            f"idempotent retry.",
            flush=True,
        )

    env = {
        **os.environ,
        "PYTHONPATH": "/app/src:/app",
        "PYTHONUNBUFFERED": "1",
    }

    cmd = [
        "python", "/app/scripts/run_qwen3_gpqa.py",
        "--modal-app", modal_app,
        "--output-dir", str(output_dir),
        "--start-from", str(effective_start_from),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"[driver] invoking: {' '.join(cmd)}", flush=True)

    # Background heartbeat: commit the volume every N seconds so mid-run
    # `modal volume get` reflects current state. Keeping it daemon=True
    # means it dies automatically when the main thread exits.
    stop_event = threading.Event()

    def _heartbeat_commit() -> None:
        while not stop_event.wait(commit_every_s):
            try:
                results_volume.commit()
            except Exception as exc:  # commit races are non-fatal; log and continue
                print(f"[heartbeat] commit failed: {type(exc).__name__}: {exc}", flush=True)

    hb = threading.Thread(target=_heartbeat_commit, daemon=True)
    hb.start()

    try:
        proc = subprocess.run(cmd, env=env, cwd="/app")
        returncode = proc.returncode
    finally:
        stop_event.set()
        try:
            results_volume.commit()
            print(
                f"[driver] subprocess exited {locals().get('returncode', '?')}; "
                f"final volume commit done.",
                flush=True,
            )
        except Exception as exc:
            print(f"[driver] final commit failed: {type(exc).__name__}: {exc}", flush=True)

    return {"returncode": returncode, "output_subdir": output_subdir}


@app.local_entrypoint()
def main(
    modal_app: str,
    output_subdir: str,
    start_from: int = 0,
    limit: int = -1,
) -> None:
    """Local entrypoint — fires ``run.remote(...)`` and exits.

    Pair with ``modal run --detach`` and you get a cloud-side driver
    that survives laptop reboot, lid-close, and network disconnect.
    """
    result = run.remote(
        modal_app=modal_app,
        output_subdir=output_subdir,
        start_from=start_from,
        limit=None if limit < 0 else limit,
    )
    print(result)
