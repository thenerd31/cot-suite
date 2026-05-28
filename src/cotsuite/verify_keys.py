"""One-shot preflight for external API keys.

Motivation: third-time placeholder-key incident (Modal 2026-04-20, Google
AI Studio 2026-04-21, OpenAI 2026-04-23) each cost wasted runtime and
produced a false-negative run that couldn't actually execute. This module
makes one trivial API call against each configured provider and fails
loudly with a provider-specific error if a key is placeholder, expired,
unauthorized, or (Anthropic) out-of-credits.

CLI usage:
    python -m cotsuite.verify_keys                   # check all configured providers
    python -m cotsuite.verify_keys --budget-check    # additionally verify Anthropic credits
    python -m cotsuite.verify_keys --providers anthropic,openai

Library usage (called at import time from every spend-incurring script):
    from cotsuite.verify_keys import require_keys
    require_keys(["anthropic", "openai", "huggingface"])

Each `check_<provider>()` returns a `CheckResult` — no exceptions raised
internally — so the runner can report every failure at once rather than
bailing on the first. `require_keys()` aggregates and raises on any
failure with a formatted multi-provider error.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER_MARKERS = ("...", "your-key-here", "changeme", "xxx")
MIN_KEY_LENGTH = 20  # shortest credible real key across providers
ANTHROPIC_MIN_CREDIT_USD = 5.0


@dataclass
class CheckResult:
    provider: str
    ok: bool
    detail: str  # one-line summary (success confirmation OR failure cause)
    hint: str = ""  # remediation hint on failure


def _looks_like_placeholder(key: str | None) -> bool:
    if not key:
        return True
    if len(key) < MIN_KEY_LENGTH:
        return True
    lower = key.lower()
    return any(marker in lower for marker in PLACEHOLDER_MARKERS)


def check_anthropic(*, budget_check: bool = False) -> CheckResult:
    """Minimal messages call on Haiku; optionally probe credit balance."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if _looks_like_placeholder(key):
        return CheckResult(
            provider="anthropic",
            ok=False,
            detail=f"ANTHROPIC_API_KEY is missing or a placeholder (len={len(key or '')}).",
            hint="Set a real sk-ant-... key in .env (console.anthropic.com/settings/keys).",
        )
    try:
        import anthropic
    except ImportError:
        return CheckResult(
            provider="anthropic",
            ok=False,
            detail="anthropic package not installed.",
            hint="pip install 'anthropic>=0.40'",
        )

    client = anthropic.Anthropic(api_key=key)
    try:
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1,
            messages=[{"role": "user", "content": "ok"}],
        )
    except anthropic.AuthenticationError as exc:
        return CheckResult(
            provider="anthropic",
            ok=False,
            detail=f"AuthenticationError: {exc}",
            hint="Key is unauthorized — regenerate at console.anthropic.com/settings/keys.",
        )
    except anthropic.BadRequestError as exc:
        msg = str(exc)
        if "credit balance" in msg.lower() or "billing" in msg.lower():
            return CheckResult(
                provider="anthropic",
                ok=False,
                detail=f"Credits exhausted: {exc}",
                hint="Top up at console.anthropic.com/settings/billing.",
            )
        return CheckResult(
            provider="anthropic",
            ok=False,
            detail=f"BadRequestError: {exc}",
            hint="Unexpected 400 from minimal call — check API status.",
        )
    except Exception as exc:
        return CheckResult(
            provider="anthropic",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            hint="Unexpected error — check network + API status.",
        )

    detail = "Haiku 4.5 minimal call OK."
    if budget_check:
        # End-user keys can't hit the admin billing API (that requires
        # sk-ant-admin-... keys). A successful 1-token call proves balance
        # > cost-of-1-token but NOT > $5. Surface that honestly rather
        # than faking a threshold check.
        if key and key.startswith("sk-ant-admin-"):
            detail += " (admin key detected — balance API would be accessible, not yet wired up)"
        else:
            detail += (
                f" [budget-check warning] Automatic balance verification against "
                f"${ANTHROPIC_MIN_CREDIT_USD:.0f} threshold requires admin API key. "
                f"Manually confirm balance at console.anthropic.com/settings/billing."
            )
    return CheckResult(provider="anthropic", ok=True, detail=detail)


def check_openai() -> CheckResult:
    """Minimal model-list call. Cheapest OpenAI call that exercises auth."""
    key = os.environ.get("OPENAI_API_KEY")
    if _looks_like_placeholder(key):
        return CheckResult(
            provider="openai",
            ok=False,
            detail=f"OPENAI_API_KEY is missing or a placeholder (len={len(key or '')}).",
            hint="Set a real sk-... (or sk-proj-...) key in .env (platform.openai.com/api-keys).",
        )
    try:
        import openai
    except ImportError:
        return CheckResult(
            provider="openai",
            ok=False,
            detail="openai package not installed.",
            hint="pip install 'openai>=2.8'",
        )

    client = openai.OpenAI(api_key=key)
    try:
        # models.list is free and minimal; proves auth + network without burning spend.
        next(iter(client.models.list()), None)
    except openai.AuthenticationError as exc:
        return CheckResult(
            provider="openai",
            ok=False,
            detail=f"AuthenticationError: {exc}",
            hint="Key is unauthorized — regenerate at platform.openai.com/api-keys.",
        )
    except openai.PermissionDeniedError as exc:
        return CheckResult(
            provider="openai",
            ok=False,
            detail=f"PermissionDeniedError: {exc}",
            hint="Key lacks model.list scope — use a key with default scopes.",
        )
    except Exception as exc:
        return CheckResult(
            provider="openai",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            hint="Unexpected error — check network + API status.",
        )
    return CheckResult(provider="openai", ok=True, detail="models.list OK.")


def check_huggingface() -> CheckResult:
    """Whoami endpoint. Validates HF_TOKEN for gated-dataset access (GPQA)."""
    key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if _looks_like_placeholder(key):
        return CheckResult(
            provider="huggingface",
            ok=False,
            detail=f"HF_TOKEN is missing or a placeholder (len={len(key or '')}).",
            hint="Set a real hf_... token in .env (huggingface.co/settings/tokens).",
        )
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        return CheckResult(
            provider="huggingface",
            ok=False,
            detail="huggingface_hub not installed.",
            hint="pip install 'huggingface_hub>=0.24' (bundled with datasets).",
        )

    try:
        info = HfApi().whoami(token=key)
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            return CheckResult(
                provider="huggingface",
                ok=False,
                detail=f"HTTP {status}: {exc}",
                hint="Token is unauthorized — regenerate at huggingface.co/settings/tokens.",
            )
        return CheckResult(
            provider="huggingface",
            ok=False,
            detail=f"HfHubHTTPError: {exc}",
            hint="Unexpected HTTP error — check network + hub status.",
        )
    except Exception as exc:
        return CheckResult(
            provider="huggingface",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            hint="Unexpected error — check network + hub status.",
        )
    return CheckResult(
        provider="huggingface",
        ok=True,
        detail=f"whoami OK (user={info.get('name', '?')}).",
    )


def check_modal() -> CheckResult:
    """`modal token info` — exits 0 iff a valid token is configured.

    Modal tokens live in ~/.modal.toml, NOT in .env. Putting token IDs
    in .env silently overrides the toml with placeholder values (see
    .env.example — this exact incident happened on 2026-04-20).

    When the script is ALREADY running inside a Modal container
    (``MODAL_TASK_ID`` set by the Modal runtime), the SDK uses task-
    injected credentials — there is no CLI token file, and checking
    for one would give a spurious fail. Short-circuit to OK in that
    case since our very presence inside the container proves Modal
    auth is working.
    """
    if os.environ.get("MODAL_TASK_ID"):
        return CheckResult(
            provider="modal",
            ok=True,
            detail="running inside Modal container (MODAL_TASK_ID set); "
            "task-injected credentials assumed.",
        )
    modal_toml = Path.home() / ".modal.toml"
    if not modal_toml.exists():
        return CheckResult(
            provider="modal",
            ok=False,
            detail=f"{modal_toml} not found.",
            hint="Run `modal setup` to authenticate.",
        )
    env_token = os.environ.get("MODAL_TOKEN_ID")
    if env_token and _looks_like_placeholder(env_token):
        return CheckResult(
            provider="modal",
            ok=False,
            detail=f"MODAL_TOKEN_ID in env is a placeholder (len={len(env_token)}); "
            f"it overrides the real token in ~/.modal.toml.",
            hint="Unset MODAL_TOKEN_ID in .env (or remove it entirely — modal CLI reads ~/.modal.toml).",
        )
    try:
        result = subprocess.run(
            ["modal", "token", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return CheckResult(
            provider="modal",
            ok=False,
            detail="`modal` CLI not on PATH.",
            hint="pip install modal (and `modal setup` if not yet authenticated).",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            provider="modal",
            ok=False,
            detail="`modal token info` timed out after 15s.",
            hint="Check network / modal.com status.",
        )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout).strip()
        return CheckResult(
            provider="modal",
            ok=False,
            detail=f"`modal token info` exited {result.returncode}: {stderr}",
            hint="Run `modal setup` or `modal token new` to re-authenticate.",
        )
    # Output shape: "Token: ak-...\nWorkspace: <name> (ac-...)\n..." — we
    # surface the workspace line so preflight logs show WHICH account is
    # about to be billed.
    workspace_line = next(
        (line.strip() for line in result.stdout.splitlines() if line.startswith("Workspace:")),
        "Workspace: <unknown>",
    )
    return CheckResult(provider="modal", ok=True, detail=f"token info OK. {workspace_line}")


def _check_openai_compatible(
    *, provider: str, env_key: str, base_url: str, key_hint: str
) -> CheckResult:
    """Shared check for OpenAI-compatible providers (Together/Fireworks/DeepSeek/OpenRouter).

    Uses the OpenAI SDK pointed at ``base_url``; ``models.list()`` exercises
    auth + network without burning generation spend.
    """
    key = os.environ.get(env_key)
    if _looks_like_placeholder(key):
        return CheckResult(
            provider=provider,
            ok=False,
            detail=f"{env_key} is missing or a placeholder (len={len(key or '')}).",
            hint=key_hint,
        )
    try:
        import openai
    except ImportError:
        return CheckResult(
            provider=provider,
            ok=False,
            detail="openai package not installed.",
            hint="pip install 'openai>=2.8'",
        )

    client = openai.OpenAI(api_key=key, base_url=base_url)
    try:
        next(iter(client.models.list()), None)
    except openai.AuthenticationError as exc:
        return CheckResult(
            provider=provider,
            ok=False,
            detail=f"AuthenticationError: {exc}",
            hint=f"{env_key} is unauthorized — regenerate it.",
        )
    except Exception as exc:
        return CheckResult(
            provider=provider,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            hint="Unexpected error — check network + API status.",
        )
    return CheckResult(provider=provider, ok=True, detail="models.list OK.")


def check_together() -> CheckResult:
    """Together AI (OpenAI-compatible)."""
    return _check_openai_compatible(
        provider="together",
        env_key="TOGETHER_API_KEY",
        base_url="https://api.together.ai/v1",
        key_hint="Set a real TOGETHER_API_KEY in .env (api.together.ai/settings/api-keys).",
    )


def check_fireworks() -> CheckResult:
    """Fireworks AI (OpenAI-compatible)."""
    return _check_openai_compatible(
        provider="fireworks",
        env_key="FIREWORKS_API_KEY",
        base_url="https://api.fireworks.ai/inference/v1",
        key_hint="Set a real FIREWORKS_API_KEY in .env (app.fireworks.ai → Settings → API Keys).",
    )


def check_deepseek() -> CheckResult:
    """DeepSeek direct API (OpenAI-compatible)."""
    return _check_openai_compatible(
        provider="deepseek",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        key_hint="Set a real DEEPSEEK_API_KEY in .env (platform.deepseek.com/api_keys).",
    )


def check_openrouter() -> CheckResult:
    """OpenRouter aggregator (OpenAI-compatible)."""
    return _check_openai_compatible(
        provider="openrouter",
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        key_hint="Set a real OPENROUTER_API_KEY in .env (openrouter.ai/keys).",
    )


PROVIDER_CHECKS = {
    "anthropic": check_anthropic,
    "openai": check_openai,
    "huggingface": check_huggingface,
    "modal": check_modal,
    "together": check_together,
    "fireworks": check_fireworks,
    "deepseek": check_deepseek,
    "openrouter": check_openrouter,
}


def run_checks(providers: list[str], *, budget_check: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    for p in providers:
        fn = PROVIDER_CHECKS.get(p)
        if fn is None:
            results.append(
                CheckResult(
                    provider=p,
                    ok=False,
                    detail=f"Unknown provider '{p}'.",
                    hint=f"Valid providers: {', '.join(PROVIDER_CHECKS)}.",
                )
            )
            continue
        if p == "anthropic":
            results.append(check_anthropic(budget_check=budget_check))
        else:
            results.append(fn())
    return results


def format_report(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        marker = "OK  " if r.ok else "FAIL"
        lines.append(f"  [{marker}] {r.provider}: {r.detail}")
        if not r.ok and r.hint:
            lines.append(f"         hint: {r.hint}")
    return "\n".join(lines)


def require_keys(providers: list[str], *, budget_check: bool = False) -> None:
    """Import-time assertion. Raises SystemExit(2) on any failure.

    Called at the top of every spend-incurring script. Prints a unified
    report of all failures at once rather than failing on the first —
    otherwise users fix one key, re-run, fail on the next, fix it,
    re-run, etc.
    """
    results = run_checks(providers, budget_check=budget_check)
    failed = [r for r in results if not r.ok]
    if failed:
        sys.stderr.write(
            f"\nPREFLIGHT FAILED ({len(failed)}/{len(results)} providers):\n"
            f"{format_report(results)}\n\n"
        )
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cotsuite.verify_keys",
        description="Verify external API keys before spend-incurring runs.",
    )
    parser.add_argument(
        "--providers",
        default="anthropic,openai,huggingface,modal",
        help="Comma-separated providers to check (default: all four).",
    )
    parser.add_argument(
        "--budget-check",
        action="store_true",
        help="Additionally probe Anthropic credit balance (best-effort).",
    )
    args = parser.parse_args(argv)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    results = run_checks(providers, budget_check=args.budget_check)
    print(format_report(results))
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\n{len(failed)}/{len(results)} checks failed.", file=sys.stderr)
        return 2
    print(f"\nAll {len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
