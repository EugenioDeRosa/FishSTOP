r"""Repeatable intent-classification smoke benchmark for Phi-4 mini.

Run from the project root:
    .venv\Scripts\python.exe scripts\eval_phi4_intent.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyzer.llm_context_analyzer import (  # noqa: E402
    active_llm_backend,
    stream_phi4_email_analysis,
)


@dataclass(frozen=True)
class IntentCase:
    name: str
    subject: str
    body: str
    action: str
    channel: str
    links: tuple[str, ...] = ()
    attachments: tuple[str, ...] = ()
    evidence_contains: str = ""


CASES = (
    IntentCase(
        "newsletter",
        "Product update",
        "Here are the improvements released this month. No action is required.",
        "informational",
        "none",
    ),
    IntentCase(
        "sales_reply",
        "Planning a demonstration",
        "We would like to show you the product. Reply with a suitable time next week.",
        "reply",
        "email_reply",
        evidence_contains="reply",
    ),
    IntentCase(
        "invoice_payment",
        "Invoice 1042",
        "Please pay the attached invoice by Friday.",
        "pay_or_transfer",
        "supplied_attachment",
        attachments=("invoice.pdf",),
        evidence_contains="pay",
    ),
    IntentCase(
        "credential_form",
        "Mailbox verification",
        "Open the form and enter your email address and password to keep the mailbox active.",
        "provide_credentials",
        "external_form",
        links=("https://example.test/form",),
        evidence_contains="password",
    ),
    IntentCase(
        "known_password_reset",
        "Password age notice",
        "Your password expires soon. Change it from the account settings in the portal you normally use.",
        "change_account_settings",
        "normal_known_procedure",
        evidence_contains="change",
    ),
    IntentCase(
        "linked_password_reset",
        "Password reset",
        "Use the supplied link to reset your password.",
        "change_account_settings",
        "supplied_link",
        links=("https://example.test/reset",),
        evidence_contains="reset",
    ),
    IntentCase(
        "account_alert",
        "Unusual sign-in",
        "Review the unusual sign-in and confirm whether it was you.",
        "verify_account",
        "unclear",
        evidence_contains="sign-in",
    ),
    IntentCase(
        "reward_claim",
        "Your refund is ready",
        "Claim your refund using the supplied link.",
        "claim_reward",
        "supplied_link",
        links=("https://example.test/refund",),
        evidence_contains="claim",
    ),
    IntentCase(
        "late_credential_request",
        "Service information",
        ("This section contains general service information. " * 90)
        + "To finish, enter your password and recovery code in the form.",
        "provide_credentials",
        "external_form",
        links=("https://example.test/form",),
        evidence_contains="password",
    ),
)


def _soc(case: IntentCase) -> dict:
    return {
        "subject": case.subject,
        "body_for_ai": case.body,
        "links": [
            {"url": url, "host": "example.test", "source": "plain_text"}
            for url in case.links
        ],
        "attachments": [
            {
                "filename": name,
                "extension_from_filename": Path(name).suffix.lower(),
                "content_type": "application/pdf",
            }
            for name in case.attachments
        ],
        "auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "pass"},
            "DMARC": {"status": "pass"},
        },
    }


def _evaluate(case: IntentCase, timeout: int) -> dict:
    events = list(stream_phi4_email_analysis(_soc(case), timeout=timeout))
    final = events[-1]
    if final.get("status") != "ok" or not final.get("analysis"):
        return {
            "name": case.name,
            "passed": False,
            "error": final.get("message") or "analysis unavailable",
        }

    analysis = final["analysis"]
    evidence = str(analysis.get("intent_evidence") or "")
    checks = {
        "action": analysis.get("requested_action") == case.action,
        "channel": analysis.get("action_channel") == case.channel,
        "evidence": (
            not case.evidence_contains
            or case.evidence_contains.lower() in evidence.lower()
        ),
    }
    return {
        "name": case.name,
        "passed": all(checks.values()),
        "checks": checks,
        "expected": {"action": case.action, "channel": case.channel},
        "actual": {
            "action": analysis.get("requested_action"),
            "channel": analysis.get("action_channel"),
            "evidence": evidence,
            "verifier_used": (
                analysis.get("semantic_extraction") or {}
            ).get("intent_verifier_used", False),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="Run only this named case; repeat the option to select more cases.",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--json", action="store_true", help="Print JSON results.")
    args = parser.parse_args()

    selected = set(args.selected_cases or ())
    cases = tuple(case for case in CASES if not selected or case.name in selected)
    unknown = selected.difference(case.name for case in CASES)
    if unknown:
        parser.error("unknown case(s): " + ", ".join(sorted(unknown)))

    print(f"Backend: {active_llm_backend()}", file=sys.stderr)
    results = [_evaluate(case, args.timeout) for case in cases]
    passed = sum(result["passed"] for result in results)

    if args.json:
        print(json.dumps({
            "passed": passed,
            "total": len(results),
            "results": results,
        }, indent=2, ensure_ascii=False))
    else:
        for result in results:
            marker = "PASS" if result["passed"] else "FAIL"
            if result.get("error"):
                detail = result["error"]
            else:
                actual = result["actual"]
                detail = (
                    f"{actual['action']} / {actual['channel']} / "
                    f"evidence={actual['evidence']!r}"
                )
            print(f"[{marker}] {result['name']}: {detail}")
        print(f"\nResult: {passed}/{len(results)} cases passed")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
