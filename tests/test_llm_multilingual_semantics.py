import pytest

from src.analyzer.llm_context_analyzer import (
    PHI4_OUTPUT_SCHEMA,
    _needs_targeted_intent_verifier,
    apply_email_risk_policy,
    normalize_semantic_extraction,
)


@pytest.mark.parametrize(
    ("body", "model", "expected_action"),
    [
        (
            "Proszę zapłacić 500 zł na podany rachunek.",
            {
                "action": "payment",
                "channel": "none",
                "evidence": "Proszę zapłacić 500 zł na podany rachunek.",
                "payment_method": "bank_transfer",
                "amount": "500 zł",
            },
            "pay_or_transfer",
        ),
        (
            "اضغط على الرابط للمطالبة بالجائزة الآن",
            {
                "action": "claim_reward",
                "channel": "link",
                "evidence": "للمطالبة بالجائزة الآن",
                "signals": ["incentive", "urgency"],
            },
            "claim_reward",
        ),
        (
            "このリンクを開いてパスワードを入力してください。",
            {
                "action": "provide_credentials",
                "channel": "link",
                "evidence": "パスワードを入力してください",
                "credential_type": "password",
            },
            "provide_credentials",
        ),
        (
            "请在表格中填写您的身份证号码。",
            {
                "action": "provide_information",
                "channel": "form",
                "evidence": "填写您的身份证号码",
            },
            "provide_information",
        ),
        (
            "Перейдіть за посиланням і підтвердьте вхід до облікового запису.",
            {
                "action": "verify_account",
                "channel": "link",
                "evidence": "підтвердьте вхід до облікового запису",
            },
            "verify_account",
        ),
    ],
)
def test_grounded_model_semantics_are_language_independent(
    body,
    model,
    expected_action,
):
    links = (
        [{"url": "https://example.test/action", "host": "example.test"}]
        if model["channel"] == "link"
        else []
    )
    analysis = apply_email_risk_policy(
        {
            "body_for_ai": body,
            "links": links,
            "attachments": [],
            "auth_results": {},
        },
        model,
    )

    assert analysis["requested_action"] == expected_action
    assert analysis["intent_evidence"] == model["evidence"]


def test_generic_result_triggers_verifier_without_language_keyword():
    soc = {
        "body_for_ai": "请尽快处理此事。",
        "links": [],
        "attachments": [],
    }
    semantic = normalize_semantic_extraction(
        {
            "action": "info",
            "channel": "none",
            "evidence": "",
            "confidence": 0.8,
            "ambiguity": "low",
        },
        soc=soc,
    )

    assert _needs_targeted_intent_verifier(soc, semantic) is True


def test_phi4_schema_exposes_real_confidence_and_ambiguity():
    properties = PHI4_OUTPUT_SCHEMA["properties"]

    assert properties["confidence"]["minimum"] == 0
    assert properties["confidence"]["maximum"] == 1
    assert properties["ambiguity"]["enum"] == ["none", "low", "high"]
    assert {"confidence", "ambiguity"} <= set(PHI4_OUTPUT_SCHEMA["required"])
