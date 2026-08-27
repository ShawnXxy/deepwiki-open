import json
from pathlib import Path

from backend.model_routing import (
    build_model_kwargs,
    select_chat_model_task,
)


ROOT = Path(__file__).resolve().parents[1]


def test_workloads_route_to_expected_model_tiers():
    assert select_chat_model_task(is_deep_research=False) == "chat"
    assert (
        select_chat_model_task(
            is_deep_research=False,
            wiki_page_request=True,
        )
        == "reasoning"
    )
    assert (
        select_chat_model_task(
            is_deep_research=False,
            wiki_structure_request=True,
        )
        == "premium_reasoning"
    )
    assert (
        select_chat_model_task(is_deep_research=True)
        == "premium_reasoning"
    )


def test_reasoning_models_do_not_receive_temperature():
    kwargs = build_model_kwargs(
        "prod-chat",
        model_name="gpt-5.6-luna",
        temperature=1.0,
        reasoning_effort="low",
        verbosity="medium",
        max_completion_tokens=16384,
        stream=True,
    )

    assert kwargs == {
        "model": "prod-chat",
        "reasoning_effort": "low",
        "verbosity": "medium",
        "max_completion_tokens": 16384,
        "stream": True,
    }


def test_non_reasoning_chat_models_keep_temperature():
    kwargs = build_model_kwargs(
        "gpt-5.1-chat",
        temperature=0.7,
        reasoning_effort="medium",
        verbosity="high",
        max_completion_tokens=16384,
    )

    assert kwargs == {
        "model": "gpt-5.1-chat",
        "temperature": 0.7,
    }


def test_infra_config_uses_three_gpt_56_tiers():
    infra_path = ROOT / "backend" / "config" / "infra.json"
    with infra_path.open(encoding="utf-8") as config_file:
        azure_openai = json.load(config_file)["azure_openai"]

    assert azure_openai["chat"]["deployment"] == "gpt-5.6-luna"
    assert azure_openai["chat"]["api_version"] == "v1"
    assert azure_openai["reasoning"]["deployment"] == "gpt-5.6-terra"
    assert azure_openai["reasoning"]["api_version"] == "v1"
    assert (
        azure_openai["premium_reasoning"]["deployment"]
        == "gpt-5.6-sol"
    )
    assert azure_openai["premium_reasoning"]["api_version"] == "v1"


def test_arm_template_defines_all_three_gpt_56_deployments():
    template_path = (
        ROOT / "Deployments" / "templates" / "AOAI.Template.json"
    )
    with template_path.open(encoding="utf-8") as template_file:
        template = json.load(template_file)

    parameters = template["parameters"]
    assert (
        parameters["openAiChatModelModelName"]["defaultValue"]
        == "gpt-5.6-luna"
    )
    assert (
        parameters["openAiReasoningModelModelName"]["defaultValue"]
        == "gpt-5.6-terra"
    )
    assert (
        parameters["openAiPremiumReasoningModelModelName"]["defaultValue"]
        == "gpt-5.6-sol"
    )
