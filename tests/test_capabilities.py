# ADR-016: IICP client SDK conformance
"""#409 — multi-intent capability advertising (chat + embedding from one backend)."""

from __future__ import annotations

from iicp_client.node import _build_capabilities

CHAT = "urn:iicp:intent:llm:chat:v1"
EMBED = "urn:iicp:intent:llm:embedding:v1"


def test_chat_plus_embedding_models_advertise_two_intents():
    # Verified LM Studio case: a chat model + an embedding model → both intents.
    # Fails on the old single-capability code.
    caps = _build_capabilities(
        ["qwen2.5-coder-14b-instruct", "text-embedding-nomic-embed-text-v1.5"],
        CHAT,
        4096,
    )
    assert len(caps) == 2
    assert caps[0]["intent"] == CHAT  # configured model leads
    assert caps[0]["models"] == ["qwen2.5-coder-14b-instruct"]
    assert caps[1]["intent"] == EMBED
    assert caps[1]["models"] == ["text-embedding-nomic-embed-text-v1.5"]


def test_chat_only_yields_single_capability():
    caps = _build_capabilities(["qwen2.5:0.5b"], CHAT, 4096)
    assert len(caps) == 1
    assert caps[0]["intent"] == CHAT
    assert caps[0]["models"] == ["qwen2.5:0.5b"]
    assert caps[0]["input_modalities"] == ["text"]


def test_empty_models_yields_default_intent_capability():
    caps = _build_capabilities([], CHAT, 1024)
    assert caps == [{"intent": CHAT, "models": [], "max_tokens": 1024, "input_modalities": ["text"]}]


def test_vision_model_advertises_image_modality_chat_capability():
    # #408/ADR-046 — vision model → chat capability with image input, distinct
    # from the text-only chat capability. Fails without modality grouping.
    caps = _build_capabilities(["qwen2.5-coder-14b", "qwen/qwen3-vl-8b"], CHAT, 4096)
    assert len(caps) == 2
    assert caps[0]["intent"] == CHAT
    assert caps[0]["input_modalities"] == ["text"]
    assert caps[0]["models"] == ["qwen2.5-coder-14b"]
    assert caps[1]["intent"] == CHAT
    assert caps[1]["input_modalities"] == ["text", "image"]
    assert caps[1]["models"] == ["qwen/qwen3-vl-8b"]


def test_audio_model_advertises_audio_modality_chat_capability():
    # B1/#414/ADR-046 — audio-in chat model → chat capability with audio input,
    # distinct from text-only chat. Fails without audio detection in
    # _modalities_for_model. Mirrors the vision modality (image-in).
    caps = _build_capabilities(["qwen2.5:0.5b", "qwen2-audio-7b"], CHAT, 4096)
    assert len(caps) == 2
    assert caps[0]["input_modalities"] == ["text"]
    assert caps[1]["intent"] == CHAT
    assert caps[1]["input_modalities"] == ["text", "audio"]
    assert caps[1]["models"] == ["qwen2-audio-7b"]


def test_omni_model_advertises_image_and_audio_modalities():
    # B1 — an "omni" model accepts both image and audio in chat.
    caps = _build_capabilities(["qwen2.5-omni-7b"], CHAT, 4096)
    assert len(caps) == 1
    assert caps[0]["input_modalities"] == ["text", "image", "audio"]
