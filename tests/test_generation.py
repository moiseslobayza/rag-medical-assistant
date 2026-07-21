from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.generation import FlanT5Generator, build_prompt


QUESTION = "  ¿Cómo confirmo mi turno?  "
CONTEXT = "Documento recuperado segundo\n\nDocumento recuperado primero"
EXPECTED_PROMPT = (
    "Use the context to answer the question.\n"
    "Answer only in Spanish.\n"
    "Do not invent information.\n"
    "If the context does not contain enough information, say that the "
    "consultorio should be contacted directly.\n\n"
    "Context:\n"
    "Documento recuperado segundo\n\n"
    "Documento recuperado primero\n\n"
    "Question:\n"
    "  ¿Cómo confirmo mi turno?  \n\n"
    "Answer in Spanish:"
)


def test_build_prompt_preserves_literal_text_context_and_question() -> None:
    assert build_prompt(question=QUESTION, context=CONTEXT) == EXPECTED_PROMPT


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    [(False, "cpu"), (True, "cuda")],
)
def test_generator_preserves_device_and_inference_contract(
    cuda_available: bool,
    expected_device: str,
) -> None:
    events: list[str] = []

    @contextmanager
    def inference_context():
        events.append("enter_inference")
        try:
            yield
        finally:
            events.append("exit_inference")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=Mock(return_value=cuda_available)),
        inference_mode=Mock(side_effect=inference_context),
    )
    input_ids = Mock(name="input_ids")
    attention_mask = Mock(name="attention_mask")
    moved_input_ids = object()
    moved_attention_mask = object()
    input_ids.to.return_value = moved_input_ids
    attention_mask.to.return_value = moved_attention_mask

    tokenizer = Mock(name="tokenizer")
    tokenizer.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    tokenizer.decode.return_value = "  Respuesta final.  "

    loaded_language_model = Mock(name="loaded_language_model")
    moved_language_model = Mock(name="moved_language_model")
    loaded_language_model.to.return_value = moved_language_model

    def generate(**kwargs: object) -> list[list[int]]:
        events.append("generate")
        return [[101, 102]]

    moved_language_model.generate.side_effect = generate
    torch_loader = Mock(return_value=fake_torch)
    tokenizer_factory = Mock(return_value=tokenizer)
    language_model_factory = Mock(return_value=loaded_language_model)

    generator = FlanT5Generator(
        llm_model_name="google/flan-t5-small",
        torch_loader=torch_loader,
        tokenizer_factory=tokenizer_factory,
        language_model_factory=language_model_factory,
    )
    response = generator.generate(question=QUESTION, context=CONTEXT)

    torch_loader.assert_called_once_with()
    fake_torch.cuda.is_available.assert_called_once_with()
    tokenizer_factory.assert_called_once_with("google/flan-t5-small")
    language_model_factory.assert_called_once_with("google/flan-t5-small")
    loaded_language_model.to.assert_called_once_with(expected_device)
    assert generator.device == expected_device
    assert generator.llm is moved_language_model
    tokenizer.assert_called_once_with(
        EXPECTED_PROMPT,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids.to.assert_called_once_with(expected_device)
    attention_mask.to.assert_called_once_with(expected_device)
    fake_torch.inference_mode.assert_called_once_with()
    moved_language_model.generate.assert_called_once_with(
        input_ids=moved_input_ids,
        attention_mask=moved_attention_mask,
        max_new_tokens=100,
        do_sample=False,
        num_beams=4,
    )
    assert events == ["enter_inference", "generate", "exit_inference"]
    tokenizer.decode.assert_called_once_with(
        [101, 102],
        skip_special_tokens=True,
    )
    assert response == "Respuesta final."
