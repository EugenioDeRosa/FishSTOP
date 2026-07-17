"""Email-level DistilBERT inference shared by training, evaluation and the app."""

import torch


MAX_BERT_TOKENS = 512
DEFAULT_CHUNK_STRIDE = 128
MAX_EMAIL_CHUNKS = 8


def encode_email_chunks(
    tokenizer,
    text: str,
    max_length: int = MAX_BERT_TOKENS,
    stride: int = DEFAULT_CHUNK_STRIDE,
    max_chunks: int = MAX_EMAIL_CHUNKS,
):
    """Tokenizza tutta l'email in finestre sovrapposte invece di troncarla."""
    if not 0 <= stride < max_length:
        raise ValueError("stride must be >= 0 and smaller than max_length")
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=True,
    )
    chunk_count = int(encoded["input_ids"].shape[0])
    if chunk_count > max_chunks:
        selected = torch.linspace(0, chunk_count - 1, steps=max_chunks).round().long().unique()
        for key, value in list(encoded.items()):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == chunk_count:
                encoded[key] = value.index_select(0, selected)
    return encoded


def aggregate_chunk_logits(logits: torch.Tensor, positive_label_id: int = 1) -> torch.Tensor:
    """
    Seleziona il blocco con il margine phishing piu' alto.

    Conservare l'intera coppia di logit del blocco rende temperature scaling,
    training evaluation e runtime matematicamente coerenti.
    """
    logits = torch.as_tensor(logits)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Expected binary chunk logits shaped [n_chunks, 2]")
    negative_label_id = 1 - int(positive_label_id)
    margins = logits[:, positive_label_id] - logits[:, negative_label_id]
    return logits[torch.argmax(margins)].unsqueeze(0)


def predict_email_logits(
    model,
    tokenizer,
    text: str,
    positive_label_id: int = 1,
    max_length: int = MAX_BERT_TOKENS,
    stride: int = DEFAULT_CHUNK_STRIDE,
    max_chunks: int = MAX_EMAIL_CHUNKS,
) -> tuple[torch.Tensor, int]:
    """Esegue inferenza su tutti i blocchi e restituisce logit email-level e numero blocchi."""
    inputs = encode_email_chunks(
        tokenizer,
        text,
        max_length=max_length,
        stride=stride,
        max_chunks=max_chunks,
    )
    inputs.pop("overflow_to_sample_mapping", None)
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    model.eval()
    with torch.inference_mode():
        chunk_logits = model(**inputs).logits
    return aggregate_chunk_logits(chunk_logits, positive_label_id).cpu(), int(chunk_logits.shape[0])
