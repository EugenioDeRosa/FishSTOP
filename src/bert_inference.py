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
    """Tokenize once, then materialize only the selected model windows."""
    if not 0 <= stride < max_length:
        raise ValueError("stride must be >= 0 and smaller than max_length")
    if max_chunks < 1:
        raise ValueError("max_chunks must be at least 1")

    raw = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = raw.get("input_ids") or []
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    payload_size = max(1, max_length - special_tokens)
    step = max(1, payload_size - stride)
    last_start = max(0, len(token_ids) - payload_size)
    all_starts = list(range(0, last_start + 1, step)) or [0]
    if all_starts[-1] != last_start:
        all_starts.append(last_start)
    if len(all_starts) > max_chunks:
        selected_indexes = (
            torch.linspace(0, len(all_starts) - 1, steps=max_chunks)
            .round()
            .long()
            .unique()
            .tolist()
        )
        starts = [all_starts[index] for index in selected_indexes]
    else:
        starts = all_starts

    prepared = [
        tokenizer.prepare_for_model(
            token_ids[start:start + payload_size],
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )
        for start in starts
    ]
    return tokenizer.pad(
        prepared,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


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
