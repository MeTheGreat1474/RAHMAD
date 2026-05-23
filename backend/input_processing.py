"""
backend/input_processing.py

Input Pre-processing module for the RAHMAD pipeline.

Implements two parallel pre-processing branches extracted from fyp.ipynb:

  1. LLM Branch  — cleans the *full* article text (preserves case, line breaks,
                    punctuation) and returns it as a plain string for the
                    reasoning LLM agents.

  2. CLIP Branch — extracts a short *lead text* (headline + first sentence),
                    cleans and lowercases it, then tokenises text and
                    pixel-normalises the image for the CLIP Early Predictor.

Usage:
    from backend.input_retrieval import load_inputs
    from backend.input_processing import (
        clean_for_llm,
        extract_and_clean_lead_text,
        tokenize_text,
        process_image,
        preprocess_inputs,
    )

    raw_text, raw_image = load_inputs()

    # Individual steps:
    llm_text        = clean_for_llm(raw_text)
    lead_text       = extract_and_clean_lead_text(raw_text)
    input_ids, mask = tokenize_text(lead_text)
    pixel_values    = process_image(raw_image)

    # Or all at once:
    result = preprocess_inputs(raw_text, raw_image)
    # result["llm_text"]       -> str
    # result["lead_text"]      -> str
    # result["input_ids"]      -> torch.Tensor  [1, 77]
    # result["attention_mask"] -> torch.Tensor  [1, 77]
    # result["pixel_values"]   -> torch.Tensor  [1, 3, 224, 224]
"""

import torch
from PIL import Image
from cleantext import clean
from transformers import CLIPTokenizer, CLIPImageProcessor

# ---------------------------------------------------------------------------
# CLIP model identifier — keep in sync with early_predictor.py
# ---------------------------------------------------------------------------
_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_MAX_TEXT_LEN  = 77   # CLIP's maximum token sequence length


# ===========================================================================
# 1. TEXT PRE-PROCESSING
# ===========================================================================

def clean_for_llm(raw_text: str) -> str:
    """
    Clean the full article text for the LLM reasoning agents.

    Normalises Unicode and encoding artefacts; strips URLs, emails, and phone
    numbers (replacing them with readable placeholders); preserves original
    casing, punctuation, line breaks, and numbers so the LLM retains full
    semantic context.

    Parameters
    ----------
    raw_text : str
        The raw article string returned by input_retrieval.load_text().

    Returns
    -------
    str
        Cleaned text suitable for passing to an LLM.
    """
    cleaned = clean(
        raw_text,
        fix_unicode=True,           # normalise unicode characters
        to_ascii=True,              # transliterate non-ASCII to closest ASCII
        lower=False,                # preserve original casing
        no_line_breaks=False,       # keep paragraph structure
        no_urls=True,               # remove raw URLs …
        no_emails=True,             # … email addresses …
        no_phone_numbers=True,      # … and phone numbers
        no_numbers=False,           # keep numerical figures
        no_digits=False,
        no_currency_symbols=False,  # keep £ $ RM etc.
        no_punct=False,             # keep punctuation
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
    )
    print(f"[input_processing] LLM text cleaned ({len(cleaned)} chars).")
    return cleaned


def extract_and_clean_lead_text(raw_text: str) -> str:
    """
    Extract and clean the *lead text* for the CLIP Early Predictor.

    The lead text is defined as the headline plus the first body sentence
    (i.e. the first two non-empty lines of the article). This mirrors the
    notebook logic where ``lines[:2]`` is joined and cleaned with aggressive
    normalisation so it fits CLIP's 77-token window.

    Parameters
    ----------
    raw_text : str
        The raw article string returned by input_retrieval.load_text().

    Returns
    -------
    str
        Lowercased, single-line cleaned lead text ready for tokenisation.
    """
    # Split on newlines and drop blank lines — same as the notebook
    lines     = [line.strip() for line in raw_text.split("\n") if line.strip()]
    lead_text = " ".join(lines[:2])

    cleaned = clean(
        lead_text,
        fix_unicode=True,
        to_ascii=True,
        lower=True,             # lowercase for CLIP embedding consistency
        no_line_breaks=True,    # collapse to a single line
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
    )
    print(f"[input_processing] Lead text extracted & cleaned: '{cleaned[:80]}…'")
    return cleaned


# ===========================================================================
# 2. TOKENISATION (CLIP text encoder input)
# ===========================================================================

def tokenize_text(
    cleaned_lead_text: str,
    model_id: str = _CLIP_MODEL_ID,
    max_length: int = _MAX_TEXT_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenise the cleaned lead text using the CLIP tokeniser.

    Parameters
    ----------
    cleaned_lead_text : str
        Output of extract_and_clean_lead_text().
    model_id : str
        HuggingFace model identifier for the CLIP tokeniser.
    max_length : int
        Maximum token sequence length (default 77 — CLIP's hard limit).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (input_ids, attention_mask), each of shape [1, max_length].
    """
    tokenizer = CLIPTokenizer.from_pretrained(model_id)

    tokenized = tokenizer(
        cleaned_lead_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids      = tokenized["input_ids"]       # [1, 77]
    attention_mask = tokenized["attention_mask"]  # [1, 77]

    print(
        f"[input_processing] Tokenised — "
        f"input_ids: {tuple(input_ids.shape)}, "
        f"attention_mask: {tuple(attention_mask.shape)}"
    )
    return input_ids, attention_mask


# ===========================================================================
# 3. IMAGE PRE-PROCESSING (CLIP image encoder input)
# ===========================================================================

def process_image(
    raw_image: Image.Image,
    model_id: str = _CLIP_MODEL_ID,
) -> torch.Tensor:
    """
    Pre-process the PIL image for the CLIP vision encoder.

    Applies the standard CLIP image pipeline: resize to 224×224, centre-crop,
    normalise pixel values to [-1, 1].

    Parameters
    ----------
    raw_image : PIL.Image.Image
        RGB image returned by input_retrieval.load_image().
    model_id : str
        HuggingFace model identifier for the CLIP image processor.

    Returns
    -------
    torch.Tensor
        pixel_values of shape [1, 3, 224, 224].
    """
    processor = CLIPImageProcessor.from_pretrained(model_id)

    processed     = processor(images=raw_image, return_tensors="pt")
    pixel_values  = processed["pixel_values"]   # [1, 3, 224, 224]

    print(
        f"[input_processing] Image processed — "
        f"pixel_values: {tuple(pixel_values.shape)}"
    )
    return pixel_values


# ===========================================================================
# 4. COMBINED ENTRY POINT
# ===========================================================================

def preprocess_inputs(
    raw_text: str,
    raw_image: Image.Image,
    clip_model_id: str = _CLIP_MODEL_ID,
) -> dict:
    """
    Run the full input pre-processing pipeline and return all outputs.

    Parameters
    ----------
    raw_text : str
        Raw article text from input_retrieval.load_text().
    raw_image : PIL.Image.Image
        Raw article image from input_retrieval.load_image().
    clip_model_id : str
        HuggingFace CLIP model identifier used for tokeniser and processor.

    Returns
    -------
    dict with keys:
        "llm_text"       : str           — cleaned full text for LLM agents
        "lead_text"      : str           — cleaned lead text for CLIP
        "input_ids"      : torch.Tensor  — shape [1, 77]
        "attention_mask" : torch.Tensor  — shape [1, 77]
        "pixel_values"   : torch.Tensor  — shape [1, 3, 224, 224]
    """
    print("[input_processing] Starting pre-processing pipeline…")

    # --- Text ---
    llm_text  = clean_for_llm(raw_text)
    lead_text = extract_and_clean_lead_text(raw_text)
    input_ids, attention_mask = tokenize_text(lead_text, model_id=clip_model_id)

    # --- Image ---
    pixel_values = process_image(raw_image, model_id=clip_model_id)

    print("[input_processing] Pre-processing complete.")
    return {
        "llm_text":       llm_text,
        "lead_text":      lead_text,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "pixel_values":   pixel_values,
    }


# ---------------------------------------------------------------------------
# Quick self-test — run directly:  python -m backend.input_processing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from backend.input_retrieval import load_inputs

    raw_text, raw_image = load_inputs()
    result = preprocess_inputs(raw_text, raw_image)

    print("\n=== Output Summary ===")
    print(f"LLM Text (first 200 chars):\n  {result['llm_text'][:200]}")
    print(f"\nLead Text:\n  {result['lead_text']}")
    print(f"\ninput_ids shape      : {tuple(result['input_ids'].shape)}")
    print(f"attention_mask shape : {tuple(result['attention_mask'].shape)}")
    print(f"pixel_values shape   : {tuple(result['pixel_values'].shape)}")
