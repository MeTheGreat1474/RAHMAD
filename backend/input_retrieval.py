"""
backend/input_retrieval.py

Input Retrieval module for the RAHMAD pipeline.

Reads the raw news article text and its associated image from the res/ folder
and returns them as Python-native objects ready for pre-processing.

Usage:
    from backend.input_retrieval import load_text, load_image, load_inputs

    raw_text = load_text()                        # from default res/fakeNews.txt
    raw_image = load_image()                      # from default res/fakenews.webp
    raw_text, raw_image = load_inputs()           # both at once

    # Custom paths:
    raw_text = load_text("res/myArticle.txt")
    raw_image = load_image("res/myPhoto.jpg")
"""

import os
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Default resource paths (relative to the project root, i.e. where app.py is)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # RAHMAD/
_DEFAULT_TEXT  = _PROJECT_ROOT / "res" / "news2.txt"
_DEFAULT_IMAGE = _PROJECT_ROOT / "res" / "news2.jpg"


def load_text(text_path: str | Path | None = None) -> str:
    """
    Read the news article text from disk and return it as a plain string.

    Parameters
    ----------
    text_path : str | Path | None
        Path to the .txt file.  Defaults to res/fakeNews.txt relative to the
        project root.

    Returns
    -------
    str
        The full raw article text.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given (or default) path.
    """
    path = Path(text_path) if text_path else _DEFAULT_TEXT

    if not path.is_file():
        raise FileNotFoundError(
            f"[input_retrieval] Text file not found: {path}\n"
            "Place your article in res/fakeNews.txt or pass an explicit path."
        )

    raw_text = path.read_text(encoding="utf-8")
    print(f"[input_retrieval] Loaded text from: {path}  ({len(raw_text)} chars)")
    return raw_text


def load_image(image_path: str | Path | None = None) -> Image.Image:
    """
    Load the news image from disk and return it as a PIL RGB Image.

    Parameters
    ----------
    image_path : str | Path | None
        Path to the image file (.webp / .jpg / .png / …).
        Defaults to res/fakenews.webp relative to the project root.

    Returns
    -------
    PIL.Image.Image
        The image converted to RGB mode, ready for the CLIP image processor.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given (or default) path.
    """
    path = Path(image_path) if image_path else _DEFAULT_IMAGE

    if not path.is_file():
        raise FileNotFoundError(
            f"[input_retrieval] Image file not found: {path}\n"
            "Place your image in res/fakenews.webp (or .png / .jpg) "
            "or pass an explicit path."
        )

    image = Image.open(path).convert("RGB")
    print(
        f"[input_retrieval] Loaded image from: {path}  "
        f"(size={image.size}, mode={image.mode})"
    )
    return image


def load_inputs(
    text_path: str | Path | None = None,
    image_path: str | Path | None = None,
) -> tuple[str, Image.Image]:
    """
    Convenience wrapper that retrieves both the text and image in one call.

    Parameters
    ----------
    text_path : str | Path | None
        Path to the article text file. Defaults to res/fakeNews.txt.
    image_path : str | Path | None
        Path to the article image. Defaults to res/fakenews.webp.

    Returns
    -------
    tuple[str, PIL.Image.Image]
        (raw_text, raw_image)
    """
    raw_text  = load_text(text_path)
    raw_image = load_image(image_path)
    return raw_text, raw_image


# ---------------------------------------------------------------------------
# Quick self-test — run directly:  python -m backend.input_retrieval
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    text, image = load_inputs()
    print("\n--- Raw Text Preview (first 200 chars) ---")
    print(text[:200])
    print("\n--- Image Info ---")
    print(f"Size: {image.size}  |  Mode: {image.mode}")
