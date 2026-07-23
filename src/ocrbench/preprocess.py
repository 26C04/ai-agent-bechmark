"""Minimal, deterministic image preprocessing for OCR model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Final, Literal

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image, ImageOps

from ocrbench.config import OcrBenchError

PREPROCESS_VERSION: Final[str] = "v1"
_PDF_SCALE: Final[float] = 300 / 72
_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class PreprocessResult:
    """A losslessly encoded, RGB model input and its preprocessing metadata."""

    png_bytes: bytes
    width: int
    height: int
    source_kind: Literal["pdf", "image"]
    duration_ms: float
    version: str


class PreprocessError(OcrBenchError):
    """Raised when a source cannot be converted by the preprocessing contract."""


def _rgb_copy(image: Image.Image) -> Image.Image:
    converted = image.convert("RGB")
    if converted is image:
        converted = image.copy()
    converted.info.clear()
    return converted


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        try:
            return _rgb_copy(oriented)
        finally:
            if oriented is not source:
                oriented.close()


def _load_pdf(path: Path) -> Image.Image:
    document = pdfium.PdfDocument(path)
    try:
        page_count = len(document)
        if page_count != 1:
            raise PreprocessError(f"PDF input must contain exactly one page; found {page_count}")

        page = document[0]
        try:
            bitmap = page.render(scale=_PDF_SCALE)
            try:
                pillow_image = bitmap.to_pil()
                try:
                    return _rgb_copy(pillow_image)
                finally:
                    pillow_image.close()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()


def _encode_png(image: Image.Image) -> tuple[bytes, int, int]:
    width, height = image.size
    with BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue(), width, height


def preprocess(path: Path) -> PreprocessResult:
    """Convert one supported source into a lossless RGB PNG without enhancement."""

    started = perf_counter()
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            source_kind: Literal["pdf", "image"] = "pdf"
            image = _load_pdf(path)
        elif suffix in _IMAGE_SUFFIXES:
            source_kind = "image"
            image = _load_image(path)
        else:
            raise PreprocessError(f"Unsupported input suffix: {path.suffix or '<none>'}")

        with image:
            png_bytes, width, height = _encode_png(image)
    except PreprocessError:
        raise
    except Exception as exc:
        raise PreprocessError(f"Failed to preprocess {path}: {exc}") from exc

    return PreprocessResult(
        png_bytes=png_bytes,
        width=width,
        height=height,
        source_kind=source_kind,
        duration_ms=(perf_counter() - started) * 1_000,
        version=PREPROCESS_VERSION,
    )


__all__ = ["PREPROCESS_VERSION", "PreprocessError", "PreprocessResult", "preprocess"]
