"""Tests for the minimal image preprocessing contract."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import NoReturn

import pytest
from PIL import Image

from ocrbench.config import OcrBenchError
from ocrbench.preprocess import (
    PREPROCESS_VERSION,
    PreprocessError,
    PreprocessResult,
    preprocess,
)
from tests.helpers import make_pdf_bytes


def _open_result(result: PreprocessResult) -> Image.Image:
    image = Image.open(BytesIO(result.png_bytes))
    image.load()
    return image


def test_preprocess_one_page_letter_pdf_at_300_dpi(tmp_path: Path) -> None:
    source = tmp_path / "letter.PDF"
    source.write_bytes(make_pdf_bytes(1))

    result = preprocess(source)

    assert result.source_kind == "pdf"
    assert abs(result.width - 2550) <= 2
    assert abs(result.height - 3300) <= 2
    assert result.duration_ms > 0
    assert result.version == PREPROCESS_VERSION
    with _open_result(result) as output:
        assert output.format == "PNG"
        assert output.mode == "RGB"


def test_preprocess_applies_exif_orientation_and_removes_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oriented.JpEg"
    exif = Image.Exif()
    exif[0x0112] = 6
    with Image.new("RGB", (200, 100), "red") as image:
        image.save(source, format="JPEG", exif=exif)

    result = preprocess(source)

    assert result.source_kind == "image"
    assert (result.width, result.height) == (100, 200)
    with _open_result(result) as output:
        assert output.mode == "RGB"
        assert output.getexif() == {}
        assert "exif" not in output.info


def test_preprocess_removes_icc_profile_from_jpeg_output(tmp_path: Path) -> None:
    source = tmp_path / "profile.jpg"
    profile = b"profile-secret"
    with Image.new("RGB", (11, 7), (12, 34, 56)) as image:
        image.save(source, format="JPEG", icc_profile=profile)
    original_bytes = source.read_bytes()

    result = preprocess(source)

    assert (result.width, result.height) == (11, 7)
    assert source.read_bytes() == original_bytes
    assert profile not in result.png_bytes
    with _open_result(result) as output:
        assert "icc_profile" not in output.info
        assert output.size == (11, 7)


def test_preprocess_removes_png_transparency_and_icc_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "metadata.png"
    profile = b"profile-secret"
    palette = [10, 20, 30] + [0, 0, 0] * 255
    with Image.new("P", (5, 3), 0) as image:
        image.putpalette(palette)
        image.save(source, format="PNG", transparency=0, icc_profile=profile)

    result = preprocess(source)

    assert profile not in result.png_bytes
    with _open_result(result) as output:
        assert output.mode == "RGB"
        assert output.size == (5, 3)
        assert "icc_profile" not in output.info
        assert "transparency" not in output.info
        assert output.getpixel((0, 0)) == (10, 20, 30)


@pytest.mark.parametrize(
    ("mode", "suffix", "format_name"),
    [("RGBA", ".png", "PNG"), ("L", ".jpeg", "JPEG")],
)
def test_preprocess_converts_images_to_rgb_without_resizing(
    tmp_path: Path, mode: str, suffix: str, format_name: str
) -> None:
    source = tmp_path / f"source{suffix}"
    color: int | tuple[int, int, int, int] = 127 if mode == "L" else (10, 20, 30, 40)
    with Image.new(mode, (73, 41), color) as image:
        image.save(source, format=format_name)

    result = preprocess(source)

    assert (result.width, result.height) == (73, 41)
    with _open_result(result) as output:
        assert output.mode == "RGB"
        assert output.size == (73, 41)
        if mode == "RGBA":
            assert output.getpixel((0, 0)) == (10, 20, 30)
        else:
            pixel = output.getpixel((0, 0))
            assert isinstance(pixel, tuple)
            red, green, blue = pixel
            assert red == green == blue


@pytest.mark.parametrize("page_count", [0, 2])
def test_preprocess_rejects_pdf_page_counts(tmp_path: Path, page_count: int) -> None:
    source = tmp_path / f"pages-{page_count}.pdf"
    source.write_bytes(make_pdf_bytes(page_count))

    with pytest.raises(PreprocessError) as caught:
        preprocess(source)

    if page_count == 0:
        assert caught.value.__cause__ is not None
    else:
        assert "exactly one page" in str(caught.value)
        assert caught.value.__cause__ is None


def test_preprocess_rejects_unsupported_suffix(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("not an image", encoding="utf-8")

    with pytest.raises(PreprocessError, match="Unsupported input suffix") as caught:
        preprocess(source)

    assert isinstance(caught.value, OcrBenchError)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("suffix", [".jpg", ".png", ".pdf"])
def test_preprocess_wraps_corrupt_supported_files(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"corrupt{suffix}"
    source.write_bytes(b"not a valid source")

    with pytest.raises(PreprocessError, match="Failed to preprocess") as caught:
        preprocess(source)

    assert caught.value.__cause__ is not None


def test_preprocess_rejects_missing_supported_file(tmp_path: Path) -> None:
    source = tmp_path / "missing.png"

    with pytest.raises(PreprocessError) as caught:
        preprocess(source)

    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_pdf_native_resources_close_when_bitmap_conversion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class FailingBitmap:
        def to_pil(self) -> NoReturn:
            raise RuntimeError("conversion failed")

        def close(self) -> None:
            events.append("bitmap")

    class TrackedPage:
        def render(self, *, scale: float) -> FailingBitmap:
            assert scale == 300 / 72
            return FailingBitmap()

        def close(self) -> None:
            events.append("page")

    class TrackedDocument:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> TrackedPage:
            assert index == 0
            return TrackedPage()

        def close(self) -> None:
            events.append("document")

    monkeypatch.setattr("ocrbench.preprocess.pdfium.PdfDocument", lambda _path: TrackedDocument())

    with pytest.raises(PreprocessError) as caught:
        preprocess(tmp_path / "tracked.pdf")

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert events == ["bitmap", "page", "document"]


def test_pillow_images_close_when_png_encoding_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    with Image.new("RGB", (2, 3), "blue") as image:
        image.save(source)

    close_calls = 0
    original_close = Image.Image.close

    def tracked_close(image: Image.Image) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(image)

    def fail_save(
        _image: Image.Image,
        _file: object,
        _format: str | None = None,
        **_params: object,
    ) -> NoReturn:
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(Image.Image, "close", tracked_close)
    monkeypatch.setattr(Image.Image, "save", fail_save)

    with pytest.raises(PreprocessError) as caught:
        preprocess(source)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert close_calls >= 1
