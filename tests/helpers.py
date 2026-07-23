"""Reusable, code-generated test inputs that do not contain sensitive data."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from ocrbench.dataset import DocEntry, Manifest, SourceKind, Split
from ocrbench.schema import LineItem, OrderDocument


def make_pdf_bytes(page_count: int) -> bytes:
    """Build a minimal PDF with the requested number of blank Letter pages."""

    page_ids = list(range(3, 3 + page_count))
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
    ]
    objects.extend(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>" for _ in page_ids)

    content = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{object_id} 0 obj\n".encode())
        content.extend(body)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(content)


def make_order_document(index: int) -> OrderDocument:
    """Return deterministic synthetic ground truth for one test document."""
    return OrderDocument(
        customer_name=f"Synthetic Customer {index:02d}",
        order_no=f"ORDER-{index:04d}",
        delivery_date=f"2026-08-{index + 1:02d}",
        items=[
            LineItem(
                part_no=f"PART-{index:04d}",
                material=f"Synthetic Material {index:02d}",
                num_pieces=index + 1,
            )
        ],
    )


def make_mini_dataset(root: Path) -> Manifest:
    """Create an eight-document synthetic dataset for integration tests.

    The manifest contains four development documents (two per source kind),
    two selection documents, and two final documents. Images, ground truth,
    and the manifest are generated at runtime; no binary or real data fixture
    is required in the repository.
    """
    raw_dir = root / "raw"
    gt_dir = root / "gt"
    raw_dir.mkdir(parents=True)
    gt_dir.mkdir()

    assignments: tuple[tuple[Split, SourceKind, int], ...] = (
        ("dev", "scan", 2),
        ("dev", "photo", 2),
        ("selection", "scan", 1),
        ("selection", "photo", 1),
        ("final", "scan", 1),
        ("final", "photo", 1),
    )
    docs: list[DocEntry] = []
    index = 0
    for split, source_kind, count in assignments:
        for _ in range(count):
            temporary_image = raw_dir / f"synthetic-{index}.png"
            color = ((index * 37) % 256, (index * 73) % 256, (index * 109) % 256)
            Image.new("RGB", (32, 24), color).save(temporary_image, format="PNG")
            sha256 = hashlib.sha256(temporary_image.read_bytes()).hexdigest()
            doc_id = f"doc-{sha256[:12]}"
            anonymous_image = raw_dir / f"{doc_id}.png"
            temporary_image.rename(anonymous_image)

            ground_truth = make_order_document(index)
            (gt_dir / f"{doc_id}.json").write_text(
                json.dumps(
                    ground_truth.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            docs.append(
                DocEntry(
                    doc_id=doc_id,
                    source_kind=source_kind,
                    file=anonymous_image.name,
                    sha256=sha256,
                    split=split,
                )
            )
            index += 1

    docs.sort(key=lambda entry: entry.doc_id)
    manifest = Manifest(schema_version=1, docs=docs)
    (root / "manifest.json").write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
