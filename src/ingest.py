"""Extract raw text from uploaded .txt / .pdf / .docx screenplays."""
from __future__ import annotations

import io
import pathlib


def extract_text(filename: str, data: bytes) -> str:
    ext = pathlib.Path(filename).suffix.lower()
    if ext == ".txt":
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        import docx

        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {ext}. Upload .txt, .pdf or .docx")
