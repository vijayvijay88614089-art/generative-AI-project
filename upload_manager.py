"""Helpers for files deliberately placed in the project's uploads directory."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = PROJECT_ROOT / "uploads"


def project_uploads(kind: str, extensions: set[str]) -> list[Path]:
    """Return supported files from uploads/<kind>, without traversing directories."""
    if kind not in {"datasets", "documents"}:
        raise ValueError("Unknown upload category.")
    folder = (UPLOAD_ROOT / kind).resolve()
    if not folder.is_relative_to(UPLOAD_ROOT.resolve()):
        return []
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.name.casefold(),
    )
