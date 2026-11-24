"""
DermAI Storage Utilities for Portable Image Path Handling.

Handles cross-platform path resolution between local development environments
(e.g., Windows with absolute drive paths) and containerized Linux deployments
(e.g., Docker with POSIX relative paths under /app/backend/uploads/).
"""

from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Union, Optional
from .config import settings


def get_relative_upload_path(
    absolute_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Converts an absolute filesystem path to a portable POSIX-style relative path
    relative to settings.UPLOAD_DIR (or base_dir).

    Example:
        get_relative_upload_path("C:\\app\\backend\\uploads\\original\\image123.jpg")
        -> "original/image123.jpg"

    Raises ValueError if the path is outside the upload directory or invalid.
    """
    if not absolute_path:
        raise ValueError("Invalid path: path cannot be empty")

    upload_root = Path(base_dir or settings.UPLOAD_DIR).resolve()
    resolved_path = Path(absolute_path).resolve()

    try:
        rel_path = resolved_path.relative_to(upload_root)
    except ValueError:
        raise ValueError(
            f"Path '{absolute_path}' is outside upload directory '{upload_root}'"
        )

    if rel_path == Path("."):
        raise ValueError(
            f"Path cannot be the upload root directory itself: '{absolute_path}'"
        )

    return rel_path.as_posix()


def resolve_upload_path(
    stored_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Resolves a stored image path to an absolute filesystem path inside settings.UPLOAD_DIR.

    Supports:
    1. Portable relative paths: e.g. "original/image123.jpg"
    2. Legacy Windows absolute paths: e.g. "C:\\...\\backend\\uploads\\original\\image123.jpg"
    3. Legacy POSIX absolute paths: e.g. "/app/backend/uploads/original/image123.jpg"

    Rejects:
    - Directory traversal attempts (e.g. "../", "..\\")
    - Absolute paths outside the configured uploads directory
    - Paths that resolve outside settings.UPLOAD_DIR

    Returns:
        Path: Absolute resolved filesystem path inside UPLOAD_DIR.
    """
    if not stored_path or not str(stored_path).strip():
        raise ValueError("Invalid path: stored path cannot be empty")

    stored_str = str(stored_path).strip()
    upload_root = Path(base_dir or settings.UPLOAD_DIR).resolve()
    upload_name = upload_root.name.lower()

    # Reject traversal attempts explicitly
    normalized_for_check = stored_str.replace("\\", "/")
    parts_check = normalized_for_check.split("/")
    if ".." in parts_check:
        raise ValueError(f"Directory traversal detected in path: '{stored_path}'")

    # Check for Windows vs POSIX absolute path representations
    pw = PureWindowsPath(stored_str)
    pp = PurePosixPath(stored_str)

    is_win_abs = pw.is_absolute() or bool(pw.drive)
    is_posix_abs = pp.is_absolute() or stored_str.startswith("/")

    if is_win_abs or is_posix_abs:
        parts = pw.parts if is_win_abs else pp.parts

        # Find the last occurrence of the uploads directory component
        idx = -1
        for i, part in enumerate(parts):
            if part.lower() in [upload_name, "uploads"]:
                idx = i

        if idx == -1:
            raise ValueError(
                f"Absolute path does not belong to configured uploads directory: '{stored_path}'"
            )

        sub_parts = parts[idx + 1:]
        if not sub_parts:
            raise ValueError(
                f"Path does not point to a file inside uploads directory: '{stored_path}'"
            )

        rel_path = Path(*sub_parts)
    else:
        # Relative path: check if it starts with 'uploads' or upload_name prefix
        clean_parts = [p for p in parts_check if p]
        if clean_parts and clean_parts[0].lower() in [upload_name, "uploads"]:
            sub_parts = clean_parts[1:]
            if not sub_parts:
                raise ValueError(
                    f"Path does not point to a file inside uploads directory: '{stored_path}'"
                )
            rel_path = Path(*sub_parts)
        else:
            rel_path = Path(*clean_parts)

    resolved = (upload_root / rel_path).resolve()

    # Ensure resolved path is strictly contained within upload_root and not the root itself
    if not resolved.is_relative_to(upload_root) or resolved == upload_root:
        raise ValueError(
            f"Resolved path '{resolved}' escapes configured upload directory '{upload_root}'"
        )

    return resolved

