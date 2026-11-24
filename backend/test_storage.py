"""
Focused Automated Unit Tests for DermAI Storage Module (app/storage.py).

Tests path resolution portability between Windows local development and Linux Docker container deployments:
- Portable relative path resolution
- Legacy absolute Windows path resolution
- Legacy absolute POSIX path resolution
- Rejection of paths outside uploads directory
- Rejection of directory traversal attacks (../)
- Rejection of absolute paths outside uploads directory
- get_relative_upload_path conversion and rejection of outside paths

Uses isolated temporary directories to guarantee no impact on real DermAI database or uploads.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add backend directory to sys.path so app modules import cleanly
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.storage import get_relative_upload_path, resolve_upload_path
from app.config import settings


class TestStoragePathPortability(unittest.TestCase):
    def setUp(self):
        # Create isolated temporary directory representing UPLOAD_DIR
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name).resolve()
        # Create standard subdirectories
        (self.upload_dir / "original").mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "preview").mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "preprocessed").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Relative path resolves correctly
    # -------------------------------------------------------------------------
    def test_relative_path_resolves_correctly(self):
        expected_original = self.upload_dir / "original" / "image123.jpg"
        expected_preview = self.upload_dir / "preview" / "image123_thumb.jpg"
        expected_preprocessed = self.upload_dir / "preprocessed" / "image123_proc.jpg"

        res_orig = resolve_upload_path("original/image123.jpg", base_dir=self.upload_dir)
        res_prev = resolve_upload_path("preview/image123_thumb.jpg", base_dir=self.upload_dir)
        res_proc = resolve_upload_path("preprocessed/image123_proc.jpg", base_dir=self.upload_dir)

        self.assertEqual(res_orig, expected_original)
        self.assertEqual(res_prev, expected_preview)
        self.assertEqual(res_proc, expected_preprocessed)

        # Test Path object input as well
        res_path_obj = resolve_upload_path(Path("original/image123.jpg"), base_dir=self.upload_dir)
        self.assertEqual(res_path_obj, expected_original)

        # Test relative path with uploads prefix (e.g. 'uploads/original/image123.jpg')
        res_with_prefix = resolve_upload_path("uploads/original/image123.jpg", base_dir=self.upload_dir)
        self.assertEqual(res_with_prefix, expected_original)

    # -------------------------------------------------------------------------
    # 2. Legacy Windows absolute path resolves correctly
    # -------------------------------------------------------------------------
    def test_legacy_windows_absolute_path_resolves_correctly(self):
        expected = self.upload_dir / "original" / "image123.jpg"

        # Standard Windows backslash path
        win_path_1 = r"C:\Users\johan\Desktop\springboard\backend\uploads\original\image123.jpg"
        self.assertEqual(resolve_upload_path(win_path_1, base_dir=self.upload_dir), expected)

        # Windows path with forward slashes
        win_path_2 = "C:/Users/johan/Desktop/springboard/backend/uploads/original/image123.jpg"
        self.assertEqual(resolve_upload_path(win_path_2, base_dir=self.upload_dir), expected)

        # Windows path from another drive
        win_path_3 = r"D:\production_app\uploads\original\image123.jpg"
        self.assertEqual(resolve_upload_path(win_path_3, base_dir=self.upload_dir), expected)

        # Windows path in preview subdirectory
        expected_prev = self.upload_dir / "preview" / "thumb.jpg"
        win_path_prev = r"C:\backend\uploads\preview\thumb.jpg"
        self.assertEqual(resolve_upload_path(win_path_prev, base_dir=self.upload_dir), expected_prev)

    # -------------------------------------------------------------------------
    # 3. Legacy POSIX absolute path resolves correctly
    # -------------------------------------------------------------------------
    def test_legacy_posix_absolute_path_resolves_correctly(self):
        expected_orig = self.upload_dir / "original" / "image123.jpg"
        expected_proc = self.upload_dir / "preprocessed" / "proc.jpg"

        # Standard Docker Linux container path
        posix_path_1 = "/app/backend/uploads/original/image123.jpg"
        self.assertEqual(resolve_upload_path(posix_path_1, base_dir=self.upload_dir), expected_orig)

        # Alternative Linux deployment path
        posix_path_2 = "/var/www/dermai/uploads/preprocessed/proc.jpg"
        self.assertEqual(resolve_upload_path(posix_path_2, base_dir=self.upload_dir), expected_proc)

    # -------------------------------------------------------------------------
    # 4. Path outside uploads is rejected
    # -------------------------------------------------------------------------
    def test_path_outside_uploads_rejected(self):
        outside_paths = [
            self.upload_dir.parent / "outside_image.jpg",
            "../outside_dir/image.jpg",
            "../../etc/passwd",
            r"C:\outside_folder\image.jpg",
            r"D:\other_drive\data\image.jpg",
            "/var/data/outside_image.jpg",
            "/etc/shadow",
        ]
        for p in outside_paths:
            with self.subTest(path=p):
                with self.assertRaises(ValueError):
                    resolve_upload_path(p, base_dir=self.upload_dir)

    # -------------------------------------------------------------------------
    # 5. ../ traversal is rejected
    # -------------------------------------------------------------------------
    def test_dot_dot_traversal_rejected(self):
        traversal_attempts = [
            "../image123.jpg",
            "../../etc/passwd",
            "../../../secret.txt",
            "original/../../secret.txt",
            "preview/../../../etc/shadow",
            r"..\Windows\System32\cmd.exe",
            r"original\..\..\secret.txt",
            r"C:\app\uploads\..\..\etc\passwd",
            r"C:\Users\johan\Desktop\springboard\backend\uploads\..\..\secret.txt",
            "/app/backend/uploads/../../etc/passwd",
        ]
        for bad_path in traversal_attempts:
            with self.subTest(traversal_vector=bad_path):
                with self.assertRaises(ValueError):
                    resolve_upload_path(bad_path, base_dir=self.upload_dir)

    # -------------------------------------------------------------------------
    # 6. Absolute path outside uploads is rejected
    # -------------------------------------------------------------------------
    def test_absolute_path_outside_uploads_rejected(self):
        forbidden_absolute_paths = [
            r"C:\Windows\System32\cmd.exe",
            r"C:\Program Files\App\data.txt",
            r"C:\Users\johan\Desktop\springboard\backend\app\main.py",
            r"C:\outside\uploads_fake\image.jpg",
            "/etc/passwd",
            "/var/log/syslog",
            "/home/user/secret.key",
            "/opt/uploads_other/file.jpg",
        ]
        for forbidden in forbidden_absolute_paths:
            with self.subTest(forbidden_path=forbidden):
                with self.assertRaises(ValueError):
                    resolve_upload_path(forbidden, base_dir=self.upload_dir)

    # -------------------------------------------------------------------------
    # 7. get_relative_upload_path rejects files outside uploads
    # -------------------------------------------------------------------------
    def test_get_relative_upload_path_rejects_files_outside_uploads(self):
        outside_files = [
            Path(tempfile.gettempdir()) / "outside_image.jpg",
            self.upload_dir.parent / "secret.txt",
            r"C:\Windows\System32\drivers\etc\hosts",
            "/etc/passwd",
        ]
        for outside_file in outside_files:
            with self.subTest(outside=outside_file):
                with self.assertRaises(ValueError):
                    get_relative_upload_path(outside_file, base_dir=self.upload_dir)

        # Upload root directory itself is not an image file
        with self.assertRaises(ValueError):
            get_relative_upload_path(self.upload_dir, base_dir=self.upload_dir)

        # Empty or None path
        with self.assertRaises(ValueError):
            get_relative_upload_path("", base_dir=self.upload_dir)

    # -------------------------------------------------------------------------
    # 8. get_relative_upload_path converts correctly to POSIX relative path
    # -------------------------------------------------------------------------
    def test_get_relative_upload_path_converts_correctly(self):
        orig_file = self.upload_dir / "original" / "sample_lesion.jpg"
        prev_file = self.upload_dir / "preview" / "sample_preview.jpg"
        proc_file = self.upload_dir / "preprocessed" / "sample_proc.jpg"

        rel_orig = get_relative_upload_path(orig_file, base_dir=self.upload_dir)
        rel_prev = get_relative_upload_path(prev_file, base_dir=self.upload_dir)
        rel_proc = get_relative_upload_path(proc_file, base_dir=self.upload_dir)

        self.assertEqual(rel_orig, "original/sample_lesion.jpg")
        self.assertEqual(rel_prev, "preview/sample_preview.jpg")
        self.assertEqual(rel_proc, "preprocessed/sample_proc.jpg")

        # Verify separator is always forward slash (POSIX)
        self.assertNotIn("\\", rel_orig)
        self.assertNotIn("\\", rel_prev)
        self.assertNotIn("\\", rel_proc)


if __name__ == "__main__":
    unittest.main()
