"""Dependency-free regression tests for the merge script's network helpers.

Run from the repository: python3 -m unittest discover -s tests -p
  test_merge_qwen_image21_lora_network.py -v
No network requests, downloads or torch imports are performed.
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_qwen_image21_lora.py"
SPEC = importlib.util.spec_from_file_location("qwen_merge_network_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ENDPOINT = "https://huggingface.co"


class MergeNetworkTests(unittest.TestCase):
    def test_https_endpoint(self):
        self.assertEqual(MODULE.normalize_endpoint(ENDPOINT + "/"), ENDPOINT)
        self.assertEqual(MODULE.normalize_endpoint("https://hf-mirror.com"), "https://hf-mirror.com")

    def test_reject_credentials_and_non_origin_urls(self):
        for value in ("http://huggingface.co", "https://user:secret@hf-mirror.com",
                      "https://hf-mirror.com/api", "https://hf-mirror.com?token=secret",
                      "https://hf-mirror.com#fragment", "https://"):
            with self.subTest(endpoint=value), self.assertRaises(ValueError):
                MODULE.normalize_endpoint(value)

    def test_httpx2_connect_timeout(self):
        error_type = type("ConnectTimeout", (Exception,), {"__module__": "httpx2"})
        self.assertTrue(MODULE.retryable_network_error(error_type("timeout")))

    def test_wrapped_timeout(self):
        outer = RuntimeError("download interrupted")
        outer.__cause__ = TimeoutError("temporary")
        self.assertTrue(MODULE.retryable_network_error(outer))

    def test_transient_status(self):
        for status in (408, 429, 500, 502, 503, 504):
            error = RuntimeError("HTTP error")
            error.response = SimpleNamespace(status_code=status)
            self.assertTrue(MODULE.retryable_network_error(error))

    def test_do_not_retry_auth_or_missing_revision(self):
        for status in (400, 401, 403, 404):
            error = RuntimeError("HTTP error")
            error.response = SimpleNamespace(status_code=status)
            error.__cause__ = TimeoutError("must not override permanent HTTP status")
            self.assertFalse(MODULE.retryable_network_error(error))

    def test_do_not_retry_disk_or_configuration_errors(self):
        for error in (OSError(28, "No space left on device"), ValueError("bad config"),
                      RuntimeError("adapter checksum mismatch")):
            self.assertFalse(MODULE.retryable_network_error(error))

    def test_disk_error_overrides_stale_network_context(self):
        error = OSError(28, "No space left on device")
        error.__context__ = TimeoutError("previous download attempt")
        self.assertFalse(MODULE.retryable_network_error(error))

    def test_certificate_error_not_retried(self):
        error_type = type("ConnectError", (Exception,), {"__module__": "httpx2"})
        error = error_type("CERTIFICATE_VERIFY_FAILED")
        self.assertFalse(MODULE.retryable_network_error(error))

    @patch.object(MODULE.time, "sleep")
    def test_retry_then_success(self, sleep):
        operation = Mock(side_effect=[TimeoutError("timeout"), "OK"])
        result = MODULE.network_call("metadata", operation, 3, ENDPOINT)
        self.assertEqual(result, "OK")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(2)

    @patch.object(MODULE.time, "sleep")
    def test_bounded_attempts(self, sleep):
        operation = Mock(side_effect=TimeoutError("timeout"))
        with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
            MODULE.network_call("metadata", operation, 3, ENDPOINT)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    @patch.object(MODULE.time, "sleep")
    def test_permanent_failure_is_not_retried(self, sleep):
        operation = Mock(side_effect=ValueError("bad config"))
        with self.assertRaises(ValueError):
            MODULE.network_call("metadata", operation, 3, ENDPOINT)
        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
