"""Unit tests for DiagnosticService: System Health & Hardware Diagnostics."""

import unittest
import asyncio
from backend.services.diagnostic_service import DiagnosticService


class TestDiagnosticService(unittest.TestCase):

    def test_get_system_diagnostic(self):
        """Verify diagnostic service returns hardware acceleration, memory, and storage metrics."""
        res = asyncio.run(DiagnosticService.get_system_diagnostic())
        self.assertEqual(res["status"], "HEALTHY")
        self.assertIn("acceleration", res)
        self.assertIn("memory", res)
        self.assertIn("local_llm", res)
        self.assertIn("storage", res)
        self.assertIn(res["memory"]["hardware_tier"], ["8GB", "16GB", "32GB+"])


if __name__ == "__main__":
    unittest.main()
