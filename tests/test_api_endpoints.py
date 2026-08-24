import unittest
from fastapi.testclient import TestClient
from backend.main import app


class TestApiEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("model", data)

    def test_models_discovery_endpoint(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
        # Verify capability schema fields
        first = models[0]
        self.assertIn("model_id", first)
        self.assertIn("supports_vision", first)
        self.assertIn("supports_text", first)
        self.assertIn("capability_mode", first)


if __name__ == "__main__":
    unittest.main()
