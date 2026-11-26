import csv
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "yakshagana_work" / "saved_models" / "split.csv"
DATASET_DIR = ROOT / "Yakshagana dataset"


def known_test_image() -> bytes:
    with SPLIT_FILE.open(newline="", encoding="utf-8") as file:
        row = next(
            item
            for item in csv.DictReader(file)
            if item["split"] == "test" and item["vesha"] == "Pundarika veshsa"
        )
    return (DATASET_DIR / row["relpath"]).read_bytes()


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()
        cls.image = known_test_image()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_metrics_and_health_are_available(self):
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})
        metrics = self.client.get("/api/metrics").json()
        self.assertEqual(metrics["counts"]["test"], 103)
        self.assertEqual(len(metrics["classes"]), 6)
        self.assertFalse(metrics["non_yakshagana"]["supported"])
        self.assertTrue(metrics["non_yakshagana"]["screen_available"])
        self.assertIsNone(metrics["non_yakshagana"]["false_acceptance_rate"])
        self.assertEqual(
            metrics["uncertainty_thresholds"]["minimum_vesha_confidence"],
            0.85,
        )
        self.assertEqual(
            len(metrics["confusion_matrix"]["values"][0]),
            len(metrics["confusion_matrix"]["predicted_labels"]),
        )

    def test_prediction_returns_all_class_scores(self):
        response = self.client.post(
            "/api/predict",
            content=self.image,
            headers={"Content-Type": "image/jpeg"},
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn(result["status"], {"yakshagana", "not_yakshagana", "uncertain"})
        self.assertEqual(len(result["class_scores"]), 6)
        self.assertTrue(result["rejection_screen_available"])
        self.assertEqual(result["rejection_method"], "positive_only_novelty_screen")
        self.assertEqual(result["class_scores"][0]["class"], "Pundarika veshsa")
        if result["status"] == "not_yakshagana":
            self.assertIsNone(result["predicted_class"])
            self.assertIsNone(result["candidate_class"])
            self.assertIsNone(result["confidence"])
        elif result["status"] == "uncertain":
            self.assertIsNone(result["predicted_class"])
            self.assertEqual(result["candidate_class"], "Pundarika veshsa")
            self.assertIsNotNone(result["confidence"])
        else:
            self.assertEqual(result["predicted_class"], "Pundarika veshsa")
            self.assertIsNone(result["candidate_class"])
            self.assertIsNotNone(result["confidence"])
        self.assertAlmostEqual(
            sum(score["score"] for score in result["class_scores"]),
            1.0,
            places=5,
        )

    def test_rejects_mismatched_mime_and_invalid_image_content(self):
        mismatch = self.client.post(
            "/api/predict",
            content=self.image,
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(mismatch.status_code, 415)

        invalid = self.client.post(
            "/api/predict",
            content=b"not an image",
            headers={"Content-Type": "image/jpeg"},
        )
        self.assertEqual(invalid.status_code, 400)

    def test_rejects_uploads_over_ten_megabytes(self):
        response = self.client.post(
            "/api/predict",
            content=b"",
            headers={
                "Content-Type": "image/jpeg",
                "Content-Length": str(10 * 1024 * 1024 + 1),
            },
        )
        self.assertEqual(response.status_code, 413)

    def test_fixed_references_are_available(self):
        references = self.client.get("/api/references")
        self.assertEqual(references.status_code, 200)
        self.assertEqual(len(references.json()), 6)
        image = self.client.get(references.json()[0]["url"])
        self.assertEqual(image.status_code, 200)


if __name__ == "__main__":
    unittest.main()
