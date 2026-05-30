"""Unit tests for core RAG components."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import precision_at_k, reciprocal_rank
from src.generation import _chunk_satisfies_groups, _mock_generate
from src.hallucination import query_terms_missing_from_context
from src.preprocessing import clean_text


class TestPreprocessing(unittest.TestCase):
    def test_clean_html(self):
        raw = "<p>Free WiFi &amp; breakfast</p>"
        out = clean_text(raw)
        self.assertIn("WiFi", out)
        self.assertNotIn("<p>", out)


class TestMetrics(unittest.TestCase):
    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"b", "d"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 5), 0.4)

    def test_reciprocal_rank(self):
        retrieved = ["x", "y", "z"]
        self.assertAlmostEqual(reciprocal_rank(retrieved, {"y"}), 0.5)


class TestGeneration(unittest.TestCase):
    def test_chunk_satisfies_wifi_breakfast(self):
        groups = [["wifi"], ["complimentary breakfast", "complimentary buffet breakfast"]]
        ok = "Free WiFi and complimentary buffet breakfast for all guests."
        bad = "Free WiFi. Breakfast available for additional charge."
        self.assertTrue(_chunk_satisfies_groups(ok, groups))
        self.assertFalse(_chunk_satisfies_groups(bad, groups))

    def test_mock_list_answer(self):
        chunks = [
            {
                "chunk_id": "amen_004_chunk_0",
                "hotel_name": "The Marina Grand",
                "category": "amenities",
                "text": "Free WiFi, complimentary breakfast for all room categories.",
                "score": 0.8,
            }
        ]
        ans = _mock_generate("Which hotels have free WiFi and complimentary breakfast?", chunks)
        self.assertIn("Marina Grand", ans)
        self.assertIn("[amen_004_chunk_0]", ans)


class TestHallucination(unittest.TestCase):
    def test_missing_query_terms(self):
        chunks = [{"text": "Small pets under 10 kg allowed."}]
        missing = query_terms_missing_from_context("Which hotels allow pet tigers?", chunks)
        self.assertIn("tigers", missing)


if __name__ == "__main__":
    unittest.main()
