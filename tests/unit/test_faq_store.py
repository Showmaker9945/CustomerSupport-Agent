"""
Unit tests for FAQ store.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.knowledge.faq_store import FAQStore, FAQResult, create_faq_store


class TestFAQResult:
    """Test FAQResult dataclass."""

    def test_create_result(self):
        """Test creating an FAQResult."""
        result = FAQResult(
            question="Test question?",
            answer="Test answer",
            category="test",
            confidence=0.95,
            metadata={"keywords": ["test"]}
        )

        assert result.question == "Test question?"
        assert result.answer == "Test answer"
        assert result.confidence == 0.95
        assert result.metadata["keywords"] == ["test"]

    def test_str_representation(self):
        """Test string representation."""
        result = FAQResult(
            question="How do I reset?",
            answer="Click reset button",
            category="account",
            confidence=0.85
        )

        str_result = str(result)
        assert "How do I reset?" in str_result
        assert "account" in str_result
        assert "85.00%" in str_result


class TestFAQStore:
    """Test FAQStore class."""

    @pytest.fixture
    def temp_faq_store(self, tmp_path):
        """Create a temporary FAQ store."""
        return FAQStore(chroma_path=tmp_path / "chroma", collection_name="test_faqs")

    def test_init_with_samples(self, temp_faq_store):
        """Test initialization loads sample FAQs."""
        # Clear first to test loading
        temp_faq_store.clear_all()
        temp_faq_store._load_sample_faqs()

        stats = temp_faq_store.get_stats()
        assert stats["total_faqs"] > 0
        assert len(stats["categories"]) > 0

    def test_add_faq(self, temp_faq_store):
        """Test adding a single FAQ."""
        # Clear samples first
        temp_faq_store.clear_all()

        faq_id = temp_faq_store.add_faq(
            question="Test question?",
            answer="Test answer",
            category="test",
            metadata={"priority": "high"}
        )

        assert faq_id is not None
        stats = temp_faq_store.get_stats()
        assert stats["total_faqs"] == 1

    def test_search(self, temp_faq_store):
        """Test searching FAQs."""
        results = temp_faq_store.search("password reset", top_k=3)

        assert len(results) > 0
        assert isinstance(results[0], FAQResult)
        assert results[0].confidence > 0
        # Most relevant result should be about password
        assert "password" in results[0].question.lower() or \
               "password" in results[0].answer.lower()

    def test_search_with_category_filter(self, temp_faq_store):
        """Test searching with category filter."""
        results = temp_faq_store.search(
            "how do I",
            category="billing",
            top_k=5
        )

        assert len(results) > 0
        for result in results:
            assert result.category == "billing"

    def test_search_min_confidence(self, temp_faq_store):
        """Test search with minimum confidence threshold."""
        results = temp_faq_store.search(
            "gibberish xyz123",
            top_k=5,
            min_confidence=0.5
        )

        # Should return fewer or no results for gibberish
        for result in results:
            assert result.confidence >= 0.5

    def test_get_categories(self, temp_faq_store):
        """Test getting all categories."""
        categories = temp_faq_store.get_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0
        # Check for expected categories from samples
        expected_categories = ["billing", "account", "support", "technical"]
        for cat in expected_categories:
            assert cat in categories

    def test_get_stats(self, temp_faq_store):
        """Test getting FAQ statistics."""
        stats = temp_faq_store.get_stats()

        assert "total_faqs" in stats
        assert "categories" in stats
        assert "category_count" in stats
        assert "embedding_model" in stats
        assert stats["total_faqs"] > 0

    def test_delete_faq(self, temp_faq_store):
        """Test deleting a FAQ."""
        # Clear samples first for clean test
        temp_faq_store.clear_all()

        # Add a specific FAQ
        temp_faq_store.add_faq(
            question="Unique question for deletion xyz123",
            answer="Answer to be deleted",
            category="test"
        )

        # Verify it exists
        results = temp_faq_store.search("Unique question for deletion xyz123")
        assert len(results) > 0

        # Delete it
        deleted = temp_faq_store.delete_faq("Unique question for deletion xyz123")
        assert deleted is True

        # Verify it's gone
        results = temp_faq_store.search("Unique question for deletion xyz123")
        assert len(results) == 0

    def test_clear_all(self, temp_faq_store):
        """Test clearing all FAQs."""
        temp_faq_store.clear_all()

        stats = temp_faq_store.get_stats()
        assert stats["total_faqs"] == 0

    def test_load_faqs_from_json(self, temp_faq_store):
        """Test loading FAQs from JSON file."""
        # Create test JSON file
        test_faqs = [
            {
                "question": "JSON test question?",
                "answer": "JSON test answer",
                "category": "json_test",
                "metadata": {"keywords": ["json", "test"]}
            },
            {
                "question": "Another JSON question?",
                "answer": "Another JSON answer",
                "category": "json_test"
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_faqs, f)
            temp_file = f.name

        try:
            # Clear first
            temp_faq_store.clear_all()

            # Load from file
            count = temp_faq_store.load_faqs_from_file(temp_file)
            assert count == 2

            # Verify search works
            results = temp_faq_store.search("JSON test")
            assert len(results) > 0

        finally:
            Path(temp_file).unlink()

    def test_load_faqs_from_csv(self, temp_faq_store):
        """Test loading FAQs from CSV file."""
        # Create test CSV file
        csv_content = """question,answer,category,keywords
CSV test question?,CSV test answer,csv_test,"csv, test"
Another CSV?,Another CSV answer,csv_test,csv"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name

        try:
            # Clear first
            temp_faq_store.clear_all()

            # Load from file
            count = temp_faq_store.load_faqs_from_file(temp_file)
            assert count == 2

            # Verify search works
            results = temp_faq_store.search("CSV test")
            assert len(results) > 0

        finally:
            Path(temp_file).unlink()

    def test_load_faqs_invalid_file(self, temp_faq_store):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            temp_faq_store.load_faqs_from_file("nonexistent.json")

    def test_load_faqs_unsupported_format(self, temp_faq_store):
        """Test loading from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Not a valid format")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                temp_faq_store.load_faqs_from_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_search_relevance_ranking(self, temp_faq_store):
        """Test that search results are ranked by relevance."""
        results = temp_faq_store.search("payment", top_k=5)

        # Check that results are sorted by confidence
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_various_queries(self, temp_faq_store):
        """Test various search queries."""
        test_queries = [
            ("how do I reset my password", "password"),
            ("payment methods", "payment"),
            ("cancel subscription", "cancel"),
            ("add team member", "team"),
            ("api access", "api"),
            ("contact support", "support")
        ]

        for query, expected_keyword in test_queries:
            results = temp_faq_store.search(query, top_k=2)
            assert len(results) > 0, f"No results for query: {query}"
            # At least one result should contain the expected keyword
            assert any(
                expected_keyword in r.question.lower() or
                expected_keyword in r.answer.lower()
                for r in results
            ), f"Expected keyword '{expected_keyword}' not found in results for: {query}"


class TestCreateFAQStore:
    """Test FAQStore factory function."""

    def test_create_with_samples(self, tmp_path):
        """Test creating store with samples."""
        store = create_faq_store(chroma_path=tmp_path / "test1", load_samples=True)
        stats = store.get_stats()
        assert stats["total_faqs"] > 0

    def test_create_without_samples(self, tmp_path):
        """Test creating store without samples."""
        store = create_faq_store(chroma_path=tmp_path / "test2", load_samples=False)
        stats = store.get_stats()
        assert stats["total_faqs"] == 0
