"""
Unit tests for sentiment analyzer.
"""

import pytest

from src.sentiment.analyzer import (
    SentimentAnalyzer,
    SentimentResult,
    ConversationSentiment,
    FRUSTRATION_KEYWORDS,
    get_sentiment_analyzer
)


class TestSentimentResult:
    """Test SentimentResult dataclass."""

    def test_create_result(self):
        """Test creating a sentiment result."""
        result = SentimentResult(
            polarity=0.5,
            subjectivity=0.7,
            label="positive",
            frustration_score=0.1,
            keywords=["great", "helpful"]
        )

        assert result.polarity == 0.5
        assert result.label == "positive"
        assert result.frustration_score == 0.1
        assert "great" in result.keywords

    def test_str_representation(self):
        """Test string representation."""
        result = SentimentResult(
            polarity=-0.6,
            subjectivity=0.8,
            label="negative",
            frustration_score=0.7,
            keywords=["angry", "terrible"]
        )

        str_result = str(result)
        assert "negative" in str_result
        assert "0.70" in str_result


class TestSentimentAnalyzer:
    """Test SentimentAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a sentiment analyzer for testing."""
        return SentimentAnalyzer()

    def test_init_default(self):
        """Test initialization with defaults."""
        analyzer = SentimentAnalyzer()
        assert analyzer.frustration_keywords is not None
        assert analyzer.positive_keywords is not None
        assert analyzer.escalation_threshold > 0

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        custom_frustration = {"test": 0.5}
        custom_positive = {"good": -0.1}

        analyzer = SentimentAnalyzer(
            frustration_keywords=custom_frustration,
            positive_keywords=custom_positive,
            escalation_threshold=0.8
        )

        assert analyzer.frustration_keywords == custom_frustration
        assert analyzer.positive_keywords == custom_positive
        assert analyzer.escalation_threshold == 0.8

    def test_analyze_positive(self, analyzer):
        """Test analyzing positive sentiment."""
        result = analyzer.analyze("I love your product! It's amazing and works perfectly.")

        assert result.label == "positive"
        assert result.polarity > 0
        assert result.frustration_score < 0.3

    def test_analyze_negative(self, analyzer):
        """Test analyzing negative sentiment."""
        result = analyzer.analyze("This is terrible and I hate it. Very disappointed.")

        assert result.label == "negative"
        assert result.polarity < 0
        assert result.frustration_score > 0.5

    def test_analyze_neutral(self, analyzer):
        """Test analyzing neutral sentiment."""
        result = analyzer.analyze("I need help with my account settings.")

        assert result.label == "neutral"
        assert -0.2 <= result.polarity <= 0.2

    def test_analyze_frustration_keywords(self, analyzer):
        """Test frustration keyword detection."""
        result = analyzer.analyze("I am frustrated and this is annoying!")

        assert result.frustration_score > 0.4
        assert len(result.keywords) > 0

    def test_analyze_high_frustration(self, analyzer):
        """Test high frustration detection."""
        result = analyzer.analyze("This is absolutely ridiculous and unacceptable! I'm furious!")

        assert result.frustration_score > 0.7
        assert result.label == "negative"

    def test_analyze_with_positive_keywords(self, analyzer):
        """Test that positive keywords offset frustration."""
        result = analyzer.analyze("I had an issue but it was resolved. Thanks for your help!")

        assert result.frustration_score < 0.3
        assert "resolved" in " ".join(result.keywords) or "thanks" in " ".join(result.keywords).lower()

    def test_analyze_caps_and_punctuation(self, analyzer):
        """Test detection of CAPS and repeated punctuation."""
        result = analyzer.analyze("This is NOT working!!! I'm very ANGRY!!!")

        # CAPS and !!! should increase frustration
        assert result.frustration_score > 0.2

    def test_analyze_empty_string(self, analyzer):
        """Test analyzing empty string."""
        result = analyzer.analyze("")

        assert result.label == "neutral"
        assert result.polarity == 0.0
        assert result.frustration_score == 0.0

    def test_analyze_conversation_empty(self, analyzer):
        """Test analyzing empty conversation."""
        result = analyzer.analyze_conversation([])

        assert result.message_count == 0
        assert result.trend == "stable"
        assert result.escalation_recommended is False

    def test_analyze_conversation_positive(self, analyzer):
        """Test analyzing conversation with positive trend."""
        messages = [
            "I'm having an issue.",
            "Thanks for looking into it.",
            "Great! It's working now. You're amazing!"
        ]

        result = analyzer.analyze_conversation(messages)

        assert result.message_count == 3
        assert result.trend in ["improving", "stable"]
        assert result.escalation_recommended is False

    def test_analyze_conversation_negative(self, analyzer):
        """Test analyzing conversation with negative trend."""
        messages = [
            "Hi, I have a question.",
            "This is not working.",
            "This is terrible! I'm frustrated and angry!"
        ]

        result = analyzer.analyze_conversation(messages)

        assert result.message_count == 3
        assert result.trend == "declining"
        # Should likely recommend escalation
        assert result.escalation_recommended is True or result.frustration_peak > 0.5

    def test_analyze_conversation_stable(self, analyzer):
        """Test analyzing stable conversation."""
        messages = [
            "I need help with my order.",
            "Sure, what's the order number?",
            "It's #12345.",
            "Thanks, checking that now."
        ]

        result = analyzer.analyze_conversation(messages)

        assert result.message_count == 4
        assert result.trend == "stable"

    def test_should_escalate_high_frustration(self, analyzer):
        """Test escalation with high frustration."""
        results = [
            analyzer.analyze("This is frustrating!")
        ]

        assert analyzer.should_escalate(results) is False  # Only one message

        results.append(analyzer.analyze("I'm very angry and this is unacceptable!"))

        # Now with high frustration
        assert analyzer.should_escalate(results) is True

    def test_should_escalate_negative_streak(self, analyzer):
        """Test escalation with negative sentiment streak."""
        results = [
            analyzer.analyze("This is terrible."),
            analyzer.analyze("I'm very disappointed."),
            analyzer.analyze("Worst service ever!")
        ]

        assert analyzer.should_escalate(results) is True

    def test_should_escalate_no_escalation(self, analyzer):
        """Test no escalation needed."""
        results = [
            analyzer.analyze("I need help with my account."),
            analyzer.analyze("Thanks for the information."),
            analyzer.analyze("That works, thank you!")
        ]

        assert analyzer.should_escalate(results) is False

    def test_get_routing_suggestion_high_frustration(self, analyzer):
        """Test routing suggestion for high frustration."""
        sentiment = SentimentResult(
            polarity=-0.8,
            subjectivity=0.9,
            label="negative",
            frustration_score=0.8,
            keywords=["furious", "terrible"]
        )

        suggestion = analyzer.get_routing_suggestion(sentiment)

        assert suggestion["route"] == "human"
        assert suggestion["priority"] == "high"
        assert "escalate" in suggestion["suggested_action"].lower()

    def test_get_routing_suggestion_moderate_frustration(self, analyzer):
        """Test routing suggestion for moderate frustration."""
        sentiment = SentimentResult(
            polarity=-0.3,
            subjectivity=0.6,
            label="negative",
            frustration_score=0.5,
            keywords=["annoying"]
        )

        suggestion = analyzer.get_routing_suggestion(sentiment)

        assert suggestion["route"] == "senior_agent"
        assert suggestion["priority"] == "medium"

    def test_get_routing_suggestion_positive(self, analyzer):
        """Test routing suggestion for positive sentiment."""
        sentiment = SentimentResult(
            polarity=0.7,
            subjectivity=0.5,
            label="positive",
            frustration_score=0.0,
            keywords=["great"]
        )

        suggestion = analyzer.get_routing_suggestion(sentiment)

        assert suggestion["route"] == "ai"
        assert suggestion["priority"] == "low"

    def test_get_routing_suggestion_neutral(self, analyzer):
        """Test routing suggestion for neutral sentiment."""
        sentiment = SentimentResult(
            polarity=0.0,
            subjectivity=0.3,
            label="neutral",
            frustration_score=0.1,
            keywords=[]
        )

        suggestion = analyzer.get_routing_suggestion(sentiment)

        assert suggestion["route"] == "ai"
        assert suggestion["priority"] == "normal"

    def test_conversation_sentiment_str(self, analyzer):
        """Test ConversationSentiment string representation."""
        conv = ConversationSentiment(
            average_polarity=-0.3,
            trend="declining",
            escalation_recommended=True,
            reason="High frustration detected",
            message_count=5,
            frustration_peak=0.8
        )

        str_result = str(conv)
        assert "declining" in str_result
        assert "ESCALATION RECOMMENDED" in str_result

    def test_frustration_keywords_exist(self):
        """Test that frustration keywords are defined."""
        assert len(FRUSTRATION_KEYWORDS) > 0
        assert "furious" in FRUSTRATION_KEYWORDS
        assert "angry" in FRUSTRATION_KEYWORDS
        assert "frustrated" in FRUSTRATION_KEYWORDS

    def test_frustration_keywords_weights(self):
        """Test that frustration keywords have appropriate weights."""
        # High frustration keywords should have high weights
        assert FRUSTRATION_KEYWORDS["furious"] >= 0.9
        assert FRUSTRATION_KEYWORDS["angry"] >= 0.7

        # Lower frustration keywords should have lower weights
        assert FRUSTRATION_KEYWORDS.get("confused", 0) < 0.5


class TestGlobalAnalyzer:
    """Test global sentiment analyzer instance."""

    def test_get_sentiment_analyzer_singleton(self):
        """Test that get_sentiment_analyzer returns singleton."""
        analyzer1 = get_sentiment_analyzer()
        analyzer2 = get_sentiment_analyzer()

        assert analyzer1 is analyzer2

    def test_global_analyzer_works(self):
        """Test that global analyzer can analyze text."""
        analyzer = get_sentiment_analyzer()
        result = analyzer.analyze("This is a test message")

        assert result is not None
        assert hasattr(result, "polarity")
        assert hasattr(result, "label")


class TestRealWorldExamples:
    """Test with real-world customer support examples."""

    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer()

    def test_happy_customer(self, analyzer):
        """Test happy customer message."""
        text = "Wow! This is amazing! Your support team is incredible. Thank you so much!"
        result = analyzer.analyze(text)

        assert result.label == "positive"
        assert result.polarity > 0.5
        assert result.frustration_score < 0.2

    def test_confused_customer(self, analyzer):
        """Test confused customer message."""
        text = "I'm confused about how to set up my account. Can you help me understand?"
        result = analyzer.analyze(text)

        assert result.label in ["neutral", "negative"]
        assert result.frustration_score < 0.5

    def test_frustrated_customer(self, analyzer):
        """Test frustrated customer message."""
        text = "I've been trying to fix this for hours! This is so frustrating and annoying. Nothing works!"
        result = analyzer.analyze(text)

        assert result.label == "negative"
        assert result.frustration_score > 0.5
        assert "frustrating" in result.keywords or "annoying" in result.keywords

    def test_very_angry_customer(self, analyzer):
        """Test very angry customer message."""
        text = "THIS IS UNACCEPTABLE! I want my money back NOW! This is the worst service I've ever seen!"
        result = analyzer.analyze(text)

        assert result.label == "negative"
        assert result.frustration_score > 0.7

    def test_escalation_scenario(self, analyzer):
        """Test realistic escalation scenario."""
        conversation = [
            "Hi, I'm having trouble with my account.",
            "Sure, I can help with that. What's the issue?",
            "I can't login. It keeps saying invalid credentials.",
            "Have you tried resetting your password?",
            "Yes, I did that three times already! This is ridiculous!",
            "I understand your frustration. Let me check on that.",
            "Well? This is taking forever! I'm about to cancel my subscription!"
        ]

        result = analyzer.analyze_conversation(conversation)

        assert result.message_count == 7
        assert result.trend == "declining"
        assert result.escalation_recommended is True
        assert "frustration" in result.reason.lower() or "declining" in result.reason.lower()

    def test_resolved_issue_scenario(self, analyzer):
        """Test scenario where issue gets resolved."""
        conversation = [
            "My payment isn't going through.",
            "I'm sorry to hear that. Let me check.",
            "Okay, what's the status?",
            "I see the issue. It should work now. Try again.",
            "Great! It worked. Thank you so much for your help!"
        ]

        result = analyzer.analyze_conversation(conversation)

        assert result.message_count == 5
        assert result.trend == "improving"
        assert result.escalation_recommended is False
