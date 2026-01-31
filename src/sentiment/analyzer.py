"""
Sentiment analysis module for customer support.

Analyzes customer messages to detect sentiment, frustration levels,
and recommend appropriate routing.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Literal

from textblob import TextBlob

from ..config import settings

logger = logging.getLogger(__name__)


# Frustration keywords and their weights
FRUSTRATION_KEYWORDS = {
    # High frustration (0.8-1.0)
    "furious": 1.0,
    "enraged": 1.0,
    "hate": 1.0,
    "terrible": 0.9,
    "horrible": 0.9,
    "awful": 0.9,
    "disgusting": 0.9,
    "pathetic": 0.9,
    "useless": 0.9,
    "incompetent": 0.9,
    "idiotic": 0.9,
    "stupid": 0.9,
    "moronic": 0.9,

    # Medium-high frustration (0.6-0.8)
    "angry": 0.8,
    "mad": 0.8,
    "pissed": 0.8,
    "frustrated": 0.8,
    "outraged": 0.8,
    "unacceptable": 0.8,
    "ridiculous": 0.75,
    "absurd": 0.75,
    "ridicule": 0.75,
    "worst": 0.8,
    "never buying": 0.8,
    "cancel my subscription": 0.8,
    "want my money back": 0.8,
    "demanding refund": 0.8,
    "sue": 0.9,

    # Medium frustration (0.4-0.6)
    "annoying": 0.6,
    "annoyed": 0.6,
    "irritated": 0.6,
    "irritating": 0.6,
    "frustrating": 0.6,
    "disappointed": 0.6,
    "upset": 0.6,
    "unhappy": 0.6,
    "not working": 0.5,
    "doesn't work": 0.5,
    "broken": 0.5,
    "waste of time": 0.6,
    "waste of money": 0.7,

    # Low-medium frustration (0.2-0.4)
    "confused": 0.3,
    "unclear": 0.2,
    "difficult": 0.3,
    "complicated": 0.3,
    "problem": 0.3,
    "issue": 0.2,
    "help": 0.1,
    "support": 0.1,
    "please": 0.0,
    "thank": 0.0,
}

# Positive keywords to offset frustration
POSITIVE_KEYWORDS = {
    "great": -0.2,
    "good": -0.1,
    "excellent": -0.3,
    "helpful": -0.2,
    "thanks": -0.1,
    "thank you": -0.2,
    "appreciate": -0.2,
    "resolved": -0.3,
    "working": -0.2,
    "fixed": -0.3,
    "love": -0.3,
    "amazing": -0.3,
    "perfect": -0.3,
}


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single message."""
    polarity: float  # -1 to 1 (negative to positive)
    subjectivity: float  # 0 to 1 (objective to subjective)
    label: Literal["positive", "negative", "neutral"]
    frustration_score: float  # 0 to 1
    keywords: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation for display."""
        return (
            f"Sentiment: {self.label} (polarity: {self.polarity:.2f}, "
            f"frustration: {self.frustration_score:.2f})"
        )


@dataclass
class ConversationSentiment:
    """Sentiment analysis over a conversation."""
    average_polarity: float
    trend: Literal["improving", "stable", "declining"]
    escalation_recommended: bool
    reason: str
    message_count: int
    frustration_peak: float

    def __str__(self) -> str:
        """String representation for display."""
        escalation = "⚠️ ESCALATION RECOMMENDED" if self.escalation_recommended else "✓ No escalation needed"
        return (
            f"Conversation Sentiment:\n"
            f"  Average Polarity: {self.average_polarity:.2f}\n"
            f"  Trend: {self.trend}\n"
            f"  Peak Frustration: {self.frustration_peak:.2f}\n"
            f"  {escalation}\n"
            f"  Reason: {self.reason}"
        )


class SentimentAnalyzer:
    """
    Analyze customer sentiment in support conversations.

    Uses TextBlob for sentiment analysis and custom keyword
    detection for frustration scoring.
    """

    def __init__(
        self,
        frustration_keywords: dict = None,
        positive_keywords: dict = None,
        escalation_threshold: float = None
    ):
        """
        Initialize sentiment analyzer.

        Args:
            frustration_keywords: Custom frustration keyword weights
            positive_keywords: Custom positive keyword weights
            escalation_threshold: Frustration threshold for escalation (0-1)
        """
        self.frustration_keywords = frustration_keywords or FRUSTRATION_KEYWORDS
        self.positive_keywords = positive_keywords or POSITIVE_KEYWORDS
        self.escalation_threshold = escalation_threshold or settings.handoff_threshold

        # Normalize escalation threshold from -1 to 1 range
        if self.escalation_threshold < 0:
            # Convert to positive 0-1 scale
            self.escalation_threshold = abs(self.escalation_threshold)

        logger.info(f"Initialized SentimentAnalyzer with escalation threshold: {self.escalation_threshold}")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single message.

        Args:
            text: Message text to analyze

        Returns:
            SentimentResult with polarity, label, and frustration score
        """
        try:
            # Use TextBlob for basic sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Determine label
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            # Calculate frustration score
            frustration_score, keywords = self._calculate_frustration(text)

            # Adjust frustration based on polarity
            # If text is very positive, reduce frustration score
            if polarity > 0.3:
                frustration_score = max(0, frustration_score - 0.3)

            return SentimentResult(
                polarity=polarity,
                subjectivity=subjectivity,
                label=label,
                frustration_score=frustration_score,
                keywords=keywords
            )

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            # Return neutral on error
            return SentimentResult(
                polarity=0.0,
                subjectivity=0.0,
                label="neutral",
                frustration_score=0.0,
                keywords=[]
            )

    def _calculate_frustration(self, text: str) -> tuple[float, List[str]]:
        """
        Calculate frustration score from keywords.

        Uses word boundary matching to avoid false matches like "sue" in "issue".

        Args:
            text: Message text

        Returns:
            Tuple of (frustration_score, matched_keywords)
        """
        import re
        text_lower = text.lower()
        total_score = 0.0
        matched_keywords = []

        # Check frustration keywords with word boundaries
        for keyword, weight in self.frustration_keywords.items():
            # Use regex with word boundaries for exact word matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                total_score += weight
                matched_keywords.append(keyword)

        # Apply positive keywords as offsets
        for keyword, offset in self.positive_keywords.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                total_score += offset  # offset is negative
                matched_keywords.append(f"+{keyword}")

        # Check for repeated punctuation (!!!, ???) indicating frustration
        if "!!!" in text or "???" in text:
            total_score += 0.2

        # Check for ALL CAPS words (indicates shouting/anger)
        words = text.split()
        caps_words = [w for w in words if len(w) > 2 and w.isupper() and w.isalpha()]
        if caps_words:
            total_score += min(0.3, len(caps_words) * 0.1)

        # Normalize to 0-1 range
        frustration_score = max(0.0, min(1.0, total_score))

        return frustration_score, matched_keywords

    def analyze_conversation(
        self,
        messages: List[str]
    ) -> ConversationSentiment:
        """
        Analyze sentiment trend over conversation.

        Args:
            messages: List of message texts in chronological order

        Returns:
            ConversationSentiment with trend analysis
        """
        if not messages:
            return ConversationSentiment(
                average_polarity=0.0,
                trend="stable",
                escalation_recommended=False,
                reason="No messages to analyze",
                message_count=0,
                frustration_peak=0.0
            )

        # Analyze each message
        results = [self.analyze(msg) for msg in messages]

        # Calculate average polarity
        avg_polarity = sum(r.polarity for r in results) / len(results)

        # Calculate trend
        trend = self._calculate_trend(results)

        # Find peak frustration
        frustration_peak = max(r.frustration_score for r in results) if results else 0.0

        # Determine escalation
        escalation, reason = self._should_escalate_conversation(results, trend)

        return ConversationSentiment(
            average_polarity=avg_polarity,
            trend=trend,
            escalation_recommended=escalation,
            reason=reason,
            message_count=len(messages),
            frustration_peak=frustration_peak
        )

    def _calculate_trend(self, results: List[SentimentResult]) -> str:
        """
        Calculate sentiment trend over time.

        Args:
            results: List of sentiment results in chronological order

        Returns:
            Trend label: improving, stable, or declining
        """
        if len(results) < 2:
            return "stable"

        # Split into first half and second half
        mid = len(results) // 2
        first_half = results[:mid]
        second_half = results[mid:]

        # Calculate average polarity for each half
        first_avg = sum(r.polarity for r in first_half) / len(first_half) if first_half else 0
        second_avg = sum(r.polarity for r in second_half) / len(second_half) if second_half else 0

        # Calculate difference
        diff = second_avg - first_avg

        # Determine trend
        if diff > 0.15:
            return "improving"
        elif diff < -0.15:
            return "declining"
        else:
            return "stable"

    def _should_escalate_conversation(
        self,
        results: List[SentimentResult],
        trend: str
    ) -> tuple[bool, str]:
        """
        Determine if conversation should escalate to human.

        Args:
            results: List of sentiment results
            trend: Conversation trend

        Returns:
            Tuple of (should_escalate, reason)
        """
        if not results:
            return False, "No data"

        # Check for very high frustration in recent messages
        recent_results = results[-3:] if len(results) >= 3 else results
        max_recent_frustration = max(r.frustration_score for r in recent_results)

        if max_recent_frustration >= 0.8:
            return True, f"High frustration detected (score: {max_recent_frustration:.2f})"

        # Check for declining trend with negative sentiment
        if trend == "declining":
            avg_polarity = sum(r.polarity for r in results) / len(results)
            if avg_polarity < -0.2:
                return True, f"Declining sentiment with negative average ({avg_polarity:.2f})"

        # Check for persistent negative sentiment
        negative_count = sum(1 for r in results if r.label == "negative")
        negative_ratio = negative_count / len(results)

        if negative_ratio > 0.6 and len(results) >= 3:
            return True, f"Persistent negative sentiment ({negative_ratio*100:.0f}% of messages)"

        # Check if customer is getting more frustrated over time
        if len(results) >= 3:
            early_frustration = sum(r.frustration_score for r in results[:3]) / 3
            late_frustration = sum(r.frustration_score for r in results[-3:]) / 3

            if late_frustration - early_frustration > 0.3:
                return True, f"Increasing frustration over time (early: {early_frustration:.2f}, recent: {late_frustration:.2f})"

        return False, "Sentiment within acceptable range"

    def should_escalate(self, sentiment_history: List[SentimentResult]) -> bool:
        """
        Determine if conversation should escalate based on sentiment history.

        Args:
            sentiment_history: List of sentiment results from conversation

        Returns:
            True if escalation is recommended
        """
        if not sentiment_history:
            return False

        # Calculate conversation-level analysis
        messages = [f"Message {i}" for i in range(len(sentiment_history))]
        # We need to reconstruct or use the sentiment results directly

        # Use the last few messages to determine recent state
        recent = sentiment_history[-3:] if len(sentiment_history) >= 3 else sentiment_history

        # Check for high frustration
        if any(r.frustration_score >= 0.7 for r in recent):
            return True

        # Check for negative sentiment streak
        if len(recent) >= 2 and all(r.label == "negative" for r in recent):
            return True

        return False

    def get_routing_suggestion(self, sentiment: SentimentResult) -> dict:
        """
        Get routing suggestion based on sentiment.

        Args:
            sentiment: Sentiment analysis result

        Returns:
            Dictionary with routing recommendation
        """
        if sentiment.frustration_score >= 0.7:
            return {
                "route": "human",
                "priority": "high",
                "reason": f"High frustration detected (score: {sentiment.frustration_score:.2f})",
                "suggested_action": "escalate immediately"
            }
        elif sentiment.frustration_score >= 0.5:
            return {
                "route": "senior_agent",
                "priority": "medium",
                "reason": f"Moderate frustration (score: {sentiment.frustration_score:.2f})",
                "suggested_action": "handle with care, consider escalation if continues"
            }
        elif sentiment.label == "negative":
            return {
                "route": "ai_with_supervision",
                "priority": "normal",
                "reason": "Negative sentiment detected",
                "suggested_action": "monitor conversation, empathize with customer"
            }
        elif sentiment.label == "positive":
            return {
                "route": "ai",
                "priority": "low",
                "reason": "Positive sentiment",
                "suggested_action": "continue standard support"
            }
        else:
            return {
                "route": "ai",
                "priority": "normal",
                "reason": "Neutral sentiment",
                "suggested_action": "continue standard support"
            }


# Global sentiment analyzer instance
_sentiment_analyzer: SentimentAnalyzer = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create global sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


def reset_sentiment_analyzer() -> None:
    """Reset the global sentiment analyzer instance (useful for testing)."""
    global _sentiment_analyzer
    _sentiment_analyzer = None


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    """Demonstrate sentiment analyzer usage."""
    print("=" * 60)
    print("Sentiment Analyzer Demo")
    print("=" * 60)

    analyzer = SentimentAnalyzer()

    # Test different messages
    test_messages = [
        "I love your product! It's amazing!",
        "This is terrible and I'm very frustrated!",
        "I need help with my account settings.",
        "THIS IS UNACCEPTABLE! I want a refund NOW!"
    ]

    for msg in test_messages:
        result = analyzer.analyze(msg)
        routing = analyzer.get_routing_suggestion(result)
        print(f"\nMessage: {msg}")
        print(f"  Sentiment: {result.label}")
        print(f"  Polarity: {result.polarity:.2f}")
        print(f"  Frustration: {result.frustration_score:.2f}")
        print(f"  Route: {routing['route']}")

    # Test conversation analysis
    print("\n" + "=" * 60)
    print("Conversation Analysis Demo")
    print("=" * 60)

    conversation = [
        "Hi, I'm having trouble with my account.",
        "Sure, I can help with that. What's the issue?",
        "I can't login. It keeps saying invalid credentials.",
        "Have you tried resetting your password?",
        "Yes, I did that three times already! This is ridiculous!",
        "I understand your frustration. Let me check on that.",
        "Well? This is taking forever! I'm about to cancel!"
    ]

    conv_result = analyzer.analyze_conversation(conversation)
    print(f"\nConversation of {conv_result.message_count} messages")
    print(f"  Trend: {conv_result.trend}")
    print(f"  Escalation Recommended: {conv_result.escalation_recommended}")
    print(f"  Reason: {conv_result.reason}")

    print("\n" + "=" * 60)
