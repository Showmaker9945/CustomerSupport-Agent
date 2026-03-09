"""
Chinese-first sentiment analysis for the customer support agent.

The analyzer is optimized for Chinese customer-service language while still
keeping reasonable English compatibility for mixed-language prompts and tests.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from ..config import settings

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover - dependency fallback
    TextBlob = None

logger = logging.getLogger(__name__)


FRUSTRATION_KEYWORDS: Dict[str, float] = {
    # Chinese high frustration
    "气死了": 1.0,
    "离谱": 0.92,
    "太坑了": 0.95,
    "无法接受": 0.95,
    "我要投诉": 0.95,
    "投诉": 0.82,
    "太差了": 0.9,
    "糟糕": 0.82,
    "太慢了": 0.72,
    "一直没解决": 0.92,
    "还没解决": 0.82,
    "根本没用": 0.88,
    "有毛病": 0.82,
    "报错": 0.55,
    "失败": 0.55,
    "异常": 0.45,
    "不行": 0.45,
    "不对": 0.35,
    "麻烦": 0.18,
    "为什么会这样": 0.35,
    "影响使用": 0.48,
    "退款": 0.8,
    "转人工": 0.7,
    "人工处理": 0.62,
    # English compatibility
    "furious": 1.0,
    "enraged": 1.0,
    "hate": 0.92,
    "terrible": 0.9,
    "horrible": 0.9,
    "awful": 0.9,
    "disgusting": 0.88,
    "useless": 0.88,
    "angry": 0.8,
    "frustrated": 0.8,
    "unacceptable": 0.82,
    "ridiculous": 0.78,
    "worst": 0.82,
    "cancel my subscription": 0.78,
    "want my money back": 0.86,
    "sue": 0.9,
    "annoying": 0.6,
    "annoyed": 0.6,
    "disappointed": 0.6,
    "upset": 0.55,
    "not working": 0.5,
    "doesn't work": 0.5,
    "broken": 0.5,
    "problem": 0.25,
    "issue": 0.18,
}

POSITIVE_KEYWORDS: Dict[str, float] = {
    # Chinese
    "谢谢": -0.2,
    "感谢": -0.2,
    "辛苦了": -0.15,
    "明白了": -0.14,
    "可以了": -0.22,
    "没事了": -0.22,
    "已经解决": -0.32,
    "解决了": -0.3,
    "搞定了": -0.3,
    "很好": -0.18,
    "太好了": -0.25,
    "满意": -0.2,
    # English
    "great": -0.2,
    "good": -0.1,
    "excellent": -0.28,
    "helpful": -0.2,
    "thanks": -0.12,
    "thank you": -0.2,
    "appreciate": -0.18,
    "resolved": -0.3,
    "working": -0.15,
    "fixed": -0.28,
    "love": -0.3,
    "amazing": -0.3,
    "perfect": -0.3,
}

INTENSIFIERS: Dict[str, float] = {
    "非常": 0.1,
    "特别": 0.08,
    "真的": 0.05,
    "太": 0.08,
    "一直": 0.08,
    "根本": 0.12,
    "完全": 0.1,
    "立刻": 0.08,
    "马上": 0.08,
    "超": 0.06,
    "so": 0.06,
    "very": 0.06,
    "extremely": 0.1,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _is_ascii_phrase(text: str) -> bool:
    return bool(text) and all(ord(char) < 128 for char in text)


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single message."""

    polarity: float
    subjectivity: float
    label: Literal["positive", "negative", "neutral"]
    frustration_score: float
    keywords: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"情感：{self.label}（极性：{self.polarity:.2f}，"
            f"挫败度：{self.frustration_score:.2f}）"
        )


@dataclass
class ConversationSentiment:
    """Conversation-level sentiment summary."""

    average_polarity: float
    trend: Literal["improving", "stable", "declining"]
    escalation_recommended: bool
    reason: str
    message_count: int
    frustration_peak: float

    def __str__(self) -> str:
        escalation = "建议升级人工" if self.escalation_recommended else "当前无需升级"
        return (
            "对话情绪分析：\n"
            f"  平均极性：{self.average_polarity:.2f}\n"
            f"  趋势：{self.trend}\n"
            f"  峰值挫败度：{self.frustration_peak:.2f}\n"
            f"  结论：{escalation}\n"
            f"  原因：{self.reason}"
        )


class SentimentAnalyzer:
    """Chinese-first customer support sentiment analyzer."""

    def __init__(
        self,
        frustration_keywords: dict | None = None,
        positive_keywords: dict | None = None,
        escalation_threshold: float | None = None,
    ):
        self.frustration_keywords = frustration_keywords or FRUSTRATION_KEYWORDS
        self.positive_keywords = positive_keywords or POSITIVE_KEYWORDS
        self.escalation_threshold = escalation_threshold or settings.handoff_threshold
        if self.escalation_threshold < 0:
            self.escalation_threshold = abs(self.escalation_threshold)
        logger.info(
            "Initialized SentimentAnalyzer with escalation threshold: %s",
            self.escalation_threshold,
        )

    def analyze(self, text: str) -> SentimentResult:
        """Analyze a single user message."""

        cleaned = str(text or "").strip()
        if not cleaned:
            return SentimentResult(
                polarity=0.0,
                subjectivity=0.0,
                label="neutral",
                frustration_score=0.0,
                keywords=[],
            )

        try:
            frustration_score, keywords, keyword_polarity = self._calculate_frustration(cleaned)
            blob_polarity, blob_subjectivity = self._blob_sentiment(cleaned)

            if _contains_chinese(cleaned):
                polarity = (keyword_polarity * 0.85) + (blob_polarity * 0.15)
            else:
                polarity = (keyword_polarity * 0.55) + (blob_polarity * 0.45)

            polarity = _clamp(polarity, -1.0, 1.0)
            subjectivity = _clamp(
                max(blob_subjectivity, min(1.0, 0.18 * len(keywords) + 0.15)),
                0.0,
                1.0,
            )

            if frustration_score >= 0.55 and polarity > -0.2:
                polarity = min(polarity, -0.22)
            if frustration_score >= 0.75:
                polarity = min(polarity, -0.55)
            if (
                any(keyword.startswith("+") for keyword in keywords)
                and frustration_score < 0.15
                and polarity > 0.28
            ):
                polarity = _clamp(polarity + 0.08, -1.0, 1.0)

            if polarity > 0.18 and frustration_score < 0.4:
                label = "positive"
            elif polarity < -0.18 or frustration_score >= 0.45:
                label = "negative"
            else:
                label = "neutral"

            return SentimentResult(
                polarity=polarity,
                subjectivity=subjectivity,
                label=label,
                frustration_score=frustration_score,
                keywords=keywords,
            )
        except Exception as error:  # pragma: no cover - defensive fallback
            logger.error(f"Sentiment analysis error: {error}")
            return SentimentResult(
                polarity=0.0,
                subjectivity=0.0,
                label="neutral",
                frustration_score=0.0,
                keywords=[],
            )

    def _blob_sentiment(self, text: str) -> Tuple[float, float]:
        if TextBlob is None:
            return 0.0, 0.0
        try:
            blob = TextBlob(text)
            return float(blob.sentiment.polarity), float(blob.sentiment.subjectivity)
        except Exception:
            return 0.0, 0.0

    def _keyword_match(self, keyword: str, text: str, lowered: str) -> bool:
        if _is_ascii_phrase(keyword):
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            return re.search(pattern, lowered) is not None
        return keyword in text

    def _calculate_frustration(self, text: str) -> Tuple[float, List[str], float]:
        lowered = text.lower()
        negative_score = 0.0
        positive_score = 0.0
        matched_keywords: List[str] = []

        for keyword, weight in self.frustration_keywords.items():
            if self._keyword_match(keyword, text, lowered):
                negative_score += weight
                matched_keywords.append(keyword)

        for keyword, offset in self.positive_keywords.items():
            if self._keyword_match(keyword, text, lowered):
                positive_score += abs(offset)
                matched_keywords.append(f"+{keyword}")

        intensity_bonus = 0.0
        for keyword, weight in INTENSIFIERS.items():
            if self._keyword_match(keyword, text, lowered):
                intensity_bonus += weight

        punctuation_bonus = 0.0
        repeated_marks = (
            text.count("!") + text.count("！") + text.count("?") + text.count("？")
        )
        if repeated_marks >= 3:
            punctuation_bonus += 0.18
        if "!!!" in text or "！！！" in text or "???" in text or "？？？" in text:
            punctuation_bonus += 0.12

        caps_bonus = 0.0
        caps_words = [
            token for token in re.findall(r"[A-Z]{3,}", text) if token.isalpha()
        ]
        if caps_words:
            caps_bonus = min(0.25, len(caps_words) * 0.08)

        frustration_score = _clamp(
            negative_score + intensity_bonus + punctuation_bonus + caps_bonus - (positive_score * 0.4),
            0.0,
            1.0,
        )
        keyword_polarity = _clamp((positive_score * 0.8) - (negative_score * 0.85), -1.0, 1.0)
        return frustration_score, matched_keywords, keyword_polarity

    def analyze_conversation(self, messages: List[str]) -> ConversationSentiment:
        """Analyze multi-turn conversation sentiment."""

        if not messages:
            return ConversationSentiment(
                average_polarity=0.0,
                trend="stable",
                escalation_recommended=False,
                reason="暂无可分析的对话。",
                message_count=0,
                frustration_peak=0.0,
            )

        results = [self.analyze(message) for message in messages]
        avg_polarity = sum(item.polarity for item in results) / len(results)
        trend = self._calculate_trend(results)
        frustration_peak = max(item.frustration_score for item in results) if results else 0.0
        escalation, reason = self._should_escalate_conversation(results, trend)

        return ConversationSentiment(
            average_polarity=avg_polarity,
            trend=trend,
            escalation_recommended=escalation,
            reason=reason,
            message_count=len(messages),
            frustration_peak=frustration_peak,
        )

    def _calculate_trend(self, results: List[SentimentResult]) -> str:
        if len(results) < 2:
            return "stable"

        mid = len(results) // 2
        first_half = results[:mid]
        second_half = results[mid:]

        first_avg = sum(item.polarity for item in first_half) / len(first_half) if first_half else 0.0
        second_avg = sum(item.polarity for item in second_half) / len(second_half) if second_half else 0.0
        diff = second_avg - first_avg

        if diff > 0.15:
            return "improving"
        if diff < -0.15:
            return "declining"
        return "stable"

    def _should_escalate_conversation(
        self,
        results: List[SentimentResult],
        trend: str,
    ) -> Tuple[bool, str]:
        if not results:
            return False, "暂无数据。"

        recent_results = results[-3:] if len(results) >= 3 else results
        max_recent_frustration = max(item.frustration_score for item in recent_results)
        if max_recent_frustration >= 0.8:
            return True, f"最近消息中检测到高挫败情绪（分数：{max_recent_frustration:.2f}）。"

        if trend == "declining":
            avg_polarity = sum(item.polarity for item in results) / len(results)
            if avg_polarity < -0.2:
                return True, f"对话情绪持续下降，且平均情感偏负面（{avg_polarity:.2f}）。"

        negative_count = sum(1 for item in results if item.label == "negative")
        negative_ratio = negative_count / len(results)
        if negative_ratio > 0.6 and len(results) >= 3:
            return True, f"负向情绪持续出现（{negative_ratio * 100:.0f}% 消息为负向）。"

        if len(results) >= 3:
            early = sum(item.frustration_score for item in results[:3]) / 3
            late = sum(item.frustration_score for item in results[-3:]) / 3
            if late - early > 0.3:
                return True, f"挫败情绪持续升高（前段 {early:.2f}，最近 {late:.2f}）。"

        return False, "情绪整体可控。"

    def should_escalate(self, sentiment_history: List[SentimentResult]) -> bool:
        if not sentiment_history:
            return False
        recent = sentiment_history[-3:] if len(sentiment_history) >= 3 else sentiment_history
        if any(item.frustration_score >= 0.7 for item in recent):
            return True
        if len(recent) >= 2 and all(item.label == "negative" for item in recent):
            return True
        return False

    def get_routing_suggestion(self, sentiment: SentimentResult) -> dict:
        if sentiment.frustration_score >= 0.7:
            return {
                "route": "human",
                "priority": "high",
                "reason": f"检测到高挫败情绪（分数：{sentiment.frustration_score:.2f}）。",
                "suggested_action": "立即转人工并优先安抚用户",
            }
        if sentiment.frustration_score >= 0.5:
            return {
                "route": "senior_agent",
                "priority": "medium",
                "reason": f"用户存在中度挫败情绪（分数：{sentiment.frustration_score:.2f}）。",
                "suggested_action": "由高级客服跟进，并持续观察是否需要升级",
            }
        if sentiment.label == "negative":
            return {
                "route": "ai_with_supervision",
                "priority": "normal",
                "reason": "检测到负向情绪。",
                "suggested_action": "继续提供帮助，同时加强共情表达",
            }
        if sentiment.label == "positive":
            return {
                "route": "ai",
                "priority": "low",
                "reason": "用户情绪积极。",
                "suggested_action": "保持标准支持流程即可",
            }
        return {
            "route": "ai",
            "priority": "normal",
            "reason": "用户情绪平稳。",
            "suggested_action": "按标准客服流程继续处理",
        }


_sentiment_analyzer: SentimentAnalyzer | None = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


def reset_sentiment_analyzer() -> None:
    global _sentiment_analyzer
    _sentiment_analyzer = None


if __name__ == "__main__":  # pragma: no cover - manual demo helper
    analyzer = SentimentAnalyzer()
    samples = [
        "谢谢，已经解决了。",
        "这个账单一直没解决，太离谱了，我要投诉！",
        "I love your product! It's amazing.",
        "THIS IS UNACCEPTABLE! I want my money back now!",
    ]
    for sample in samples:
        result = analyzer.analyze(sample)
        print(sample)
        print(result)
        print(analyzer.get_routing_suggestion(result))
        print("-" * 40)
