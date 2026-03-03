"""Basic logging filter for redacting secrets in logs."""

import logging
import re
from typing import Any


class SensitiveDataFilter(logging.Filter):
    """Redact common API key patterns from log records."""

    _patterns = [
        # Generic assignment style keys
        re.compile(r"(api[_-]?key\s*[:=]\s*)([^\s,;]+)", re.IGNORECASE),
        re.compile(r"(authorization\s*[:=]\s*bearer\s+)([^\s,;]+)", re.IGNORECASE),
        # Common raw key tokens
        re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,})\b"),
        re.compile(r"\b(sk-[A-Za-z0-9]{16,})\b"),
    ]

    @staticmethod
    def _redact(text: str) -> str:
        redacted = text
        for pattern in SensitiveDataFilter._patterns:
            # Preserve prefix group when present.
            if pattern.groups >= 2:
                redacted = pattern.sub(r"\1[REDACTED]", redacted)
            else:
                redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            # Best effort: redact any string args, keep others untouched.
            if isinstance(record.args, tuple):
                record.args = tuple(
                    self._redact(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: (self._redact(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
        return True

