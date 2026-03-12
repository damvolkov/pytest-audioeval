"""Word/character-level transcription quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from jiwer import cer, process_words, wer


@dataclass(slots=True, frozen=True)
class TextMetrics:
    """WER/CER transcription quality. O(n) via jiwer."""

    wer: float
    cer: float
    substitutions: int
    insertions: int
    deletions: int

    @classmethod
    def compute(cls, reference: str, hypothesis: str) -> Self:
        """Compute all text metrics in a single pass."""
        output = process_words(reference, hypothesis)
        return cls(
            wer=wer(reference, hypothesis),
            cer=cer(reference, hypothesis),
            substitutions=output.substitutions,
            insertions=output.insertions,
            deletions=output.deletions,
        )

    def assert_quality(self, *, max_wer: float = 0.2, max_cer: float = 0.15) -> None:
        """Raise AssertionError with detailed breakdown on failure."""
        violations: list[str] = []
        if self.wer > max_wer:
            violations.append(f"WER {self.wer:.3f} > {max_wer}")
        if self.cer > max_cer:
            violations.append(f"CER {self.cer:.3f} > {max_cer}")
        if violations:
            msg = (
                f"Audio quality assertion failed: {', '.join(violations)} | "
                f"subs={self.substitutions} ins={self.insertions} del={self.deletions}"
            )
            raise AssertionError(msg)
