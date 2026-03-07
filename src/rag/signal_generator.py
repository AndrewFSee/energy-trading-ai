"""LLM-based qualitative trading signal generator.

Uses the RAG pipeline and current market context to generate qualitative
bullish/bearish signals with confidence scores.  These signals augment
the quantitative model outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.rag.qa_chain import QAChain

logger = logging.getLogger(__name__)

SIGNAL_PROMPT = """You are an energy market trading strategist. Based on the following current market context and retrieved knowledge from energy trading resources, generate a directional trading signal for {instrument}.

Current Market Context:
{market_context}

Retrieved Knowledge:
{rag_context}

Task: Provide a structured trading signal with the following fields:
1. DIRECTION: [BULLISH / BEARISH / NEUTRAL]
2. CONFIDENCE: [0.0 to 1.0 — your confidence in this signal]
3. REASONING: [2-3 sentence explanation citing specific factors]
4. KEY_RISKS: [Main risks that could invalidate this signal]
5. TIME_HORIZON: [INTRADAY / SHORT_TERM (1-5 days) / MEDIUM_TERM (1-4 weeks)]

Format your response EXACTLY as:
DIRECTION: <value>
CONFIDENCE: <value>
REASONING: <value>
KEY_RISKS: <value>
TIME_HORIZON: <value>"""


@dataclass
class TradingSignal:
    """Structured output from the LLM signal generator.

    Attributes:
        instrument: Trading instrument (e.g. ``"WTI Crude Oil"``).
        direction: Signal direction — ``"BULLISH"``, ``"BEARISH"``, or ``"NEUTRAL"``.
        confidence: Confidence score in range [0, 1].
        reasoning: Explanation of the signal rationale.
        key_risks: Risks that could invalidate the signal.
        time_horizon: Expected holding period.
        generated_at: Timestamp when the signal was generated.
        raw_response: Original LLM response text.
    """

    instrument: str
    direction: str
    confidence: float
    reasoning: str
    key_risks: str
    time_horizon: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    raw_response: str = ""

    @property
    def numeric_signal(self) -> float:
        """Convert direction to a numeric signal value.

        Returns:
            ``1.0`` for BULLISH, ``-1.0`` for BEARISH, ``0.0`` for NEUTRAL.
            Scaled by confidence.
        """
        direction_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
        base = direction_map.get(self.direction.upper(), 0.0)
        return base * self.confidence


class LLMSignalGenerator:
    """Generates LLM-based qualitative trading signals using RAG context.

    Combines current quantitative market data with retrieved knowledge
    from the energy trading document store to produce structured signals.

    Attributes:
        qa_chain: RAG QA chain for context retrieval.
        confidence_threshold: Minimum confidence to emit a non-neutral signal.
    """

    def __init__(
        self,
        qa_chain: QAChain,
        confidence_threshold: float = 0.65,
    ) -> None:
        """Initialise the signal generator.

        Args:
            qa_chain: Configured ``QAChain`` instance.
            confidence_threshold: Signals below this confidence level are
                overridden to ``NEUTRAL``.
        """
        self.qa_chain = qa_chain
        self.confidence_threshold = confidence_threshold
        logger.info(
            "LLMSignalGenerator initialised (confidence_threshold=%.2f)",
            confidence_threshold,
        )

    def generate(
        self,
        instrument: str,
        market_context: str,
        rag_query: str | None = None,
    ) -> TradingSignal:
        """Generate a qualitative trading signal.

        Args:
            instrument: Instrument name (e.g. ``"WTI Crude Oil (CL=F)"``).
            market_context: Current market data summary (price, technicals,
                sentiment, fundamentals).
            rag_query: Optional custom RAG query.  Defaults to a general
                market outlook query for the instrument.

        Returns:
            Structured ``TradingSignal`` object.
        """
        if rag_query is None:
            rag_query = f"What are the key factors driving {instrument} prices and what is the current market outlook?"

        # Retrieve context from knowledge base
        rag_result = self.qa_chain.ask(rag_query, return_sources=True)
        rag_context = rag_result.get("answer", "")

        # Construct signal prompt
        prompt = SIGNAL_PROMPT.format(
            instrument=instrument,
            market_context=market_context,
            rag_context=rag_context[:2000],
        )

        logger.info("Generating LLM signal for %s", instrument)
        try:
            raw_response = self.qa_chain.llm.complete(prompt)
            signal = self._parse_signal(raw_response, instrument)
        except Exception as exc:
            logger.error("Signal generation failed: %s", exc)
            signal = TradingSignal(
                instrument=instrument,
                direction="NEUTRAL",
                confidence=0.0,
                reasoning=f"Signal generation error: {exc}",
                key_risks="LLM service unavailable",
                time_horizon="N/A",
            )

        # Apply confidence threshold
        if signal.confidence < self.confidence_threshold and signal.direction != "NEUTRAL":
            logger.info(
                "Signal confidence %.2f below threshold %.2f — overriding to NEUTRAL",
                signal.confidence,
                self.confidence_threshold,
            )
            signal.direction = "NEUTRAL"

        return signal

    def _parse_signal(self, raw_response: str, instrument: str) -> TradingSignal:
        """Parse the structured LLM response into a ``TradingSignal``.

        Args:
            raw_response: Raw LLM text output.
            instrument: Instrument name.

        Returns:
            Parsed ``TradingSignal`` object.
        """
        fields: dict[str, str] = {
            "direction": "NEUTRAL",
            "confidence": "0.5",
            "reasoning": "",
            "key_risks": "",
            "time_horizon": "SHORT_TERM",
        }

        for line in raw_response.split("\n"):
            for key in fields:
                prefix = f"{key.upper()}: "
                if line.upper().startswith(prefix):
                    fields[key] = line[len(prefix) :].strip()
                    break

        try:
            confidence = float(fields["confidence"])
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        return TradingSignal(
            instrument=instrument,
            direction=fields["direction"].upper(),
            confidence=confidence,
            reasoning=fields["reasoning"],
            key_risks=fields["key_risks"],
            time_horizon=fields["time_horizon"],
            raw_response=raw_response,
        )
