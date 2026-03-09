"""FinBERT-based sentiment analysis for energy market news.

Uses the ProsusAI/finbert model (a financial domain BERT) to classify
news headlines and articles into positive, negative, or neutral sentiment.
Designed for batch processing of news article collections.
"""

from __future__ import annotations

import logging

import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Default model — FinBERT fine-tuned on financial text
DEFAULT_MODEL = "ProsusAI/finbert"
SENTIMENT_LABELS = ["positive", "negative", "neutral"]


class SentimentAnalyzer:
    """Transformer-based sentiment classifier for financial news text.

    Wraps the HuggingFace ``transformers`` pipeline for FinBERT inference.
    Supports batch processing for efficiency.

    Attributes:
        model_name: HuggingFace model identifier.
        batch_size: Number of texts per inference batch.
        max_length: Maximum tokenization length.
        device: PyTorch device (cpu or cuda).
        pipeline: HuggingFace sentiment analysis pipeline.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        """Initialise the sentiment analyser.

        Args:
            model_name: HuggingFace model path or identifier.
            batch_size: Batch size for inference.
            max_length: Maximum token sequence length.
            device: Device string (``"cpu"``, ``"cuda"``).  Auto-detected
                if ``None``.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.pipeline = None
        logger.info("SentimentAnalyzer initialised (model=%s, device=%s)", model_name, self.device)

    def _load_pipeline(self) -> None:
        """Lazy-load the HuggingFace pipeline on first use."""
        if self.pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            self.pipeline = hf_pipeline(
                task="text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                max_length=self.max_length,
                truncation=True,
            )
            logger.info("FinBERT pipeline loaded successfully")
        except Exception as exc:
            logger.error("Failed to load sentiment pipeline: %s", exc)
            raise

    def analyse_text(self, text: str) -> dict[str, float]:
        """Analyse sentiment of a single text string.

        Args:
            text: Input text (headline or article excerpt).

        Returns:
            Dictionary with keys ``label`` (str), ``score`` (float),
            and individual class scores ``positive``, ``negative``, ``neutral``.
        """
        self._load_pipeline()
        if not text or not text.strip():
            return {
                "label": "neutral",
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }

        try:
            result = self.pipeline(text, top_k=None)[0]  # type: ignore[index]
            scores = {item["label"].lower(): item["score"] for item in result}
            label = max(scores, key=scores.get)  # type: ignore[arg-type]
            return {
                "label": label,
                "score": scores[label],
                "positive": scores.get("positive", 0.0),
                "negative": scores.get("negative", 0.0),
                "neutral": scores.get("neutral", 0.0),
            }
        except Exception as exc:
            logger.warning("Sentiment analysis failed for text: %s", exc)
            return {
                "label": "neutral",
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }

    def analyse_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Analyse sentiment for a batch of texts.

        Args:
            texts: List of text strings.

        Returns:
            List of sentiment dictionaries (one per input text).
        """
        self._load_pipeline()
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Filter out empty texts and track their positions
            valid = [(j, t) for j, t in enumerate(batch) if t and t.strip()]
            batch_results: list[dict[str, float]] = [
                {"label": "neutral", "score": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
            ] * len(batch)
            if valid:
                indices, valid_texts = zip(*valid, strict=True)
                try:
                    raw = self.pipeline(list(valid_texts), top_k=None)  # type: ignore[arg-type]
                    for idx, item in zip(indices, raw, strict=True):
                        scores = {r["label"].lower(): r["score"] for r in item}
                        label = max(scores, key=scores.get)  # type: ignore[arg-type]
                        batch_results[idx] = {
                            "label": label,
                            "score": scores[label],
                            "positive": scores.get("positive", 0.0),
                            "negative": scores.get("negative", 0.0),
                            "neutral": scores.get("neutral", 0.0),
                        }
                except Exception as exc:
                    logger.warning("Batch sentiment analysis failed: %s", exc)
            results.extend(batch_results)
        return results

    def analyse_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "title",
    ) -> pd.DataFrame:
        """Analyse sentiment for all texts in a DataFrame column.

        Args:
            df: DataFrame containing a text column.
            text_column: Name of the column with text to analyse.

        Returns:
            Input DataFrame with additional columns:
            ``sentiment_label``, ``sentiment_score``,
            ``sentiment_positive``, ``sentiment_negative``, ``sentiment_neutral``.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        texts = df[text_column].fillna("").tolist()
        logger.info("Analysing sentiment for %d articles...", len(texts))
        sentiments = self.analyse_batch(texts)
        result_df = df.copy()
        result_df["sentiment_label"] = [s["label"] for s in sentiments]
        result_df["sentiment_score"] = [s["score"] for s in sentiments]
        result_df["sentiment_positive"] = [s["positive"] for s in sentiments]
        result_df["sentiment_negative"] = [s["negative"] for s in sentiments]
        result_df["sentiment_neutral"] = [s["neutral"] for s in sentiments]
        # Net sentiment: positive - negative (range -1 to +1)
        result_df["net_sentiment"] = (
            result_df["sentiment_positive"] - result_df["sentiment_negative"]
        )
        logger.info("Sentiment analysis complete")
        return result_df

    def compute_composite_score(self, row: dict) -> float:
        """Compute a composite sentiment score from FinBERT probabilities.

        Maps class probabilities onto a -1 to +1 scale:
        ``score = positive_prob - negative_prob``

        Args:
            row: Dictionary or Series with ``positive`` and ``negative`` keys.

        Returns:
            Float score in range [-1, +1].
        """
        return float(row.get("positive", 0.0)) - float(row.get("negative", 0.0))
