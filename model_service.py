from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

from utils import (
    CLASSICAL_DIR,
    DL_DIR,
    load_json,
    clean_text,
)


logger = logging.getLogger(__name__)

# Suppress excessive TensorFlow informational logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class FlightAuraModelService:
    """Load trained models and provide sentiment/recommendation predictions."""

    SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

    SENTIMENT_MODEL_FILES = {
        "ANN": "ann_sentiment.keras",
        "GRU": "gru_sentiment.keras",
        "BiLSTM": "bilstm_sentiment.keras",
    }

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.tokenizer = None

        self.sequence_length = 365

        self.sentiment_weights: dict[str, float] = {}
        self.sentiment_models: list[str] = []

        self.recommendation_model_name = "BiLSTM"
        self.recommendation_threshold = 0.5

        self.lr_vectorizer = None
        self.lr_model = None

        self.ann_vectorizer = None
        self.ann_svd = None
        self.ann_scaler = None

        self.loaded = False

        self.load_all()

    # -----------------------------------------------------------------
    # ARTIFACT LOADING
    # -----------------------------------------------------------------

    def _require_file(self, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact is missing: {path}"
            )
        return path

    def load_all(self) -> None:
        """Load every artifact required by the selected application models."""
        logger.info("Loading FlightAura models...")

        sequence_config = load_json(DL_DIR / "sequence_config.json")
        sentiment_config = load_json(
            DL_DIR / "final_sentiment_config.json"
        )
        recommendation_config = load_json(
            DL_DIR / "recommendation_config.json"
        )

        self.sequence_length = int(
            sequence_config["max_sequence_length"]
        )

        # Tokenizer used by recurrent models.
        tokenizer_path = self._require_file(
            DL_DIR / "flightaura_tokenizer.json"
        )

        with open(tokenizer_path, "r", encoding="utf-8") as file:
            self.tokenizer = tokenizer_from_json(file.read())

        # Logistic Regression artifacts.
        self.lr_vectorizer = joblib.load(
            self._require_file(
                CLASSICAL_DIR / "classical_tfidf_vectorizer.joblib"
            )
        )

        self.lr_model = joblib.load(
            self._require_file(
                CLASSICAL_DIR / "logisticregression_sentiment.joblib"
            )
        )

        # ANN preprocessing artifacts.
        self.ann_vectorizer = joblib.load(
            self._require_file(
                CLASSICAL_DIR / "ann_tfidf_vectorizer.joblib"
            )
        )

        self.ann_svd = joblib.load(
            self._require_file(
                CLASSICAL_DIR / "ann_truncated_svd.joblib"
            )
        )

        self.ann_scaler = joblib.load(
            self._require_file(
                CLASSICAL_DIR / "ann_scaler.joblib"
            )
        )

        # Load only the architectures used in the final sentiment ensemble.
        selected_models = sentiment_config.get(
            "selected_ensemble_models",
            ["LogisticRegression", "ANN", "GRU", "BiLSTM"],
        )

        self.sentiment_models = list(selected_models)

        self.sentiment_weights = {
            str(name): float(weight)
            for name, weight in sentiment_config[
                "ensemble_weights"
            ].items()
        }

        for model_name in self.sentiment_models:
            if model_name == "LogisticRegression":
                continue

            filename = self.SENTIMENT_MODEL_FILES.get(model_name)

            if filename is None:
                raise ValueError(
                    f"Unsupported sentiment model in config: {model_name}"
                )

            model_path = self._require_file(DL_DIR / filename)

            self.models[model_name] = tf.keras.models.load_model(
                model_path,
                compile=False,
            )

            logger.info("Loaded sentiment model: %s", model_name)

        # Recommendation model selected in Notebook 03.
        self.recommendation_model_name = recommendation_config.get(
            "selected_model",
            "BiLSTM",
        )

        self.recommendation_threshold = float(
            recommendation_config.get("threshold", 0.5)
        )

        recommendation_files = {
            "ANN": "ann_recommendation.keras",
            "GRU": "gru_recommendation.keras",
            "BiLSTM": "bilstm_recommendation.keras",
        }

        recommendation_filename = recommendation_files.get(
            self.recommendation_model_name
        )

        if recommendation_filename is None:
            raise ValueError(
                "Unsupported recommendation model: "
                f"{self.recommendation_model_name}"
            )

        self.models["recommendation"] = (
            tf.keras.models.load_model(
                self._require_file(
                    DL_DIR / recommendation_filename
                ),
                compile=False,
            )
        )

        self.loaded = True

        logger.info("All FlightAura models loaded successfully.")

    # -----------------------------------------------------------------
    # INPUT PREPROCESSING
    # -----------------------------------------------------------------

    def _prepare_sequences(self, texts: list[str]) -> np.ndarray:
        """Tokenize and pad text using the Notebook 03 configuration."""
        cleaned = [clean_text(text) for text in texts]

        sequences = self.tokenizer.texts_to_sequences(cleaned)

        return pad_sequences(
            sequences,
            maxlen=self.sequence_length,
            padding="post",
            truncating="post",
        )

    # -----------------------------------------------------------------
    # MODEL PREDICTIONS
    # -----------------------------------------------------------------

    @staticmethod
    def _align_probabilities(
        probabilities: np.ndarray,
        model_classes: np.ndarray,
        number_of_classes: int = 3,
    ) -> np.ndarray:
        """Align model probabilities to integer class IDs 0, 1 and 2."""
        aligned = np.zeros(
            (len(probabilities), number_of_classes),
            dtype=np.float64,
        )

        for source_index, class_value in enumerate(model_classes):
            class_id = int(class_value)

            if 0 <= class_id < number_of_classes:
                aligned[:, class_id] = probabilities[:, source_index]

        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0

        return aligned / row_sums

    def _predict_logistic_regression(
        self,
        texts: list[str],
    ) -> np.ndarray:
        features = self.lr_vectorizer.transform(texts)

        probabilities = self.lr_model.predict_proba(features)

        return self._align_probabilities(
            probabilities,
            self.lr_model.classes_,
        )

    def _predict_ann(
        self,
        texts: list[str],
    ) -> np.ndarray:
        tfidf_features = self.ann_vectorizer.transform(texts)

        reduced_features = self.ann_svd.transform(tfidf_features)

        scaled_features = self.ann_scaler.transform(reduced_features)

        probabilities = self.models["ANN"].predict(
            scaled_features,
            verbose=0,
        )

        return np.asarray(probabilities, dtype=np.float64)

    def _predict_recurrent_sentiment(
        self,
        texts: list[str],
        model_name: str,
    ) -> np.ndarray:
        sequences = self._prepare_sequences(texts)

        probabilities = self.models[model_name].predict(
            sequences,
            verbose=0,
        )

        return np.asarray(probabilities, dtype=np.float64)

    def predict_sentiment(
        self,
        texts: str | list[str],
    ) -> list[dict[str, Any]]:
        """
        Predict review sentiment using the saved weighted-soft-voting system.

        Returns one prediction dictionary per input review.
        """
        if not self.loaded:
            raise RuntimeError("FlightAura models are not loaded.")

        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = [clean_text(text) for text in texts]

        if not cleaned_texts:
            return []

        if any(not text for text in cleaned_texts):
            raise ValueError("Review text cannot be empty.")

        model_probabilities: dict[str, np.ndarray] = {}

        for model_name in self.sentiment_models:
            if model_name == "LogisticRegression":
                model_probabilities[model_name] = (
                    self._predict_logistic_regression(cleaned_texts)
                )

            elif model_name == "ANN":
                model_probabilities[model_name] = (
                    self._predict_ann(cleaned_texts)
                )

            elif model_name in {"GRU", "BiLSTM"}:
                model_probabilities[model_name] = (
                    self._predict_recurrent_sentiment(
                        cleaned_texts,
                        model_name,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported sentiment model: {model_name}"
                )

        combined = np.zeros(
            (len(cleaned_texts), 3),
            dtype=np.float64,
        )

        total_weight = 0.0

        for model_name, probabilities in model_probabilities.items():
            weight = self.sentiment_weights.get(model_name, 0.0)

            if weight <= 0:
                continue

            if probabilities.shape[1] != 3:
                raise ValueError(
                    f"{model_name} returned an unexpected output shape: "
                    f"{probabilities.shape}"
                )

            combined += weight * probabilities
            total_weight += weight

        if total_weight <= 0:
            raise ValueError("Sentiment ensemble weights are invalid.")

        combined /= total_weight

        predictions = []

        for probabilities in combined:
            class_id = int(np.argmax(probabilities))

            predictions.append({
                "sentiment": self.SENTIMENT_LABELS[class_id],
                "confidence": round(
                    float(probabilities[class_id]), 4
                ),
                "probabilities": {
                    label: round(float(probabilities[index]), 4)
                    for index, label in enumerate(
                        self.SENTIMENT_LABELS
                    )
                },
            })

        return predictions

    def predict_recommendation(
        self,
        texts: str | list[str],
    ) -> list[dict[str, Any]]:
        """Predict whether a reviewer would recommend the airline."""
        if not self.loaded:
            raise RuntimeError("FlightAura models are not loaded.")

        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = [clean_text(text) for text in texts]

        if not cleaned_texts:
            return []

        if any(not text for text in cleaned_texts):
            raise ValueError("Review text cannot be empty.")

        sequences = self._prepare_sequences(cleaned_texts)

        probabilities = self.models["recommendation"].predict(
            sequences,
            verbose=0,
        )

        probabilities = np.asarray(probabilities).reshape(-1)

        results = []

        for probability in probabilities:
            probability = float(probability)

            results.append({
                "recommendation": (
                    "Recommended"
                    if probability >= self.recommendation_threshold
                    else "Not Recommended"
                ),
                "probability_recommended": round(probability, 4),
                "threshold": self.recommendation_threshold,
            })

        return results

    def analyze_review(self, review_text: str) -> dict[str, Any]:
        """Run both trained prediction systems on one user-submitted review."""
        text = clean_text(review_text)

        if len(text) < 15:
            raise ValueError(
                "Please enter a review containing at least 15 characters."
            )

        sentiment = self.predict_sentiment(text)[0]
        recommendation = self.predict_recommendation(text)[0]

        return {
            "sentiment": sentiment,
            "recommendation": recommendation,
        }