from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from utils import (
    MAX_EVIDENCE_REVIEWS,
    MIN_RETRIEVAL_SIMILARITY,
    clean_text,
    normalize_airline_name,
    normalize_recommendation,
    safe_float,
    safe_string,
    split_sentences,
    json_safe,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# ASPECT DEFINITIONS
# ---------------------------------------------------------------------

ASPECTS = {
    "seat_comfort": {
        "label": "Seat Comfort",
        "keywords": [
            "seat", "seats", "seating", "legroom", "leg room",
            "pitch", "recline", "reclining", "cabin space",
            "spacious", "cramped", "comfort",
        ],
        "rating_field": "seat_comfort",
        "search_terms": "seat comfort legroom seat pitch seating space",
    },
    "cabin_staff": {
        "label": "Cabin Staff Service",
        "keywords": [
            "cabin crew", "flight attendant", "flight attendants",
            "cabin staff", "crew", "staff", "service",
            "attentive", "rude", "friendly", "unhelpful",
        ],
        "rating_field": "cabin_staff",
        "search_terms": "cabin crew flight attendants staff service",
    },
    "food": {
        "label": "Food and Beverages",
        "keywords": [
            "food", "meal", "meals", "beverage", "beverages",
            "drink", "drinks", "dinner", "breakfast", "lunch",
            "catering", "snack",
        ],
        "rating_field": "food",
        "search_terms": "food meals beverages drinks catering dining",
    },
    "ground_service": {
        "label": "Ground Service",
        "keywords": [
            "ground service", "airport staff", "check in",
            "check-in", "boarding", "baggage", "luggage",
            "airport", "transfer", "lounge", "queue",
            "customer service",
        ],
        "rating_field": "ground_service",
        "search_terms": "ground service airport check in boarding baggage",
    },
    "wifi": {
        "label": "Wi-Fi and Connectivity",
        "keywords": [
            "wifi", "wi-fi", "internet", "connectivity",
            "connection", "online", "broadband",
        ],
        "rating_field": "wifi",
        "search_terms": "wifi wi-fi internet connectivity connection",
    },
    "entertainment": {
        "label": "In-flight Entertainment",
        "keywords": [
            "entertainment", "inflight entertainment",
            "in-flight entertainment", "screen", "screens",
            "movies", "movie", "music", "headphones",
            "content", "display",
        ],
        "rating_field": "entertainment",
        "search_terms": "inflight entertainment screens movies music headphones",
    },
    "value": {
        "label": "Value for Money",
        "keywords": [
            "value", "price", "pricing", "cost", "expensive",
            "cheap", "worth", "money", "fare", "overpriced",
        ],
        "rating_field": "value",
        "search_terms": "value for money price cost fare worth expensive",
    },
    "overall": {
        "label": "Overall Experience",
        "keywords": [
            "overall", "experience", "flight", "airline",
            "journey", "trip", "recommend", "satisfaction",
        ],
        "rating_field": "overall_rating",
        "search_terms": "overall airline flight experience journey satisfaction",
    },
}


# ---------------------------------------------------------------------
# SIMPLE POLARITY LEXICON
# ---------------------------------------------------------------------

POSITIVE_TERMS = {
    "excellent", "amazing", "great", "good", "comfortable",
    "comfort", "friendly", "professional", "helpful",
    "attentive", "smooth", "efficient", "clean", "delicious",
    "pleasant", "wonderful", "outstanding", "superb",
    "spacious", "reliable", "impressive", "enjoyable",
    "fantastic", "exceptional", "satisfied", "best",
    "recommend", "recommended", "easy", "quick",
}

NEGATIVE_TERMS = {
    "bad", "poor", "terrible", "awful", "horrible",
    "uncomfortable", "unfriendly", "rude", "slow",
    "delayed", "delay", "dirty", "cramped", "disappointing",
    "disappointed", "worst", "inattentive", "unhelpful",
    "broken", "damaged", "unpleasant", "overpriced",
    "expensive", "inadequate", "unacceptable", "dissatisfied",
    "waste", "disorganized", "cancelled", "canceled",
}

NEGATION_TERMS = {
    "not", "never", "no", "hardly", "isnt", "wasnt",
    "weren't", "couldn't", "didn't", "doesn't", "without",
}


def _lexicon_polarity(text: str) -> tuple[str, float]:
    """
    Estimate polarity with a small transparent word lexicon.

    This is a fallback for aspect-related sentences, not a replacement
    for a trained language model.
    """
    words = re.findall(r"[a-z']+", clean_text(text).lower())

    positive_score = 0
    negative_score = 0

    for index, word in enumerate(words):
        negated = any(
            previous in NEGATION_TERMS
            for previous in words[max(0, index - 3):index]
        )

        if word in POSITIVE_TERMS:
            if negated:
                negative_score += 1
            else:
                positive_score += 1

        elif word in NEGATIVE_TERMS:
            if negated:
                positive_score += 1
            else:
                negative_score += 1

    total = positive_score + negative_score

    if total == 0:
        return "Neutral", 0.0

    difference = positive_score - negative_score

    if difference > 0:
        return "Positive", difference / total

    if difference < 0:
        return "Negative", abs(difference) / total

    return "Neutral", 0.0


# ---------------------------------------------------------------------
# CLAIM PARSING
# ---------------------------------------------------------------------

class ClaimEngine:
    """Retrieve historical airline reviews and evaluate claim evidence."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        columns: dict[str, str | None],
        model_service: Any,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.columns = columns
        self.model_service = model_service

        self.airline_column = columns["airline"]
        self.review_column = columns["review_text"]

        if not self.airline_column or not self.review_column:
            raise ValueError(
                "The dataset must contain airline and review text columns."
            )

        self.airline_names = sorted(
            self.df[self.airline_column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        self.normalized_airlines = {
            normalize_airline_name(name): name
            for name in self.airline_names
        }

        # Build this once. It avoids repeatedly vectorizing the dataset
        # every time a user investigates a claim.
        logger.info("Building local review retrieval index...")

        vectorizer = self.model_service.lr_vectorizer
        review_texts = self.df[self.review_column].fillna("").astype(str)

        self.review_matrix = vectorizer.transform(review_texts)

        logger.info(
            "Review retrieval index ready with %d reviews.",
            len(self.df),
        )

    # -----------------------------------------------------------------
    # PUBLIC DATA
    # -----------------------------------------------------------------

    def get_airlines(self) -> list[str]:
        return self.airline_names

    # -----------------------------------------------------------------
    # CLAIM UNDERSTANDING
    # -----------------------------------------------------------------

    def identify_airline(
        self,
        claim: str,
        selected_airline: str | None = None,
    ) -> str | None:
        """Find an airline in the claim or use the supplied selection."""
        if selected_airline:
            selected_normalized = normalize_airline_name(selected_airline)

            for normalized_name, original_name in (
                self.normalized_airlines.items()
            ):
                if selected_normalized == normalized_name:
                    return original_name

            return None

        normalized_claim = normalize_airline_name(claim)

        # Longest match first avoids matching a short airline alias
        # before a full airline name.
        for normalized_name, original_name in sorted(
            self.normalized_airlines.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if not normalized_name:
                continue

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(normalized_name)
                + r"(?![a-z0-9])"
            )

            if re.search(pattern, normalized_claim):
                return original_name

        return None

    @staticmethod
    def identify_aspect(
        claim: str,
        selected_aspect: str | None = None,
    ) -> str | None:
        """Identify a service aspect from the claim or explicit selection."""
        if selected_aspect:
            if selected_aspect in ASPECTS:
                return selected_aspect
            return None

        normalized_claim = clean_text(claim).lower()

        matches = []

        for aspect_id, aspect in ASPECTS.items():
            if aspect_id == "overall":
                continue

            count = sum(
                1
                for keyword in aspect["keywords"]
                if keyword in normalized_claim
            )

            if count:
                matches.append((aspect_id, count))

        if not matches:
            return "overall"

        matches.sort(key=lambda item: item[1], reverse=True)

        return matches[0][0]

    @staticmethod
    def identify_claim_direction(claim: str) -> str:
        """Identify whether the claim expresses positive or negative quality."""
        sentiment, strength = _lexicon_polarity(claim)

        if strength == 0 or sentiment == "Neutral":
            return "unspecified"

        return "positive" if sentiment == "Positive" else "negative"

    # -----------------------------------------------------------------
    # ASPECT-SPECIFIC REVIEW ANALYSIS
    # -----------------------------------------------------------------

    def _get_rating(
        self,
        row: pd.Series,
        aspect_id: str,
    ) -> float | None:
        """Get the rating associated with the selected aspect."""
        aspect = ASPECTS[aspect_id]
        canonical_field = aspect["rating_field"]

        column = self.columns.get(canonical_field)

        if not column:
            return None

        return safe_float(row.get(column))

    @staticmethod
    def _find_aspect_sentences(
        review: str,
        aspect_id: str,
    ) -> list[str]:
        """Find sentences that mention the identified service aspect."""
        keywords = ASPECTS[aspect_id]["keywords"]

        matched = []

        for sentence in split_sentences(review):
            normalized = sentence.lower()

            if any(keyword in normalized for keyword in keywords):
                matched.append(sentence)

        return matched

    def _classify_evidence(
        self,
        row: pd.Series,
        aspect_id: str,
        full_review_sentiment: dict[str, Any],
    ) -> tuple[str, str, float | None]:
        """
        Determine evidence polarity.

        Priority:
        1. Relevant structured aspect rating.
        2. Aspect-specific review sentence.
        3. Whole-review model sentiment as a fallback.
        """
        rating = self._get_rating(row, aspect_id)

        if rating is not None and aspect_id != "overall":
            if rating >= 4:
                return "Positive", "Aspect rating", rating

            if rating <= 2:
                return "Negative", "Aspect rating", rating

            return "Neutral", "Aspect rating", rating

        if aspect_id == "overall":
            rating = self._get_rating(row, "overall")

            if rating is not None:
                if rating >= 7:
                    return "Positive", "Overall rating", rating

                if rating <= 4:
                    return "Negative", "Overall rating", rating

                return "Neutral", "Overall rating", rating

        review_text = clean_text(row.get(self.review_column, ""))

        aspect_sentences = self._find_aspect_sentences(
            review_text,
            aspect_id,
        )

        if aspect_sentences:
            sentence_text = " ".join(aspect_sentences)
            sentiment, strength = _lexicon_polarity(sentence_text)

            if sentiment != "Neutral" and strength > 0:
                return (
                    sentiment,
                    "Aspect-related text",
                    rating,
                )

        return (
            full_review_sentiment["sentiment"],
            "Whole-review model fallback",
            rating,
        )

    # -----------------------------------------------------------------
    # REVIEW RETRIEVAL
    # -----------------------------------------------------------------

    def _retrieve_reviews(
        self,
        claim: str,
        airline: str,
        aspect_id: str,
        candidate_count: int = 40,
    ) -> list[dict[str, Any]]:
        """Retrieve similar reviews belonging to the selected airline."""
        airline_mask = (
            self.df[self.airline_column]
            .astype(str)
            .map(normalize_airline_name)
            == normalize_airline_name(airline)
        )

        airline_indices = np.flatnonzero(airline_mask.to_numpy())

        if len(airline_indices) == 0:
            return []

        aspect_search_terms = ASPECTS[aspect_id]["search_terms"]

        search_query = f"{claim} {aspect_search_terms}"

        vectorizer = self.model_service.lr_vectorizer
        query_vector = vectorizer.transform([search_query])

        similarities = (
            self.review_matrix[airline_indices] @ query_vector.T
        ).toarray().ravel()

        # Slightly prefer reviews that explicitly discuss the aspect.
        aspect_keywords = ASPECTS[aspect_id]["keywords"]
        review_texts = self.df.iloc[airline_indices][
            self.review_column
        ].astype(str)

        aspect_boost = np.array([
            0.08
            if any(
                keyword in review.lower()
                for keyword in aspect_keywords
            )
            else 0.0
            for review in review_texts
        ])

        combined_scores = similarities + aspect_boost

        best_positions = np.argsort(combined_scores)[::-1]
        best_positions = best_positions[:candidate_count]

        results = []

        for position in best_positions:
            score = float(combined_scores[position])

            # Don't return completely unrelated reviews just because
            # the aspect boost was applied.
            if float(similarities[position]) < MIN_RETRIEVAL_SIMILARITY:
                continue

            dataframe_index = int(airline_indices[position])

            results.append({
                "index": dataframe_index,
                "similarity": score,
                "row": self.df.iloc[dataframe_index],
            })

        return results

    # -----------------------------------------------------------------
    # EVIDENCE PRESENTATION
    # -----------------------------------------------------------------

    def _format_review(
        self,
        row: pd.Series,
        sentiment: str,
        sentiment_source: str,
        rating: float | None,
        similarity: float,
    ) -> dict[str, Any]:
        """Prepare one review for the frontend."""
        def value_for(field: str) -> Any:
            column = self.columns.get(field)

            if not column:
                return None

            return json_safe(row.get(column))

        recommendation = normalize_recommendation(
            row.get(self.columns["recommended"])
            if self.columns.get("recommended")
            else None
        )

        return {
            "review": clean_text(row.get(self.review_column, "")),
            "sentiment": sentiment,
            "sentiment_source": sentiment_source,
            "aspect_rating": rating,
            "overall_rating": safe_float(
                row.get(self.columns["overall_rating"])
            ) if self.columns.get("overall_rating") else None,
            "review_date": safe_string(value_for("review_date"), "Not available"),
            "month_flown": safe_string(value_for("month_flown"), "Not available"),
            "seat_type": safe_string(value_for("seat_type"), "Not available"),
            "traveller_type": safe_string(
                value_for("traveller_type"),
                "Not available",
            ),
            "aircraft": safe_string(value_for("aircraft"), "Not available"),
            "route": safe_string(value_for("route"), "Not available"),
            "recommended": recommendation,
            "similarity": round(similarity, 4),
        }

    # -----------------------------------------------------------------
    # FINAL CLAIM INVESTIGATION
    # -----------------------------------------------------------------

    def investigate(
        self,
        claim: str,
        selected_airline: str | None = None,
        selected_aspect: str | None = None,
    ) -> dict[str, Any]:
        """Investigate a passenger-experience claim using historical reviews."""
        claim = clean_text(claim)

        if len(claim) < 10:
            raise ValueError(
                "Please enter a more detailed claim (at least 10 characters)."
            )

        airline = self.identify_airline(
            claim,
            selected_airline,
        )

        if not airline:
            raise ValueError(
                "Please select an airline or mention an airline in your claim."
            )

        aspect_id = self.identify_aspect(
            claim,
            selected_aspect,
        )

        if not aspect_id:
            raise ValueError("Please select a supported service aspect.")

        direction = self.identify_claim_direction(claim)

        retrieved = self._retrieve_reviews(
            claim=claim,
            airline=airline,
            aspect_id=aspect_id,
        )

        if not retrieved:
            return {
                "claim": claim,
                "airline": airline,
                "aspect": ASPECTS[aspect_id]["label"],
                "claim_direction": direction,
                "verdict": "Insufficient Evidence",
                "explanation": (
                    "The dataset did not contain sufficiently similar "
                    "reviews for this airline and service aspect."
                ),
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "neutral_evidence": [],
                "statistics": {
                    "retrieved_reviews": 0,
                    "supporting": 0,
                    "contradicting": 0,
                    "neutral": 0,
                },
            }

        candidate_texts = [
            item["row"][self.review_column]
            for item in retrieved
        ]

        model_predictions = self.model_service.predict_sentiment(
            candidate_texts
        )

        supporting = []
        contradicting = []
        neutral = []

        for item, model_prediction in zip(
            retrieved,
            model_predictions,
        ):
            row = item["row"]

            sentiment, source, aspect_rating = (
                self._classify_evidence(
                    row,
                    aspect_id,
                    model_prediction,
                )
            )

            formatted = self._format_review(
                row=row,
                sentiment=sentiment,
                sentiment_source=source,
                rating=aspect_rating,
                similarity=item["similarity"],
            )

            if direction == "positive":
                if sentiment == "Positive":
                    supporting.append(formatted)
                elif sentiment == "Negative":
                    contradicting.append(formatted)
                else:
                    neutral.append(formatted)

            elif direction == "negative":
                if sentiment == "Negative":
                    supporting.append(formatted)
                elif sentiment == "Positive":
                    contradicting.append(formatted)
                else:
                    neutral.append(formatted)

            else:
                if sentiment == "Positive":
                    supporting.append(formatted)
                elif sentiment == "Negative":
                    contradicting.append(formatted)
                else:
                    neutral.append(formatted)

        # Sort evidence by retrieval relevance.
        supporting.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )
        contradicting.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )
        neutral.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        support_count = len(supporting)
        contradiction_count = len(contradicting)
        neutral_count = len(neutral)

        total_directional = support_count + contradiction_count

        if direction == "unspecified":
            verdict = "Insufficient Evidence"

            explanation = (
                "The claim does not clearly express a positive or negative "
                "judgment. The retrieved reviews are shown by sentiment, "
                "but they cannot be reliably classified as supporting or "
                "contradicting the claim."
            )

        elif total_directional < 2:
            verdict = "Insufficient Evidence"

            explanation = (
                "There are not enough clearly directional reviews to "
                "support a reliable conclusion."
            )

        elif (
            support_count >= 2
            and support_count >= 2 * max(contradiction_count, 1)
        ):
            verdict = "Supported"

            explanation = (
                "The retrieved historical reviews contain more evidence "
                "consistent with the claim than evidence opposing it."
            )

        elif (
            contradiction_count >= 2
            and contradiction_count >= 2 * max(support_count, 1)
        ):
            verdict = "Contradicted"

            explanation = (
                "The retrieved historical reviews contain more evidence "
                "opposing the claim than evidence consistent with it."
            )

        elif support_count > 0 and contradiction_count > 0:
            verdict = "Mixed Evidence"

            explanation = (
                "The retrieved historical reviews contain both supporting "
                "and contradicting experiences."
            )

        else:
            verdict = "Insufficient Evidence"

            explanation = (
                "The available reviews do not provide enough consistent "
                "evidence for a conclusion."
            )

        if direction == "unspecified":
            supporting_label = "Positive reviews"
            contradicting_label = "Negative reviews"
        else:
            supporting_label = "Supporting reviews"
            contradicting_label = "Contradicting reviews"

        return {
            "claim": claim,
            "airline": airline,
            "aspect": ASPECTS[aspect_id]["label"],
            "claim_direction": direction,
            "verdict": verdict,
            "explanation": explanation,
            "supporting_label": supporting_label,
            "contradicting_label": contradicting_label,
            "supporting_evidence": supporting[:3],
            "contradicting_evidence": contradicting[:3],
            "neutral_evidence": neutral[:3],
            "statistics": {
                "retrieved_reviews": len(retrieved),
                "supporting": support_count,
                "contradicting": contradiction_count,
                "neutral": neutral_count,
            },
            "methodology": (
                "Evidence is ranked using TF-IDF similarity within the "
                "selected airline. Aspect-specific ratings are preferred; "
                "aspect-related text is used when ratings are unavailable. "
                "Whole-review model sentiment is a fallback."
            ),
            "limitation": (
                "These results summarize reviews in the available dataset. "
                "They do not independently verify airline service quality "
                "or represent every passenger."
            ),
        }