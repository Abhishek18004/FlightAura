from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request

from claim_engine import ClaimEngine, ASPECTS
from model_service import FlightAuraModelService
from utils import (
    DATASET_PATH,
    load_review_dataset,
)


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# APPLICATION FACTORY
# ---------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.getenv(
        "SECRET_KEY",
        "flightaura-local-development-key",
    )

    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

    # Services are initialized once, not once per API request.
    app.model_service = None
    app.claim_engine = None
    app.startup_error = None

    try:
        logger.info("Loading review dataset: %s", DATASET_PATH)

        dataframe, columns = load_review_dataset(DATASET_PATH)

        logger.info(
            "Loaded %d usable review records.",
            len(dataframe),
        )

        model_service = FlightAuraModelService()

        claim_engine = ClaimEngine(
            dataframe=dataframe,
            columns=columns,
            model_service=model_service,
        )

        app.model_service = model_service
        app.claim_engine = claim_engine

        logger.info("FlightAura application is ready.")

    except Exception as error:
        app.startup_error = str(error)
        logger.exception("FlightAura startup failed.")

    # -----------------------------------------------------------------
    # FRONTEND
    # -----------------------------------------------------------------

    @app.get("/")
    def home():
        return render_template("index.html")

    # -----------------------------------------------------------------
    # HEALTH
    # -----------------------------------------------------------------

    @app.get("/api/health")
    def health():
        ready = (
            app.model_service is not None
            and app.claim_engine is not None
        )

        return jsonify({
            "status": "ready" if ready else "error",
            "dataset": str(DATASET_PATH),
            "message": (
                "FlightAura is ready."
                if ready
                else app.startup_error
            ),
        }), 200 if ready else 503

    # -----------------------------------------------------------------
    # AIRLINES
    # -----------------------------------------------------------------

    @app.get("/api/airlines")
    def airlines():
        if app.claim_engine is None:
            return jsonify({
                "error": "The application has not initialized correctly."
            }), 503

        return jsonify({
            "airlines": app.claim_engine.get_airlines(),
        })

    # -----------------------------------------------------------------
    # ASPECTS
    # -----------------------------------------------------------------

    @app.get("/api/aspects")
    def aspects():
        return jsonify({
            "aspects": [
                {
                    "id": aspect_id,
                    "label": aspect["label"],
                }
                for aspect_id, aspect in ASPECTS.items()
            ]
        })

    # -----------------------------------------------------------------
    # CLAIM INVESTIGATION
    # -----------------------------------------------------------------

    @app.post("/api/investigate")
    def investigate():
        if app.claim_engine is None:
            return jsonify({
                "error": "The application is not ready. Check the terminal."
            }), 503

        data = request.get_json(silent=True) or {}

        claim = str(data.get("claim", "")).strip()
        airline = data.get("airline") or None
        aspect = data.get("aspect") or None

        if len(claim) < 10:
            return jsonify({
                "error": (
                    "Please enter a claim containing at least "
                    "10 characters."
                )
            }), 400

        try:
            result = app.claim_engine.investigate(
                claim=claim,
                selected_airline=airline,
                selected_aspect=aspect,
            )

            return jsonify(result)

        except ValueError as error:
            return jsonify({
                "error": str(error),
            }), 400

        except Exception:
            logger.exception("Claim investigation failed.")

            return jsonify({
                "error": (
                    "An unexpected error occurred during investigation. "
                    "Check the Flask terminal for details."
                )
            }), 500

    # -----------------------------------------------------------------
    # USER REVIEW ANALYSIS
    # -----------------------------------------------------------------

    @app.post("/api/analyze-review")
    def analyze_review():
        if app.model_service is None:
            return jsonify({
                "error": "The prediction models are not available."
            }), 503

        data = request.get_json(silent=True) or {}

        review_text = str(data.get("review", "")).strip()

        if len(review_text) < 15:
            return jsonify({
                "error": "Please enter a review containing at least 15 characters."
            }), 400

        try:
            result = app.model_service.analyze_review(review_text)

            return jsonify(result)

        except ValueError as error:
            return jsonify({
                "error": str(error),
            }), 400

        except Exception:
            logger.exception("Review analysis failed.")

            return jsonify({
                "error": (
                    "An unexpected error occurred while analyzing the review."
                )
            }), 500

    # -----------------------------------------------------------------
    # ERROR HANDLERS
    # -----------------------------------------------------------------

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({
            "error": "The submitted content is too large.",
        }), 413

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({
            "error": "The requested resource was not found.",
        }), 404

    return app


# ---------------------------------------------------------------------
# RUN LOCALLY
# ---------------------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    debug_mode = (
        os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=debug_mode,
        use_reloader=False,
    )