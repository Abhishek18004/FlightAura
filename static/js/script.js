"use strict";

// ============================================================
// FLIGHTAURA — FRONTEND CONTROLLER
// ============================================================

const API = {
    health: "/api/health",
    airlines: "/api/airlines",
    aspects: "/api/aspects",
    investigate: "/api/investigate",
    analyzeReview: "/api/analyze-review"
};


const elements = {
    claimForm: document.getElementById("claim-form"),
    claimInput: document.getElementById("claim-input"),
    claimCharacterCount: document.getElementById(
        "claim-character-count"
    ),

    airlineSelect: document.getElementById("airline-select"),
    aspectSelect: document.getElementById("aspect-select"),
    investigateButton: document.getElementById("investigate-button"),
    investigationError: document.getElementById(
        "investigation-error"
    ),

    loadingSection: document.getElementById("loading-section"),
    resultsSection: document.getElementById("results-section"),

    reviewInput: document.getElementById("review-input"),
    analyzeReviewButton: document.getElementById(
        "analyze-review-button"
    ),
    reviewError: document.getElementById("review-error"),
    reviewResults: document.getElementById("review-results")
};


// ============================================================
// API HELPERS
// ============================================================

async function apiRequest(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        },
        ...options
    });

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error("The server returned an invalid response.");
    }

    if (!response.ok) {
        throw new Error(
            data.error || "Something went wrong. Please try again."
        );
    }

    return data;
}


// ============================================================
// BASIC UI HELPERS
// ============================================================

function showError(element, message) {
    element.textContent = message;
    element.classList.remove("hidden");
}

function hideError(element) {
    element.textContent = "";
    element.classList.add("hidden");
}

function setButtonLoading(button, loading, loadingText, originalText) {
    button.disabled = loading;

    const label = button.querySelector("span:first-child");

    if (label) {
        label.textContent = loading ? loadingText : originalText;
    }
}

function formatPercentage(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${(number * 100).toFixed(1)}%`;
}

function formatRating(value) {
    if (value === null || value === undefined || value === "") {
        return null;
    }

    const number = Number(value);

    return Number.isFinite(number)
        ? number.toString()
        : null;
}


// ============================================================
// INITIALIZE DROPDOWNS
// ============================================================

async function loadAirlines() {
    const data = await apiRequest(API.airlines);

    const placeholder = elements.airlineSelect.options[0];

    elements.airlineSelect.replaceChildren(placeholder);

    data.airlines.forEach(airline => {
        const option = document.createElement("option");

        option.value = airline;
        option.textContent = airline;

        elements.airlineSelect.appendChild(option);
    });
}


async function loadAspects() {
    const data = await apiRequest(API.aspects);

    const placeholder = elements.aspectSelect.options[0];

    elements.aspectSelect.replaceChildren(placeholder);

    data.aspects.forEach(aspect => {
        const option = document.createElement("option");

        option.value = aspect.id;
        option.textContent = aspect.label;

        elements.aspectSelect.appendChild(option);
    });
}


async function checkBackendHealth() {
    try {
        const data = await apiRequest(API.health);

        console.info("FlightAura backend:", data.status);
    } catch (error) {
        console.error("FlightAura backend is unavailable:", error);
    }
}


// ============================================================
// CLAIM INPUT
// ============================================================

elements.claimInput.addEventListener("input", () => {
    const length = elements.claimInput.value.length;

    elements.claimCharacterCount.textContent = `${length} / 1000`;
});


document.querySelectorAll(".example-chip").forEach(button => {
    button.addEventListener("click", () => {
        elements.claimInput.value = button.dataset.claim || "";

        elements.claimInput.dispatchEvent(
            new Event("input")
        );

        elements.claimInput.focus();

        document.getElementById("investigate").scrollIntoView({
            behavior: "smooth"
        });
    });
});


// ============================================================
// EVIDENCE CARD RENDERING
// ============================================================

function createElement(tag, className, text = "") {
    const element = document.createElement(tag);

    if (className) {
        element.className = className;
    }

    element.textContent = text;

    return element;
}


function renderReviewCard(review, category) {
    const card = createElement("article", "review-card");

    const text = createElement(
        "p",
        "review-text",
        `“${review.review || "Review text unavailable."}”`
    );

    card.appendChild(text);

    const badges = createElement("div", "review-badges");

    const sentimentClass = {
        Positive: "positive",
        Negative: "negative",
        Neutral: "neutral"
    }[review.sentiment] || "neutral";

    badges.appendChild(
        createElement(
            "span",
            `review-badge ${sentimentClass}`,
            review.sentiment || "Unknown sentiment"
        )
    );

    badges.appendChild(
        createElement(
            "span",
            "review-badge",
            review.sentiment_source || "Unknown evidence source"
        )
    );

    if (review.aspect_rating !== null &&
        review.aspect_rating !== undefined) {

        badges.appendChild(
            createElement(
                "span",
                "review-badge",
                `Aspect rating: ${review.aspect_rating}`
            )
        );
    }

    if (review.overall_rating !== null &&
        review.overall_rating !== undefined) {

        badges.appendChild(
            createElement(
                "span",
                "review-badge",
                `Overall: ${review.overall_rating}/10`
            )
        );
    }

    card.appendChild(badges);

    const metadata = createElement("div", "review-meta");

    const metadataValues = [
        review.review_date,
        review.month_flown !== "Not available"
            ? `Flown: ${review.month_flown}`
            : null,
        review.seat_type !== "Not available"
            ? review.seat_type
            : null,
        review.traveller_type !== "Not available"
            ? review.traveller_type
            : null,
        review.aircraft !== "Not available"
            ? review.aircraft
            : null,
        review.route !== "Not available"
            ? review.route
            : null,
        review.recommended === true
            ? "Recommended"
            : review.recommended === false
                ? "Not recommended"
                : null
    ];

    metadataValues
        .filter(value => value && value !== "Not available")
        .forEach(value => {
            metadata.appendChild(
                createElement("span", "", String(value))
            );
        });

    card.appendChild(metadata);

    return card;
}


function renderReviewGroup(container, reviews, category, emptyMessage) {
    container.replaceChildren();

    if (!reviews || reviews.length === 0) {
        container.appendChild(
            createElement("div", "review-empty", emptyMessage)
        );
        return;
    }

    reviews.forEach(review => {
        container.appendChild(
            renderReviewCard(review, category)
        );
    });
}


// ============================================================
// VERDICT STYLING
// ============================================================

function updateVerdict(verdict) {
    const container = document.getElementById("verdict-container");

    container.classList.remove(
        "verdict-supported",
        "verdict-contradicted",
        "verdict-mixed",
        "verdict-insufficient"
    );

    const styles = {
        "Supported": "verdict-supported",
        "Contradicted": "verdict-contradicted",
        "Mixed Evidence": "verdict-mixed",
        "Insufficient Evidence": "verdict-insufficient"
    };

    container.classList.add(
        styles[verdict] || "verdict-insufficient"
    );
}


// ============================================================
// DISPLAY INVESTIGATION RESULTS
// ============================================================

function renderInvestigationResults(data) {
    elements.resultsSection.classList.remove("hidden");

    document.getElementById("result-claim").textContent =
        data.claim || "";

    document.getElementById("result-airline").textContent =
        data.airline || "";

    document.getElementById("result-aspect").textContent =
        data.aspect || "";

    document.getElementById("result-verdict").textContent =
        data.verdict || "Insufficient Evidence";

    document.getElementById("result-explanation").textContent =
        data.explanation || "";

    updateVerdict(data.verdict);

    const stats = data.statistics || {};

    document.getElementById("stat-total").textContent =
        stats.retrieved_reviews ?? 0;

    document.getElementById("stat-supporting").textContent =
        stats.supporting ?? 0;

    document.getElementById("stat-contradicting").textContent =
        stats.contradicting ?? 0;

    document.getElementById("stat-neutral").textContent =
        stats.neutral ?? 0;


    document.getElementById("supporting-heading").textContent =
        data.supporting_label || "Supporting evidence";

    document.getElementById("contradicting-heading").textContent =
        data.contradicting_label || "Contradicting evidence";


    const supporting = data.supporting_evidence || [];
    const contradicting = data.contradicting_evidence || [];
    const neutral = data.neutral_evidence || [];

    renderReviewGroup(
        document.getElementById("supporting-reviews"),
        supporting,
        "supporting",
        "No clearly supporting reviews were found."
    );

    renderReviewGroup(
        document.getElementById("contradicting-reviews"),
        contradicting,
        "contradicting",
        "No clearly contradicting reviews were found."
    );

    const neutralSection = document.getElementById(
        "neutral-evidence-section"
    );

    if (neutral.length > 0) {
        neutralSection.classList.remove("hidden");

        renderReviewGroup(
            document.getElementById("neutral-reviews"),
            neutral,
            "neutral",
            "No neutral reviews were found."
        );
    } else {
        neutralSection.classList.add("hidden");
    }

    const visibleEvidenceCount =
        supporting.length + contradicting.length + neutral.length;

    document.getElementById("evidence-count").textContent =
        `${visibleEvidenceCount} REVIEWS SHOWN`;


    document.getElementById("result-methodology").textContent =
        data.methodology || "";

    document.getElementById("result-limitation").textContent =
        data.limitation || "";


    elements.resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// ============================================================
// SUBMIT CLAIM INVESTIGATION
// ============================================================

elements.claimForm.addEventListener("submit", async event => {
    event.preventDefault();

    hideError(elements.investigationError);

    const claim = elements.claimInput.value.trim();

    if (claim.length < 10) {
        showError(
            elements.investigationError,
            "Please enter a more detailed claim."
        );
        return;
    }

    const payload = {
        claim,
        airline: elements.airlineSelect.value || null,
        aspect: elements.aspectSelect.value || null
    };

    elements.resultsSection.classList.add("hidden");
    elements.loadingSection.classList.remove("hidden");

    setButtonLoading(
        elements.investigateButton,
        true,
        "Investigating...",
        "Investigate claim"
    );

    try {
        const result = await apiRequest(API.investigate, {
            method: "POST",
            body: JSON.stringify(payload)
        });

        renderInvestigationResults(result);

    } catch (error) {
        showError(
            elements.investigationError,
            error.message
        );

    } finally {
        elements.loadingSection.classList.add("hidden");

        setButtonLoading(
            elements.investigateButton,
            false,
            "Investigating...",
            "Investigate claim"
        );
    }
});


// ============================================================
// REVIEW ANALYZER
// ============================================================

elements.analyzeReviewButton.addEventListener(
    "click",
    async () => {
        hideError(elements.reviewError);

        const review = elements.reviewInput.value.trim();

        if (review.length < 15) {
            showError(
                elements.reviewError,
                "Please enter a review containing at least 15 characters."
            );
            return;
        }

        setButtonLoading(
            elements.analyzeReviewButton,
            true,
            "Analyzing...",
            "Analyze review"
        );

        elements.reviewResults.classList.add("hidden");

        try {
            const result = await apiRequest(API.analyzeReview, {
                method: "POST",
                body: JSON.stringify({ review })
            });

            renderReviewAnalysis(result);

        } catch (error) {
            showError(elements.reviewError, error.message);

        } finally {
            setButtonLoading(
                elements.analyzeReviewButton,
                false,
                "Analyzing...",
                "Analyze review"
            );
        }
    }
);


function renderReviewAnalysis(data) {
    const sentiment = data.sentiment || {};
    const recommendation = data.recommendation || {};

    document.getElementById("review-sentiment").textContent =
        sentiment.sentiment || "Unknown";

    const probabilityContainer = document.getElementById(
        "sentiment-probabilities"
    );

    probabilityContainer.replaceChildren();

    const probabilities = sentiment.probabilities || {};

    Object.entries(probabilities).forEach(([label, probability]) => {
        const item = createElement("div", "probability-item");

        item.appendChild(
            createElement("span", "", label)
        );

        item.appendChild(
            createElement(
                "strong",
                "",
                formatPercentage(probability)
            )
        );

        probabilityContainer.appendChild(item);
    });


    document.getElementById("review-recommendation").textContent =
        recommendation.recommendation || "Unknown";

    document.getElementById(
        "review-recommendation-probability"
    ).textContent =
        `Model probability of recommendation: ${
            formatPercentage(
                recommendation.probability_recommended
            )
        }`;

    elements.reviewResults.classList.remove("hidden");

    elements.reviewResults.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


// ============================================================
// STARTUP
// ============================================================

async function initializeFlightAura() {
    try {
        await checkBackendHealth();
        await Promise.all([
            loadAirlines(),
            loadAspects()
        ]);

        console.info("FlightAura frontend initialized.");

    } catch (error) {
        console.error("FlightAura initialization failed:", error);

        showError(
            elements.investigationError,
            "Could not load airline information. Check that Flask is running."
        );
    }
}


initializeFlightAura();