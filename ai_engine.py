import re
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# ============================================================
# ENTERPRISE INCIDENT CATEGORIES
# ============================================================

CATEGORY_ENTRIES = [
    {
        "label": "Network Outage",
        "description": (
            "Network devices are down, users cannot access systems, "
            "internet connectivity is unavailable, or there is an outage "
            "affecting LAN, WAN, DNS, routers, switches, or connectivity."
        ),
        "team": "Network Operations",
    },
    {
        "label": "Security Incident",
        "description": (
            "A potential security breach, unauthorized access, malware "
            "infection, ransomware attack, suspicious login, compromised "
            "credentials, or other security-related activity."
        ),
        "team": "Security Operations",
    },
    {
        "label": "Application Failure",
        "description": (
            "A business application, website, API, payment system, email "
            "application, or internal software is crashing, returning errors, "
            "timing out, or not responding correctly."
        ),
        "team": "Application Support",
    },
    {
        "label": "Infrastructure Alert",
        "description": (
            "A server, database, cloud infrastructure, storage system, "
            "backup process, CPU, memory, disk, or other infrastructure "
            "component has a technical problem or alert."
        ),
        "team": "Infrastructure Team",
    },
    {
        "label": "Service Request",
        "description": (
            "A standard employee or customer service request such as "
            "account creation, onboarding, password reset, software setup, "
            "access permission, license request, or configuration change."
        ),
        "team": "Service Desk",
    },
]


CATEGORY_TEXTS = [
    entry["description"] for entry in CATEGORY_ENTRIES
]

CATEGORY_LABELS = [
    entry["label"] for entry in CATEGORY_ENTRIES
]

CATEGORY_TEAMS = [
    entry["team"] for entry in CATEGORY_ENTRIES
]


# ============================================================
# LOCAL AI MODELS
# ============================================================

# These models run locally and do not require an API key.
# One model creates semantic embeddings for incident descriptions.
# The other creates concise summaries for support operators.

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)

CATEGORY_EMBEDDINGS = MODEL.encode(
    CATEGORY_TEXTS,
    normalize_embeddings=True
)

SUMMARY_MODEL_NAME = "google/flan-t5-small"

try:
    SUMMARY_GENERATOR = pipeline(
        "text2text-generation",
        model=SUMMARY_MODEL_NAME,
        tokenizer=SUMMARY_MODEL_NAME,
        device=-1,
    )
except Exception:
    SUMMARY_GENERATOR = None


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Clean incident text before processing.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# INCIDENT CLASSIFICATION
# ============================================================

def classify_incident(
    title: str,
    description: str
) -> Dict[str, object]:
    """
    Classify an incident using semantic similarity.

    The incident title and description are converted into
    an embedding and compared with predefined enterprise
    incident categories.
    """

    combined_text = f"{title}. {description}"

    normalized = normalize_text(combined_text)

    if not normalized:
        return {
            "category": "Service Request",
            "suggested_team": "Service Desk",
            "confidence": 0.0,
            "confidence_percentage": 0.0,
            "reason": "No incident text was provided.",
        }

    # Generate semantic embedding for the incident.
    incident_embedding = MODEL.encode(
        [normalized],
        normalize_embeddings=True
    )

    # Compare incident meaning with category meanings.
    similarity_scores = cosine_similarity(
        incident_embedding,
        CATEGORY_EMBEDDINGS
    )[0]

    best_index = int(similarity_scores.argmax())

    category = CATEGORY_LABELS[best_index]
    suggested_team = CATEGORY_TEAMS[best_index]

    confidence = float(similarity_scores[best_index])

    # Convert similarity into a percentage.
    confidence_percentage = round(
        max(0.0, min(confidence, 1.0)) * 100,
        1
    )

    reason = (
        f"Semantic similarity matched the incident with "
        f"the '{category}' category."
    )

    return {
        "category": category,
        "suggested_team": suggested_team,
        "confidence": round(confidence, 4),
        "confidence_percentage": confidence_percentage,
        "reason": reason,
    }


# ============================================================
# ENTERPRISE PRIORITY ENGINE
# ============================================================

def estimate_priority(
    title: str,
    description: str,
    category: str
) -> Dict[str, object]:
    """
    Determine incident priority using explainable enterprise
    business rules.

    AI performs semantic classification.
    Business rules determine operational severity.

    IMPORTANT:
    Both title AND description are checked for priority signals.
    """

    # FIX:
    # Previously only description was checked.
    # Now title + description are checked.
    text = normalize_text(
        f"{title}. {description}"
    )

    # --------------------------------------------------------
    # CRITICAL SIGNALS
    # --------------------------------------------------------

    critical_signals = [
        "security breach",
        "data breach",
        "ransomware",
        "malware",
        "hacked",
        "unauthorized access",
        "unauthorized login",
        "unauthorized activity",
        "compromised",
        "compromised account",
        "compromised credentials",
        "credential theft",
        "stolen credentials",
        "cyber attack",
        "cyberattack",
        "security incident",
    ]

    # --------------------------------------------------------
    # HIGH SIGNALS
    # --------------------------------------------------------

    high_signals = [
        "outage",
        "system down",
        "website down",
        "server down",
        "cannot access",
        "unable to access",
        "complete failure",
        "service unavailable",
        "production down",
        "major failure",
    ]

    # --------------------------------------------------------
    # MEDIUM SIGNALS
    # --------------------------------------------------------

    medium_signals = [
        "slow",
        "latency",
        "timeout",
        "error",
        "degraded",
        "intermittent",
    ]

    matched_critical: List[str] = []
    matched_high: List[str] = []
    matched_medium: List[str] = []

    # --------------------------------------------------------
    # CHECK CRITICAL SIGNALS
    # --------------------------------------------------------

    for signal in critical_signals:
        if signal in text:
            matched_critical.append(signal)

    if matched_critical:
        return {
            "priority": "Critical",
            "reason": (
                "Critical security or business-risk indicators detected."
            ),
            "signals": matched_critical,
        }

    # --------------------------------------------------------
    # CHECK HIGH SIGNALS
    # --------------------------------------------------------

    for signal in high_signals:
        if signal in text:
            matched_high.append(signal)

    if matched_high:
        return {
            "priority": "High",
            "reason": (
                "Major availability or service-impact indicators detected."
            ),
            "signals": matched_high,
        }

    # --------------------------------------------------------
    # SERVICE REQUEST
    # --------------------------------------------------------

    if category == "Service Request":
        return {
            "priority": "Low",
            "reason": (
                "Standard service request with no major "
                "incident indicators."
            ),
            "signals": [],
        }

    # --------------------------------------------------------
    # MEDIUM SIGNALS
    # --------------------------------------------------------

    for signal in medium_signals:
        if signal in text:
            matched_medium.append(signal)

    if matched_medium:
        return {
            "priority": "Medium",
            "reason": (
                "Service degradation or technical issue detected."
            ),
            "signals": matched_medium,
        }

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return {
        "priority": "Medium",
        "reason": (
            "No critical or high-severity indicators detected."
        ),
        "signals": [],
    }


# ============================================================
# SUMMARY GENERATION
# ============================================================

def summarize_incident(description: str) -> str:
    """
    Generate a concise extractive summary.

    This deliberately avoids depending on an external API.
    """

    text = description.strip()

    if not text:
        return "No description provided."

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return text

    summary = sentences[0]
    lower_summary = summary.lower()

    separators = [
        " because ",
        " when ",
        " after ",
        " while ",
        " although ",
    ]

    for separator in separators:
        if separator in lower_summary:
            parts = re.split(
                re.escape(separator),
                summary,
                flags=re.IGNORECASE
            )
            if parts and len(parts[0].strip()) >= 30:
                summary = parts[0].strip()
                break

    words = summary.split()
    if len(words) > 25:
        summary = " ".join(words[:25]) + "..."

    summary = summary.rstrip(".!?")
    return summary + "."


def _looks_like_repetition(text: str) -> bool:
    """Detect obvious repeated-word output from a weak local generator."""

    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower().strip()
    if not cleaned:
        return True

    words = cleaned.split()
    if len(words) <= 8:
        return False

    unique_ratio = len(set(words)) / len(words)
    repeated_phrase = any(
        words[i] == words[i + 1] for i in range(len(words) - 1)
    )

    return unique_ratio < 0.5 or repeated_phrase


def generate_summary(title: str, description: str) -> str:
    """
    Use a local Hugging Face model to generate an incident summary.

    If the generated text is repetitive or poor quality,
    fall back to a deterministic extractive summary.
    """

    text = (description or "").strip()
    if not text:
        return "No description provided."

    fallback_summary = summarize_incident(description)

    if SUMMARY_GENERATOR is None:
        return fallback_summary

    prompt = (
        "Write a concise 1-sentence incident summary for an IT support team. "
        f"Title: {title}. Description: {description}"
    )

    try:
        result = SUMMARY_GENERATOR(
            prompt,
            max_length=60,
            min_length=15,
            num_beams=4,
            do_sample=False,
            truncation=True,
        )

        generated = result[0].get("generated_text", "").strip()
        if not generated:
            return fallback_summary

        generated = re.sub(r"\s+", " ", generated).strip()

        if generated.lower().startswith("summary:"):
            generated = generated.split(":", 1)[1].strip()

        if len(generated) < 20:
            return fallback_summary

        if _looks_like_repetition(generated):
            return fallback_summary

        return generated.rstrip(".!?") + "."

    except Exception:
        return fallback_summary


# ============================================================
# COMPLETE AI ANALYSIS
# ============================================================

def analyze_incident(
    title: str,
    description: str
) -> Dict[str, object]:
    """
    Run the complete intelligence pipeline.

    1. Semantic AI classification
    2. Enterprise priority analysis
    3. Incident summarization
    """

    # --------------------------------------------------------
    # STEP 1: AI CLASSIFICATION
    # --------------------------------------------------------

    classification = classify_incident(
        title,
        description
    )

    # --------------------------------------------------------
    # STEP 2: PRIORITY ANALYSIS
    # --------------------------------------------------------

    priority_result = estimate_priority(
        title,
        description,
        classification["category"]
    )

    # --------------------------------------------------------
    # STEP 3: LOCAL GENERATIVE SUMMARY
    # --------------------------------------------------------

    summary = generate_summary(
        title,
        description,
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "category": classification["category"],
        "suggested_team": classification["suggested_team"],
        "confidence": classification["confidence"],
        "confidence_percentage": classification["confidence_percentage"],
        "classification_reason": classification["reason"],
        "priority": priority_result["priority"],
        "priority_reason": priority_result["reason"],
        "priority_signals": priority_result["signals"],
        "summary": summary,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    title = "unauthorized access in the website"

    description = "someone accessed our website"

    result = analyze_incident(
        title,
        description
    )

    print("\n========================================")
    print("AI INCIDENT ANALYSIS")
    print("========================================")

    print(f"Category          : {result['category']}")
    print(f"Suggested Team    : {result['suggested_team']}")
    print(f"Confidence        : {result['confidence_percentage']}%")

    print(f"Priority          : {result['priority']}")

    print(
        f"Priority Reason   : "
        f"{result['priority_reason']}"
    )

    print(
        f"Detected Signals  : "
        f"{result['priority_signals']}"
    )

    print(
        f"AI Summary        : "
        f"{result['summary']}"
    )

    print("========================================")