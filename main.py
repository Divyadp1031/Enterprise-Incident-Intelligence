from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db
from ai_engine import analyze_incident


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Enterprise Incident Intelligence",
    description=(
        "AI-powered enterprise incident classification, "
        "priority analysis, and incident intelligence."
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATIC FILES
# ============================================================

STATIC_DIR = BASE_DIR / "static"

STATIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# ============================================================
# APPLICATION STARTUP
# ============================================================

@app.on_event("startup")
def startup_event() -> None:
    """
    Initialize the persistent SQLite database
    when the application starts.
    """

    db.init_db()


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def index() -> HTMLResponse:
    """
    Serve the main frontend page.
    """

    html_path = BASE_DIR / "index.html"

    if not html_path.exists():
        return HTMLResponse(
            content="""
            <h1>Frontend not found</h1>
            <p>Please make sure index.html exists.</p>
            """,
            status_code=500,
        )

    html_content = html_path.read_text(
        encoding="utf-8"
    )

    return HTMLResponse(
        content=html_content
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check() -> JSONResponse:
    """
    Simple API health check.

    Useful when demonstrating the backend
    during the challenge.
    """

    return JSONResponse(
        {
            "status": "healthy",
            "application": "Enterprise Incident Intelligence",
            "ai_engine": "Sentence Transformer",
            "database": "SQLite",
        }
    )


# ============================================================
# CREATE INCIDENT
# ============================================================

@app.post("/api/incidents")
async def create_incident(
    request: Request
) -> JSONResponse:
    """
    Create and analyze a new enterprise incident.

    Processing pipeline:

        Browser
          ↓
        FastAPI
          ↓
        Sentence Transformer
          ↓
        Classification
          ↓
        Priority Engine
          ↓
        Summary
          ↓
        SQLite
    """

    # --------------------------------------------------------
    # Read request
    # --------------------------------------------------------

    try:
        payload = await request.json()

    except Exception:
        return JSONResponse(
            {
                "error": "Invalid JSON request."
            },
            status_code=400,
        )

    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

    title = str(
        payload.get("title", "")
    ).strip()

    description = str(
        payload.get("description", "")
    ).strip()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not title:
        return JSONResponse(
            {
                "error": "Incident title is required."
            },
            status_code=400,
        )

    if not description:
        return JSONResponse(
            {
                "error": "Incident description is required."
            },
            status_code=400,
        )

    if len(title) > 200:
        return JSONResponse(
            {
                "error": "Incident title is too long."
            },
            status_code=400,
        )

    if len(description) > 5000:
        return JSONResponse(
            {
                "error": "Incident description is too long."
            },
            status_code=400,
        )

    # --------------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------------

    try:
        analysis = analyze_incident(
            title=title,
            description=description,
        )

    except Exception as exc:
        return JSONResponse(
            {
                "error": "AI analysis failed.",
                "details": str(exc),
            },
            status_code=500,
        )

    # --------------------------------------------------------
    # Prepare priority signals
    # --------------------------------------------------------

    priority_signals = ", ".join(
        analysis.get(
            "priority_signals",
            []
        )
    )

    # --------------------------------------------------------
    # Persist complete result
    # --------------------------------------------------------

    incident_id = db.insert_incident(
        title=title,
        description=description,
        category=analysis["category"],
        priority=analysis["priority"],
        suggested_team=analysis["suggested_team"],
        summary=analysis["summary"],
        confidence=float(
            analysis["confidence"]
        ),
        confidence_percentage=float(
            analysis["confidence_percentage"]
        ),
        classification_reason=analysis[
            "classification_reason"
        ],
        priority_reason=analysis[
            "priority_reason"
        ],
        priority_signals=priority_signals,
    )

    # --------------------------------------------------------
    # Fetch persisted record
    # --------------------------------------------------------

    created_incident = db.fetch_incident(
        incident_id
    )

    if created_incident is None:
        return JSONResponse(
            {
                "error": "Incident was saved but could not be retrieved."
            },
            status_code=500,
        )

    return JSONResponse(
        created_incident,
        status_code=201,
    )


# ============================================================
# LIST INCIDENTS
# ============================================================

@app.get("/api/incidents")
def list_incidents() -> JSONResponse:
    """
    Return all persisted incidents.
    """

    records = db.fetch_incidents()

    return JSONResponse(
        records
    )


# ============================================================
# GET SINGLE INCIDENT
# ============================================================

@app.get("/api/incidents/{incident_id}")
def get_incident(
    incident_id: int
) -> JSONResponse:
    """
    Return a single incident by ID.
    """

    incident = db.fetch_incident(
        incident_id
    )

    if incident is None:
        return JSONResponse(
            {
                "error": "Incident not found."
            },
            status_code=404,
        )

    return JSONResponse(
        incident
    )


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

@app.get("/api/statistics")
def statistics() -> JSONResponse:
    """
    Return statistics calculated from persisted incidents.
    """

    statistics_data = db.fetch_statistics()

    return JSONResponse(
        statistics_data
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )