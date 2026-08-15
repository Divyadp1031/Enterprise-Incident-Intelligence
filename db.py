import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_FILE = BASE_DIR / "incidents.db"


# ============================================================
# DATABASE SCHEMA
# ============================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    title TEXT NOT NULL,

    description TEXT NOT NULL,

    category TEXT NOT NULL,

    priority TEXT NOT NULL,

    suggested_team TEXT NOT NULL,

    summary TEXT NOT NULL,

    confidence REAL DEFAULT 0.0,

    confidence_percentage REAL DEFAULT 0.0,

    classification_reason TEXT,

    priority_reason TEXT,

    priority_signals TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Create a connection to the SQLite database.

    SQLite is used because it is:
    - free
    - open-source
    - locally runnable
    - persistent
    - easy to demonstrate
    """

    connection = sqlite3.connect(DB_FILE)

    # Return rows as dictionary-like objects.
    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db() -> None:
    """
    Create the incidents table if it does not already exist.
    """

    connection = get_connection()

    try:
        connection.execute(CREATE_TABLE_SQL)
        connection.commit()

    finally:
        connection.close()


# ============================================================
# INSERT INCIDENT
# ============================================================

def insert_incident(
    title: str,
    description: str,
    category: str,
    priority: str,
    suggested_team: str,
    summary: str,
    confidence: float,
    confidence_percentage: float,
    classification_reason: str,
    priority_reason: str,
    priority_signals: str,
) -> int:
    """
    Store a complete analyzed incident.

    The AI output is persisted together with the original
    incident so the result remains available after restart.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO incidents (
                title,
                description,
                category,
                priority,
                suggested_team,
                summary,
                confidence,
                confidence_percentage,
                classification_reason,
                priority_reason,
                priority_signals
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                description,
                category,
                priority,
                suggested_team,
                summary,
                confidence,
                confidence_percentage,
                classification_reason,
                priority_reason,
                priority_signals,
            ),
        )

        connection.commit()

        return int(cursor.lastrowid)

    finally:
        connection.close()


# ============================================================
# FETCH ALL INCIDENTS
# ============================================================

def fetch_incidents() -> List[Dict]:
    """
    Retrieve all incidents.

    Newest incidents appear first.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM incidents
            ORDER BY created_at DESC, id DESC
            """
        )

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


# ============================================================
# FETCH ONE INCIDENT
# ============================================================

def fetch_incident(
    incident_id: int
) -> Optional[Dict]:
    """
    Retrieve a single incident by ID.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM incidents
            WHERE id = ?
            """,
            (incident_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        connection.close()


# ============================================================
# INCIDENT STATISTICS
# ============================================================

def fetch_statistics() -> Dict:
    """
    Return dashboard statistics.

    These values are calculated from persisted database
    records rather than hard-coded.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # Total incidents
        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM incidents
            """
        )

        total = cursor.fetchone()["total"]

        # Critical incidents
        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM incidents
            WHERE priority = 'Critical'
            """
        )

        critical = cursor.fetchone()["total"]

        # High-priority incidents
        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM incidents
            WHERE priority = 'High'
            """
        )

        high = cursor.fetchone()["total"]

        # Medium-priority incidents
        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM incidents
            WHERE priority = 'Medium'
            """
        )

        medium = cursor.fetchone()["total"]

        # Low-priority incidents
        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM incidents
            WHERE priority = 'Low'
            """
        )

        low = cursor.fetchone()["total"]

        # Average AI confidence
        cursor.execute(
            """
            SELECT AVG(confidence_percentage) AS average_confidence
            FROM incidents
            """
        )

        average_confidence = cursor.fetchone()["average_confidence"]

        if average_confidence is None:
            average_confidence = 0.0

        return {
            "total": total,
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "average_confidence": round(
                float(average_confidence),
                1
            ),
        }

    finally:
        connection.close()