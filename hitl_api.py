from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from typing import Optional, List, Dict, Any

app = FastAPI(title="Gazeloop HITL Audit API", version="1.0")

DB_CONFIG = {
    "host": "localhost",
    "database": "gazeloop",
    "user": "postgres",
    "password": "123456"
}

class ReviewRequest(BaseModel):
    decision: str  # e.g., "CONFIRMED" or "DISMISSED"
    operator_name: str
    notes: Optional[str] = ""

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# Fetching Alerts Endpoint (GET /api/alerts)
# 1. Queries your audit_logs table in PostgreSQL, sorting records in descending order (ORDER BY id DESC) 
# so the newest alerts appear at the top.
# 2. Accepts an optional query parameter (?status=PENDING). If provided, it filters rows by their HITL decision state 
# (e.g., PENDING, CONFIRMED, DISMISSED).
# 3. Maps raw database column indices into a clean, structured dictionary (alert_id, timestamp, image_path, 
# suggested_action, gemini_response, hitl_status, current_hash) that the frontend expects.
@app.get("/api/alerts", response_model=List[Dict[str, Any]])
def get_alerts(status: Optional[str] = None):
    """Fetch audit logs/alerts from PostgreSQL, optionally filtered by status."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if status and status != "ALL":
            cursor.execute(
                "SELECT id, timestamp, snapshot_path, model_metadata, llm_payload, hitl_decision, current_hash FROM audit_logs WHERE hitl_decision = %s ORDER BY id DESC",
                (status,)
            )
        else:
            cursor.execute(
                "SELECT id, timestamp, snapshot_path, model_metadata, llm_payload, hitl_decision, current_hash FROM audit_logs ORDER BY id DESC"
            )
        rows = cursor.fetchall()
        
        alerts = []
        for row in rows:
            alerts.append({
                "alert_id": str(row[0]),
                "timestamp": row[1].isoformat() if row[1] else "",
                "image_path": row[2],
                "suggested_action": row[3] if isinstance(row[3], dict) else {},
                "gemini_response": row[4] if isinstance(row[4], dict) else {},
                "hitl_status": row[5],
                "current_hash": row[6]
            })
        return alerts
    finally:
        cursor.close()
        conn.close()

# Updating Review Status Endpoint (PATCH /api/alerts/{alert_id}/review)
# 1. Existence Check: Verifies that the specific alert_id exists in the database. 
# If not, it raises an HTTP 404 Not Found error.
# 2. PostgreSQL JSONB Manipulation: Executes an UPDATE statement that changes the 
# hitl_decision column (e.g., to "CONFIRMED") and uses PostgreSQL's native jsonb_set function to 
# inject the operator's review details (reviewed_by and notes) directly into the existing llm_payload 
# JSON column without overwriting other data.
# 3. Transaction Safety: Uses conn.commit() to save changes permanently, or conn.rollback()
# if any error occurs, ensuring database integrity.
@app.patch("/api/alerts/{alert_id}/review")
def update_alert(alert_id: int, review: ReviewRequest):
    """Update the HITL decision and store operator notes in PostgreSQL JSONB/columns."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if record exists
        cursor.execute("SELECT id FROM audit_logs WHERE id = %s", (alert_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Alert not found")

        # Update decision
        cursor.execute(
            """
            UPDATE audit_logs 
            SET hitl_decision = %s,
                llm_payload = jsonb_set(llm_payload, '{human_review}', %s::jsonb, true)
            WHERE id = %s
            """,
            (
                review.decision,
                f'{{"reviewed_by": "{review.operator_name}", "notes": "{review.notes}"}}',
                alert_id
            )
        )
        conn.commit()
        return {"status": "success", "alert_id": alert_id, "new_decision": review.decision}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()