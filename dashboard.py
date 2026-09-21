import os
import json
import streamlit as st
from datetime import datetime

# Set page title, icon, and layout configuration
st.set_page_config(
    page_title="AI Vision HITL Alert Center",
    page_icon="🚨",
    layout="wide"
)

JSON_PATH = "hitl_alerts_queue.json"

def load_hitl_queue():
    """Loads the HITL event database/log file."""
    if not os.path.exists(JSON_PATH):
        return []
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to read queue: {e}")
        return []

def update_alert_decision(alert_id: str, decision: str, operator_name: str, notes: str = ""):
    """Updates the human review decision and writes it back to the JSON log."""
    if not os.path.exists(JSON_PATH):
        return
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    for record in records:
        if record.get("alert_id") == alert_id:
            record["hitl_status"] = decision
            record["human_review"] = {
                "reviewed_by": operator_name,
                "reviewed_at": datetime.now().isoformat(),
                "decision": decision,
                "notes": notes
            }
            break

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# --- Sidebar Configuration ---
st.sidebar.title("Control Panel")
operator = st.sidebar.text_input("Operator Name / ID", value="Operator_01")
filter_status = st.sidebar.selectbox("Filter Status", ["PENDING", "CONFIRMED", "DISMISSED", "ALL"])

if st.sidebar.button("Refresh Data"):
    st.rerun()

st.title("Intelligent Vision Security - Human-in-the-Loop (HITL) Alert Center")
st.caption("An integrated early-warning review system combining YOLO real-time perception with Gemini multimodal reasoning.")

records = load_hitl_queue()

# Filter records based on selected status
if filter_status != "ALL":
    filtered_records = [r for r in records if r.get("hitl_status") == filter_status]
else:
    filtered_records = records

# Calculate summary metrics
pending_count = sum(1 for r in records if r.get("hitl_status") == "PENDING")
confirmed_count = sum(1 for r in records if r.get("hitl_status") == "CONFIRMED")
dismissed_count = sum(1 for r in records if r.get("hitl_status") == "DISMISSED")

col1, col2, col3 = st.columns(3)
col1.metric("Pending Alerts (PENDING)", pending_count, delta_color="inverse")
col2.metric("Confirmed Incidents (CONFIRMED)", confirmed_count)
col3.metric("Dismissed False Alarms (DISMISSED)", dismissed_count)

st.divider()

if not filtered_records:
    st.info(f"No records found with status: 【{filter_status}】.")
else:
    # Display in reverse chronological order (newest first)
    for record in reversed(filtered_records):
        alert_id = record.get("alert_id")
        status = record.get("hitl_status", "UNKNOWN")
        timestamp = record.get("timestamp", "")
        img_path = record.get("image_path", "")
        suggested_action = record.get("suggested_action") or {}
        gemini_resp = record.get("gemini_response", {})

        # Card container border color styling
        border_color = "red" if status == "PENDING" else ("green" if status == "CONFIRMED" else "gray")

        with st.container(border=True):
            st.markdown(f"#### Alert ID: `{alert_id}` | Status: **:{border_color}[{status}]**")
            st.caption(f"Timestamp: {timestamp}")

            c1, c2 = st.columns([1, 1])

            with c1:
                # Display the captured incident image frame
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=f"Trigger Frame: {img_path}")
                else:
                    st.warning("Unable to load image file or path does not exist.")

            with c2:
                st.subheader("AI Reasoning Diagnostic Report")
                
                # Extract suggestion parameters
                reason = suggested_action.get("reason", "No detailed reason provided")
                severity = suggested_action.get("severity", "Medium")
                
                st.write(f"**Suggested Action:** `{suggested_action.get('tool_name', 'N/A')}`")
                st.write(f"**Risk Level (Severity):** `{severity}`")
                st.write(f"**Analysis Reasoning:** {reason}")

                # Display Gemini raw function calls or text output
                if gemini_resp.get("function_calls"):
                    st.json(gemini_resp["function_calls"])
                elif gemini_resp.get("text"):
                    st.info(f"LLM Text Output: {gemini_resp['text']}")

                st.divider()

                # Provide human review buttons if status is PENDING
                if status == "PENDING":
                    st.subheader("Human Verification Controls")
                    notes = st.text_input("Review Notes (Optional)", key=f"notes_{alert_id}")
                    
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        if st.button("Confirm True Fall (Confirm)", key=f"conf_{alert_id}", type="primary", use_container_width=True):
                            update_alert_decision(alert_id, "CONFIRMED", operator, notes)
                            st.success(f"Alert {alert_id} confirmed. Emergency protocol triggered!")
                            st.rerun()

                    with btn_col2:
                        if st.button("Dismiss False Alarm (Dismiss)", key=f"dism_{alert_id}", use_container_width=True):
                            update_alert_decision(alert_id, "DISMISSED", operator, notes)
                            st.info(f"Alert {alert_id} dismissed and archived to false-positive dataset.")
                            st.rerun()

                else:
                    # Display historical review metadata
                    review = record.get("human_review", {})
                    st.write(f"**Reviewed By:** {review.get('reviewed_by')}")
                    st.write(f"**Reviewed At:** {review.get('reviewed_at')}")
                    if review.get("notes"):
                        st.write(f"**Notes:** {review.get('notes')}")