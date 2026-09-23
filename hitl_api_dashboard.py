import os
import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000/api/alerts"

# Set page title, icon, and layout configuration
st.set_page_config(
    page_title="AI Vision HITL Alert Center",
    page_icon="🚨",
    layout="wide"
)

# fetches live audit records from your FastAPI server (http://localhost:8000/api/alerts)
def load_hitl_queue(status_filter: str = "ALL"):
    """Fetches records from the FastAPI backend connected to PostgreSQL."""
    try:
        response = requests.get(API_BASE_URL, params={"status": status_filter})
        if response.status_code == 200:
            return response.json()
        st.error(f"API Error: {response.text}")
        return []
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")
        return []

# When an operator clicks Confirm or Dismiss, it sends an HTTP PATCH request with the operator's decision and notes to update PostgreSQL securely.
def update_alert_decision(alert_id: str, decision: str, operator_name: str, notes: str = ""):
    """Sends patch request to FastAPI to update status in PostgreSQL."""
    url = f"{API_BASE_URL}/{alert_id}/review"
    payload = {
        "decision": decision,
        "operator_name": operator_name,
        "notes": notes
    }
    try:
        response = requests.patch(url, json=payload)
        if response.status_code != 200:
            st.error(f"Failed to update: {response.text}")
    except Exception as e:
        st.error(f"Connection error during update: {e}")

# --- Sidebar Configuration ---
st.sidebar.title("Control Panel")
operator = st.sidebar.text_input("Operator Name / ID", value="Operator_01")
filter_status = st.sidebar.selectbox("Filter Status", ["PENDING", "CONFIRMED", "DISMISSED", "ALL"])

if st.sidebar.button("Refresh Data"):
    st.rerun()

st.title("Intelligent Vision Security - Human-in-the-Loop (HITL) Alert Center")
st.caption("An integrated early-warning review system backed by PostgreSQL cryptographic audit logs.")

# Fetch records from API based on filter
records = load_hitl_queue(filter_status if filter_status != "ALL" else None)

# Calculate summary metrics (fetch ALL for metrics summary)
all_records = load_hitl_queue("ALL")
pending_count = sum(1 for r in all_records if r.get("hitl_status") == "PENDING")
confirmed_count = sum(1 for r in all_records if r.get("hitl_status") == "CONFIRMED")
dismissed_count = sum(1 for r in all_records if r.get("hitl_status") == "DISMISSED")

col1, col2, col3 = st.columns(3)
col1.metric("Pending Alerts (PENDING)", pending_count, delta_color="inverse")
col2.metric("Confirmed Incidents (CONFIRMED)", confirmed_count)
col3.metric("Dismissed False Alarms (DISMISSED)", dismissed_count)

st.divider()

if not records:
    st.info(f"No records found with status: 【{filter_status}】.")
else:
    for record in records:  # API already returns desc order based on query
        alert_id = record.get("alert_id")
        status = record.get("hitl_status", "UNKNOWN")
        timestamp = record.get("timestamp", "")
        img_path = record.get("image_path", "")
        suggested_action = record.get("suggested_action") or {}
        gemini_resp = record.get("gemini_response", {})

        border_color = "red" if status == "PENDING" else ("green" if status == "CONFIRMED" else "gray")

        with st.container(border=True):
            st.markdown(f"#### Alert ID: `{alert_id}` | Status: **:{border_color}[{status}]**")
            st.caption(f"Timestamp: {timestamp} | Hash: `{record.get('current_hash', '')[:12]}...`")

            c1, c2 = st.columns([1, 1])

            with c1:
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=f"Trigger Frame: {img_path}")
                else:
                    st.warning("Unable to load image file or path does not exist.")

            with c2:
                st.subheader("AI Reasoning Diagnostic Report")
                
                reason = suggested_action.get("reason", "No detailed reason provided")
                severity = suggested_action.get("severity", "Medium")
                
                st.write(f"**Suggested Action:** `{suggested_action.get('tool_name', 'N/A')}`")
                st.write(f"**Risk Level (Severity):** `{severity}`")
                st.write(f"**Analysis Reasoning:** {reason}")

                if gemini_resp.get("function_calls"):
                    st.json(gemini_resp["function_calls"])
                elif gemini_resp.get("text"):
                    st.info(f"LLM Text Output: {gemini_resp['text']}")

                st.divider()

                if status == "PENDING":
                    st.subheader("Human Verification Controls")
                    notes = st.text_input("Review Notes (Optional)", key=f"notes_{alert_id}")
                    
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        if st.button("Confirm True Fall (Confirm)", key=f"conf_{alert_id}", type="primary", use_container_width=True):
                            update_alert_decision(alert_id, "CONFIRMED", operator, notes)
                            st.success(f"Alert {alert_id} confirmed!")
                            st.rerun()

                    with btn_col2:
                        if st.button("Dismiss False Alarm (Dismiss)", key=f"dism_{alert_id}", use_container_width=True):
                            update_alert_decision(alert_id, "DISMISSED", operator, notes)
                            st.info(f"Alert {alert_id} dismissed.")
                            st.rerun()
                else:
                    review = gemini_resp.get("human_review", {})
                    st.write(f"**Reviewed By:** {review.get('reviewed_by', 'N/A')}")
                    if review.get("notes"):
                        st.write(f"**Notes:** {review.get('notes')}")