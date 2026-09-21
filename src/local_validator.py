import cv2
import uuid
from datetime import datetime
import ollama
from src.tools import trigger_alert

class LocalEdgeValidator:
    def __init__(self, model_name: str = "llama3.2-vision"):
        """
        Initializes the local vision validator.
        Can be set to 'llama3.2-vision' or 'moondream'.
        """
        self.model_name = model_name
        print(f"[*] Initialized LocalEdgeValidator using model: {self.model_name}")

    def verify_fall_event(self, frame_bgr, image_save_path: str = None) -> dict:
        """
        Takes an OpenCV BGR frame, performs local VLM inference via Ollama,
        and returns a standard record dictionary compliant with the HITL review queue format.
        """
        alert_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        mime_type = "image/jpeg"
        
        success, encoded_img = cv2.imencode(".jpg", frame_bgr)
        if not success:
            error_msg = "Frame encoding failed for local validator."
            print(f"[Error] {error_msg}")
            return self._build_error_record(alert_id, timestamp, mime_type, image_save_path, error_msg)

        image_bytes = encoded_img.tobytes()

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Analyze this surveillance frame. Is there a person who has "
                            "fallen down, slipped, or lying flat on the ground in distress? "
                            "Your response must start with 'YES:' or 'NO:', followed by a comprehensive, detailed "
                            "explanation of what you observe in the image, why it indicates a fall or safe posture, "
                            "and your assessed risk severity."
                        ),
                        "images": [image_bytes]
                    }
                ]
            )
            
            message_obj = response.get("message", {})
            raw_text = message_obj.get("content", "").strip()
            print(f"[Local VLM Verdict]: {raw_text}")
            
            # Determine if an alert is suggested based on whether the text starts with 'YES'
            is_suggested_alert = raw_text.upper().startswith("YES")
            
            # Build the suggested action structure
            severity = "High" if is_suggested_alert else "Low"
            suggested_action = {
                "tool_name": "local_vlm_verify",
                "reason": raw_text,
                "severity": severity
            }
            
            # Assemble the complete record dictionary
            record = {
                "alert_id": alert_id,
                "timestamp": timestamp,
                "mime_type": mime_type,
                "image_path": image_save_path,  
                "ai_suggested_alert": is_suggested_alert,
                "suggested_action": suggested_action,
                "gemini_response": {
                    "text": raw_text,
                    "function_calls": []  # Local VLM can keep function calls empty if unsupported
                },
                "hitl_status": "PENDING" if is_suggested_alert else "AUTO_PASSED",
                "human_review": {
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "decision": None,  # Will store "CONFIRMED" or "DISMISSED"
                    "notes": None
                },
                "error": None
            }
            
            # Trigger the alert action only if an alert is suggested
            if is_suggested_alert:
                trigger_alert(
                    reason=raw_text,
                    severity="High"
                )
                # Optionally append the triggered action into function_calls for parity
                record["gemini_response"]["function_calls"].append({
                    "name": "trigger_alert",
                    "arguments": {
                        "reason": raw_text,
                        "severity": "High"
                    }
                })

            return record
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Error] Local Ollama inference failed: {error_msg}")
            return self._build_error_record(alert_id, timestamp, mime_type, image_save_path, error_msg)

    def _build_error_record(self, alert_id, timestamp, mime_type, image_path, error_log) -> dict:
        """Helper function: Returns a compliant error record structure when a failure occurs."""
        return {
            "alert_id": alert_id,
            "timestamp": timestamp,
            "mime_type": mime_type,
            "image_path": image_path,  
            "ai_suggested_alert": False,
            "suggested_action": {
                "tool_name": "error_handler",
                "reason": "Inference failed due to system error.",
                "severity": "Low"
            },
            "gemini_response": {
                "text": "",
                "function_calls": []
            },
            "hitl_status": "ERROR",
            "human_review": {
                "reviewed_by": None,
                "reviewed_at": None,
                "decision": None,
                "notes": None
            },
            "error": error_log
        }