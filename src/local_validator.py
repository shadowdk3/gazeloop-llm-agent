import cv2
import uuid
from datetime import datetime
import ollama
from src.tools import trigger_alert
import os

from src.rag.retriever import RAGRetriever

classification_prompt = """
Is there clear visual evidence that a person is falling
or has fallen?

Answer only YES or NO.
"""

description_prompt = """
Analyze this surveillance image for a possible fall.

Do not guess or speculate.

Return ONLY JSON:

{
  "posture": "walking | standing | sitting | lying | falling",
  "body_contact_with_ground": true | false,
  "movement": "normal | unstable | falling | unknown",
  "fall_detected": true | false,
  "confidence": 0.0
}

A fall should only be detected if there is visible evidence that
the person is falling or has fallen. A person walking with bent
legs or outstretched arms is NOT sufficient evidence of a fall.
"""

class LocalEdgeValidator:
    def __init__(self, model_name: str = "llama3.2-vision"):
        """
        Initializes the local vision validator.
        Can be set to 'llama3.2-vision' or 'moondream'.
        """
        self.ollama_client = ollama.Client(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        self.model_name = model_name
        
        self.retriever = RAGRetriever()
        
        print(f"[*] Initialized LocalEdgeValidator using model: {self.model_name}")
        print( f"[*] Ollama host: " f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}" )
        
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
            # First call: Fall classification
            verdict_response = self.ollama_client.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": (classification_prompt),
                    "images": [image_bytes]
                }]
            )

            verdict = verdict_response.message.content.strip().upper()

            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": (description_prompt),
                        "images": [image_bytes]
                    }
                ]
            )
            
            message_obj = response.get("message", {})
            raw_text = message_obj.get("content", "").strip()
            print(f"[Local VLM Verdict]: {raw_text}")
            
            # Convert the VLM description into an embedding and retrieve matching guidelines
            embedding_chuck= self.retriever.retrieve(raw_text, top_k=3)

            # Determine if an alert is suggested based on whether the text starts with 'YES'
            is_suggested_alert = verdict.upper().startswith("YES")
            
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
                
                # Original VLM output
                "gemini_response": {
                    "text": raw_text,
                    "function_calls": []  # Local VLM can keep function calls empty if unsupported
                },
                
                # RAG results
                "rag": {
                    "query": raw_text,
                    "retrieved_chunks": [
                        {
                            "document_name": chunk.document_name,
                            "section": chunk.section,
                            "content": chunk.content,
                            "similarity": chunk.similarity
                        }
                        for chunk in embedding_chuck
                    ],
                    "llm_analysis": raw_text  # Fill after sending the RAG prompt to the LLM
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
            
            # RAG results
            "rag": {
                "query": None,
                "retrieved_chunks": [],
                "llm_analysis": None
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