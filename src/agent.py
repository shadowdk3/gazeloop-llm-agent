from pathlib import Path
from typing import List, Union
from google import genai
from google.genai import types
from src.tools import trigger_alert
import src.methods
import cv2
import time
from google.genai.errors import ServerError
import logging
from ultralytics import YOLO
import uuid
import json
from datetime import datetime

# Define alert keywords separately
ALERT_KEYWORDS = [
    "alert",
    "abnormal",
    "fire",
    "fall",
]
        
MAX_RETRIES = 5
DELAY_SECOND = 5
INTERVAL_SECOND = 5.0

class VisionAgent:
    def __init__(self, image_paths: List[Union[str, Path]] = None):
        self.client = genai.Client()
        self.image_paths = [Path(p) for p in (image_paths or [])]
        self.config = types. GenerateContentConfig(
            tools=[trigger_alert],
            temperature=0.2,
            system_instruction=(
                "You are a strict and attentive intelligent vision guard Agent. "
                "You will receive designated surveillance frame images. "
                "If there is any abnormal situation in the images (e.g., unauthorized intruder, fire, fall), "
                "you must actively call the `trigger_alert` tool. "
                "If everything is normal, do not call any tool, just reply briefly: 'Current status is safe and normal.'"
            ),
        )
        
        self.model = YOLO("yolo11m.pt")

    def _inspect_frame_data(self, encoded_img, mime_type) -> bool:
        """
        Sends an image frame to the Gemini model for anomaly inspection with retry handling.
        Parses tool calls or text responses to determine if a HITL alert should be raised.
        """
        # Wrap raw image bytes into a GenAI Part object for multimodal ingestion
        img_part = types.Part.from_bytes(
            data=encoded_img.tobytes(), mime_type=mime_type
        )

        response = None
        error_log = None
    
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Sending to Gemini for analysis...")
                
                response = self.client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=[
                        img_part,
                        f"Please inspect this frame for any abnormal situations that require an alert.",
                    ],
                    config=self.config,
                )
                break  # If successful, exit the retry loop
            except ServerError as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Server overloaded (503). Retrying in {DELAY_SECOND} seconds...")
                    time.sleep(DELAY_SECOND)
                else:
                    print("Max retries reached. API is still unavailable.")
                    raise e
                
        is_suggested_alert = False
        suggested_action = None
        function_calls_data = []
        
        # Check if the model triggered a tool call (e.g., trigger_alert)
        if response.function_calls:
            for call in response.function_calls:
                args = dict(call.args) if call.args else {}

                if call.name == "trigger_alert":
                    is_suggested_alert = True
                    
                    suggested_action = {
                        "tool_name": "trigger_alert",
                        "reason": args.get("reason", "Unknown anomaly"),
                        "severity": args.get("severity", "Medium")
                    }
                    print(f"[Gemini Suggestion]: Recommended Alert -> Reason: {suggested_action['reason']}")
                    
                function_calls_data.append({
                    "name": call.name,
                    "args": args
                })

        # Fallback to analyzing raw text response if no tool calls were made
        raw_text = response.text if (response and hasattr(response, "text")) else ""
        if not response.function_calls and raw_text:
            print(f"[Agent Reasoning Result]: {raw_text}")
            if any(word in raw_text.lower() for word in ALERT_KEYWORDS):
                is_suggested_alert = True
                suggested_action = {
                    "tool_name": "keyword_match",
                    "reason": raw_text,
                    "severity": "Medium"
                }
            
        # Build the standard Human-in-the-Loop (HITL) queue record dictionary
        alert_id = str(uuid.uuid4())
        record = {
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat(),
            "mime_type": mime_type,
            "image_path": None,  
            "ai_suggested_alert": is_suggested_alert,
            "suggested_action": suggested_action,
            "gemini_response": {
                "text": raw_text,
                "function_calls": function_calls_data
            },
            
            # HITL State Tracking: PENDING requires operator review; AUTO_PASSED is skipped
            "hitl_status": "PENDING" if is_suggested_alert else "AUTO_PASSED",
            "human_review": {
                "reviewed_by": None,
                "reviewed_at": None,
                "decision": None,  # Will store "CONFIRMED" or "DISMISSED"
                "notes": None
            },
            "error": error_log
        }
        
        # if is_suggested_alert:
            # self._notify_operator_for_review(record)
        
        return record       
        
    def _save_to_json(self, data: dict, output_filepath: str = "hitl_alerts_queue.json"):
        try:
            try:
                with open(output_filepath, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []

            logs.append(data)

            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
            print(f"[HITL Log Saved]: Alert ID {data['alert_id']} stored with status: {data['hitl_status']}")
        except Exception as e:
            print(f"Failed to save JSON log: {e}")
        
    def _draw_status(self, frame, is_alert: bool) -> str:
        # Display the image locally using OpenCV
        status_text = "abnormal" if is_alert else "normal"
        # Choose text color: Red if alert triggered, Green if safe/normal
        text_color = (0, 0, 255) if is_alert else (0, 255, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        margin = 15

        (text_w, text_h), baseline = cv2.getTextSize(
            status_text, font, font_scale, thickness
        )
        
        h, w, _ = frame.shape
        
        # Calculate right bottom coordinates
        x = w - text_w - margin
        y = h - margin
        
        cv2.putText(
            frame, status_text, (x, y), font, font_scale, text_color, thickness
        )
    
        return status_text
    
    def save_analyze_result(self, frame, status_text, output_path):
        cv2.imwrite(output_path, frame)
        logging.info(f"[Result Saved]: {output_path} | Status: {status_text}")

    def analyze_images(self):
        for i, img_path in enumerate(self.image_paths):
            if not img_path.exists():
                print(f"Image file not found: {img_path}")
                continue
        
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            mime_type = (
                "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            )
            ext = ".png" if mime_type == "image/png" else ".jpg"

            success, encoded_img = cv2.imencode(ext, frame)
            if not success:
                continue
  
            # Check if 5 seconds have passed since the last analysis
            if i > 0:
                print(f"Waiting {INTERVAL_SECOND} seconds before next image...")
                time.sleep(INTERVAL_SECOND)
                            
            logging.info(f"--- Analyzing frame at {time.strftime('%H:%M:%S')} ---")
            is_alert = self._inspect_frame_data(encoded_img, mime_type)
            status_text = self._draw_status(frame, is_alert)
    
            # save result
            output_path = f"data/output_{img_path.name}"
            self.save_analyze_result(frame, status_text, output_path)
       
    def hybrid_analyze_stream(self, source: Union[int, str]):
        
        # stream handler for both camera / video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Cannot open source: {source}")
            return
        
        # Automatically get the video's actual FPS (fallback to 30 if unavailable or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        frame_idx = 0
        last_analysis_time = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- TIER 1: YOLO inference (classes=[0] filters for 'person' only) ---
            results = self.model(frame, classes=[0], verbose=False)
            
            is_potential_fall = False
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates [x1, y1, x2, y2]
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords[:4])
                    conf = float(box.conf[0].cpu())
                    
                    # Filter out low-confidence detections
                    if conf < 0.4:
                        continue
                        
                    # Run Aspect Ratio Check on the Bounding Box
                    is_potential_fall = src.methods.check_box_aspect_ratio(x1, y1, x2, y2)
                    
                    # Draw bounding box (Green if normal, Red if potential fall)
                    box_color = (0, 0, 255) if is_potential_fall else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    
            current_time = time.time()
            # --- TIER 2: Gemini LLM Verification ---
            if is_potential_fall:
                if (current_time - last_analysis_time) >= INTERVAL_SECOND:
                    print("--- Tier 1 Triggered (YOLO): Potential horizontal fall detected! Engaging LLM Tier 2 ---")
                    
                    # Encode current frame to JPEG format for API transmission
                    success, encoded_img = cv2.imencode(".jpg", frame)
                    
                    if not success:
                        continue
        
                    logging.info(f"--- Analyzing frame at {time.strftime('%H:%M:%S')} ---")
                    hitl_record = self._inspect_frame_data(encoded_img, mime_type="image/jpeg")
                    last_analysis_time = current_time
                    
                    # Retrieve record identifiers
                    if hitl_record.get("ai_suggested_alert"):
                        alert_id = hitl_record["alert_id"]
                        is_alert = hitl_record["ai_suggested_alert"]
                        file_text = f"HITL_PENDING_ID_{alert_id[:8]}"
                    
                    # save result
                    Path("data/snapshots").mkdir(parents=True, exist_ok=True)
                    output_path = f"data/snapshots/{file_text}.jpg"
                    hitl_record["image_path"] = output_path
                    
                    status_text = self._draw_status(frame, is_alert)    
                    self._save_to_json(hitl_record)
                    self.save_analyze_result(frame, status_text, output_path)     
            
            frame_idx += 1
            
        cap.release()