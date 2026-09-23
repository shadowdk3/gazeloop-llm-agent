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
from src.local_validator import LocalEdgeValidator
import asyncio
from src.audit_logger_pg import PGAuditLogger

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
        self.edge_validator = LocalEdgeValidator(model_name="moondream")
        self.queue = asyncio.Queue(maxsize=30)
        
        # Initialize PostgreSQL Cryptographic Audit Logger
        self.audit_logger = PGAuditLogger()
        
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
                    # hitl_record = self._inspect_frame_data(encoded_img, mime_type="image/jpeg")
                    hitl_record = self.edge_validator.verify_fall_event(frame)
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
        
    # --- Asynchronous Frame Producer: Continuous video stream capture ---
    async def frame_producer(self, source: Union[int, str]):
        """
        Continuously captures frames from a video stream (file or camera) 
        without blocking the main inference loop.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Unable to open video source: {source}")
            return

        print("[Producer] Starting real-time video stream capture...")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Handle backpressure: If the processing queue is full,
                # drop the oldest frame to ensure real-time latency and prevent memory bloat.
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                # Asynchronously push the newly captured frame into the queue
                await self.queue.put(frame)
                
                # Yield control back to the asyncio event loop briefly
                await asyncio.sleep(0.001)
        finally:
            cap.release()
            # Send a 'poison pill' (None) to signal the consumer 
            # that the video stream has finished.
            await self.queue.put(None)

    # --- Asynchronous Frame Consumer: Pipeline inference, event validation, and display ---
    async def frame_consumer(self):
        """
        Consumes frames from the queue asynchronously, runs fast YOLO screening,
        triggers deep verification (rate-limited), and displays the live stream via cv2.imshow.
        """
        last_alert_time = 0.0
        cooldown_period = 1.0  # Cooldown in seconds between expensive model verifications
        
        print("[Consumer] Starting inference worker thread and display stream...")
        
        # Initialize an auto-fit resizable window before starting the loop
        # window_name = "Gazeloop Vision Agent - Live Stream"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(window_name, 1280, 720)  # Optional initial size
        
        try:
            while True:
                # Retrieve the next frame from the queue (blocks until a frame is available)
                frame = await self.queue.get()
                
                # Check for the 'poison pill' signaling stream termination
                if frame is None:
                    self.queue.task_done()
                    break

                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                try:
                    # --- Stage 1: Fast Local YOLO Screening (Runs on every frame) ---
                    results = self.model(frame, classes=[0], verbose=False)
                    is_potential_fall = False

                    for r in results:
                        
                        for box in r.boxes:
                            coords = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, coords[:4])
                            conf = float(box.conf[0].cpu())
                            
                            # Filter out low-confidence detections
                            if conf < 0.4:
                                continue
                                
                            # Quick geometric check (e.g., bounding box aspect ratio)
                            is_potential_fall = src.methods.check_box_aspect_ratio(x1, y1, x2, y2)
                            box_color = (0, 0, 255) if is_potential_fall else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    current_time = time.time()
                    
                    # --- Stage 2: Deep Verification & HITL Trigger (Rate-limited by cooldown) ---
                    if is_potential_fall and (current_time - last_alert_time) >= cooldown_period:
                        print("--- Potential fall detected: Delegating to edge/cloud model for deep verification ---")
                        
                        loop = asyncio.get_running_loop()
                        
                        # Run the blocking edge validator call in a separate background thread
                        hitl_record = await loop.run_in_executor(
                            None, self.edge_validator.verify_fall_event, frame
                        )
                        
                        # Reset the cooldown timestamp
                        last_alert_time = current_time

                        # If the deep validation confirms a high-risk event, process the alert
                        if hitl_record.get("ai_suggested_alert"):
                            alert_id = hitl_record["alert_id"]
                            is_alert = hitl_record["ai_suggested_alert"]
                            file_text = f"HITL_PENDING_ID_{alert_id[:8]}"
                            
                            Path("data/snapshots").mkdir(parents=True, exist_ok=True)
                            output_path = f"data/snapshots/{file_text}.jpg"
                            hitl_record["image_path"] = output_path
                            
                            status_text = self._draw_status(frame, is_alert)
                            
                            # Offload I/O-bound disk writing to the background thread pool
                            # log into sql
                            await loop.run_in_executor(
                                None, 
                                self.audit_logger.log_event, 
                                output_path, 
                                hitl_record.get("metadata", {}), 
                                hitl_record, 
                                "PENDING"
                            )
                            await loop.run_in_executor(None, self._save_to_json, hitl_record)
                            await loop.run_in_executor(None, self.save_analyze_result, frame, status_text, output_path)

                    # --- Stage 3: Real-time Display ---
                    # Draw normal/safe status if no alert is currently active
                    if not is_potential_fall:
                        self._draw_status(frame, is_alert=False)

                    # Render the frame to an OpenCV GUI window
                    # cv2.imshow(window_name, frame)
                    
                    # cv2.waitKey(1) is required to refresh the GUI window and capture keyboard input.
                    # Press 'q' to break out of the loop early if needed.
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     print("[Consumer] User requested exit via keyboard ('q').")
                    #     break

                except Exception as e:
                    print(f"[Consumer Error]: {e}")
                finally:
                    # Always notify the queue that processing for this frame is complete
                    self.queue.task_done()
                    
        finally:
            # Clean up and close any open OpenCV windows when finished
            # cv2.destroyAllWindows()
            pass

    async def run_async_stream(self, source: Union[int, str]):
        producer_task = asyncio.create_task(self.frame_producer(source))
        consumer_task = asyncio.create_task(self.frame_consumer())
        await asyncio.gather(producer_task, consumer_task)