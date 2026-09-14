from pathlib import Path
from typing import List, Union
from google import genai
from google.genai import types
from src.tools import trigger_alert
import cv2

# Define alert keywords separately
ALERT_KEYWORDS = [
    "alert",
    "abnormal",
    "fire",
    "fall",
]
        
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

    def _inspect_frame_data(self, encoded_img, mime_type) -> bool:
        img_part = types.Part.from_bytes(
            data=encoded_img.tobytes(), mime_type=mime_type
        )
        
        print(f"Sending to Gemini for analysis...")
        response = self.client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[
                img_part,
                f"Please inspect this frame for any abnormal situations that require an alert.",
            ],
            config=self.config,
        )
        
        is_alert = False
              
        if response.function_calls:
            for call in response.function_calls:
                if call.name == "trigger_alert":
                    args = call.args
                    res = trigger_alert(
                        reason=args.get("reason", "Unknown anomaly"),
                        severity=args.get("severity", "Medium")
                    )
                    print(f"[Agent Execution Feedback]: {res}")
                    is_alert = True
        else:
            print(f"[Agent Reasoning Result]: {response.text}")
            
            if any(word in response.text.lower() for word in ALERT_KEYWORDS):
                is_alert = True
                
        return is_alert       
        
    def _draw_status(self, frame, is_alert: bool):
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
    
    def analyze_images(self):
        for img_path in self.image_paths:
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
  
            is_alert = self._inspect_frame_data(encoded_img, mime_type)
            self._draw_status(frame, is_alert)
        
            cv2.imshow(f"GazeLoop - Inspecting: {img_path}", frame)
            cv2.waitKey(0)
                
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    def analyze_stream(self, source: Union[int, str]):
        print(f"Stream started for '{source}'. Press 'q' in the window to exit.")
        
        # stream handler for both camera / video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Cannot open source: {source}")
            return
        
        frame_idx = 0
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break
            
            success, encoded_img = cv2.imencode(".jpg", frame)
            if not success:
                continue
        
            is_alert = self._inspect_frame_data(encoded_img, mime_type="image/jpeg")
            self._draw_status(frame, is_alert)
            
            cv2.imshow(f"GazeLoop - Inspecting: stream, idx: {frame_idx}", frame)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            
            frame_idx += 1
            
        cap.release()
        cv2.destroyAllWindows()