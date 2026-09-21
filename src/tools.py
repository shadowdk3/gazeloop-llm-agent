import datetime
from pathlib import Path
import logging

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "alert.log"
  
# Configure Python's built-in logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- MUTE NOISY THIRD-PARTY SDK LOGS ---
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def trigger_alert(reason: str, severity: str = "Medium") -> str:
    """
    Trigger an alert and log the details to a local log file.

    Args:
        reason: Description of the alert cause
        severity: Severity level (e.g., Low, Medium, High)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = (
        f"[{timestamp}] [Severity: {severity}] Alert triggered: {reason}\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    return f"Successfully logged: {reason} (Level: {severity})"