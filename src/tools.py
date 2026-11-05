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

logger = logging.getLogger("gazeloop")

# Mute noisy third-party SDK logs
for name in ["google", "google.genai", "httpx", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a GazeLoop component."""
    return logging.getLogger(f"gazeloop.{name}")

def trigger_alert(reason: str, severity: str = "Medium") -> str:
    """
    Trigger an alert and record it in the application log.
    """

    timestamp = datetime.datetime.now().isoformat(
        timespec="seconds"
    )

    log_message = (
        f"Alert triggered | Severity: {severity} | "
        f"Reason: {reason}"
    )

    logger.warning(log_message)

    return (
        f"Successfully logged: {reason} "
        f"(Level: {severity}) at {timestamp}"
    )
