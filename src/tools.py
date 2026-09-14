import datetime
from pathlib import Path


def trigger_alert(reason: str, severity: str = "Medium") -> str:
  """
  Trigger an alert and log the details to a local log file.

  Args:
      reason: Description of the alert cause
      severity: Severity level (e.g., Low, Medium, High)
  """
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  log_dir = Path("logs")
  log_dir.mkdir(parents=True, exist_ok=True)
  log_file = log_dir / "alert.log"

  log_entry = (
      f"[{timestamp}] [Severity: {severity}] Alert triggered: {reason}\n"
  )
  with open(log_file, "a", encoding="utf-8") as f:
    f.write(log_entry)

  return f"Successfully logged: {reason} (Level: {severity})"