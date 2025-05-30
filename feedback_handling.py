# feedback_handling.py
import os
from datetime import datetime

FEEDBACK_DIR = "feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def save_feedback(name: str, email: str, message: str) -> str:
    """Saves feedback to a uniquely named text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{name.replace(' ', '_')}.txt"
    filepath = os.path.join(FEEDBACK_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Name: {name}\n")
        f.write(f"Email: {email}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Message:\n")
        f.write(message.strip())

    return filepath

def get_feedback_files():
    return [f for f in os.listdir(FEEDBACK_DIR) if f.endswith(".txt")]
