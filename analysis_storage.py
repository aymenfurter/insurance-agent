import json
import os
from datetime import datetime
from typing import Dict, List, Optional

ANALYSIS_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "stored_analyses")
os.makedirs(ANALYSIS_STORAGE_DIR, exist_ok=True)

def save_analysis(analysis_data: Dict, name: Optional[str] = None) -> str:
    """Save analysis results with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        filename = f"{timestamp}_{name}.json"
    else:
        filename = f"{timestamp}_analysis.json"

    filepath = os.path.join(ANALYSIS_STORAGE_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(analysis_data, f)
    return filename

def list_saved_analyses() -> List[str]:
    """List all saved analyses."""
    if not os.path.exists(ANALYSIS_STORAGE_DIR):
        return []
    return sorted([f for f in os.listdir(ANALYSIS_STORAGE_DIR) if f.endswith('.json')], reverse=True)

def load_analysis(filename: str) -> Optional[Dict]:
    """Load analysis from file."""
    try:
        filepath = os.path.join(ANALYSIS_STORAGE_DIR, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def delete_analysis(filename: str) -> bool:
    """Delete analysis file."""
    try:
        filepath = os.path.join(ANALYSIS_STORAGE_DIR, filename)
        os.remove(filepath)
        return True
    except Exception:
        return False
