import os
import json

# Locate the root data directory
# Since this file is in project_root/src/utils.py, parent of parent is project_root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
METRICS_FILE = os.path.join(DATA_DIR, "pipeline_metrics.json")
HISTORY_FILE = os.path.join(DATA_DIR, "historical_runs.json")

def log_daily_run(metrics):
    """
    Log MLOps daily run metrics.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save the latest daily run metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Load and update metrics run history
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            pass
            
    # Remove older runs with the exact same date to prevent duplicates on manual reruns
    history = [h for h in history if h.get("date") != metrics.get("date")]
    history.append(metrics)
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"Logged daily run metrics successfully to {METRICS_FILE} and {HISTORY_FILE}")
