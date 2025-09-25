from typing import List, Dict, Any

def summarize_trajectory(steps: List[Dict[str, Any]]) -> str:
    """
    Keep key steps: reward>0 OR location changed.
    """
    key = []
    for s in steps:
        moved = (s.get("next_location") is not None) and (s.get("next_location") != s.get("location"))
        if s.get("reward", 0) > 0 or moved:
            key.append(s)

    if not key and steps:
        key = [steps[0], steps[-1]]

    lines = []
    for s in key[:8]:
        moved_txt = ""
        if s.get("next_location") and s.get("next_location") != s.get("location"):
            moved_txt = f" (moved {s['location']} → {s['next_location']})"
        lines.append(f"- {s['action_text']}{moved_txt}: {s['local_reason']}")

    return "Trajectory summary:\n" + "\n".join(lines)
