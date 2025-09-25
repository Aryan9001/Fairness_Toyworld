# simple scoring heuristic for triples -> higher = more relevant
from src.kg.graph import Triple
from src.policy.action_space import Action

def score_triple(t: Triple, action: Action, current_location: str) -> float:
    h, r, v = t
    score = 0.0

    # 1) Direct action arguments matter most
    if action.name in {"take", "open", "use_on"}:
        arg = action.params.get("object")
        if h == arg or v == arg:
            score += 3.0
    if action.name == "go":
        d = action.params.get("direction", "")
        if r == "dir" and h == current_location and v.startswith(f"{d}:"):
            score += 3.0

    # 2) Inventory facts help for use/take
    if r == "has" and action.name in {"use_on", "take"}:
        score += 1.5

    # 3) Location grounding
    if r == "in" and v == current_location:
        score += 1.0

    # 4) Salient room attributes (dim/dark) help explain lamp usage
    if r == "is" and h in {"room", "door"}:
        if "dim" in v or "dark" in v: score += 1.2
        if "locked" in v and action.name == "open": score += 1.2

    return score
