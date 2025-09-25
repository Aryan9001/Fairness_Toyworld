from typing import List, Tuple
from src.kg.graph import KnowledgeGraph, Triple
from src.policy.action_space import Action
from src.config import MAX_FACTS_PER_EXPLANATION
from src.explain.attribution import score_triple


def _format_triple(t: Triple) -> str:
    h, r, v = t
    if r == "has":
        return f"I have {v}"
    if r == "is":
        return f"{h} is {v}"
    if r == "in":
        return f"{h} is in {v}"
    if r == "dir":
        dir_, dest = v.split(":", 1)
        return f"From {h}, there is a path {dir_} to {dest}"
    return f"{h} {r} {v}"


def _unique_triples(triples: List[Triple]) -> List[Triple]:
    seen = set()
    out = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _pick_supporting(kg: KnowledgeGraph, action: Action, current_location: str) -> List[Triple]:
    triples = kg.to_list()

    if action.name == "take":
        obj = action.params["object"]
        here = [t for t in triples if t[1] == "in" and t[0] == obj and t[2] == current_location]
        have = [t for t in triples if t[1] == "has" and t[2] == obj]
        return here + have

    if action.name == "go":
        d = action.params["direction"]
        return [t for t in triples if t[1] == "dir" and t[0] == current_location and t[2].startswith(f"{d}:")]

    if action.name == "open":
        obj = action.params["object"]
        # e.g., ("door","is","locked")
        return [t for t in triples if t[1] == "is" and t[0] == obj]

    if action.name == "use_on":
        # Prefer: “I have lamp” + “room is dim/dark” (if extractor captured it)
        support = [t for t in triples if t[1] == "has" and t[2] == action.params["object"]]
        support += [t for t in triples if t[1] == "is" and t[0] == "room" and t[2] in {"dim", "dark"}]
        return support

    return triples[:2]


def local_explanation(kg: KnowledgeGraph, action: Action, current_location: str) -> str:
    cands = _pick_supporting(kg, action, current_location)
    cands = _unique_triples(cands)
    # sort by descending score
    cands.sort(key=lambda t: score_triple(t, action, current_location), reverse=True)
    facts = cands[:MAX_FACTS_PER_EXPLANATION]

    if facts:
        reasons = "; ".join(_format_triple(t) for t in facts)
        return f"I chose to {action.to_text()} because {reasons}."
    # Friendly fallbacks
    if action.name == "go":
        return f"I chose to {action.to_text()} because no known path {action.params['direction']} from {current_location}; exploring."
    if action.name == "use_on":
        return f"I chose to {action.to_text()} to improve visibility here."
    return f"I chose to {action.to_text()} because current context offers no clear supporting facts."
