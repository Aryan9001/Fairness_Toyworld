import re
from typing import List, Tuple

Triple = Tuple[str, str, str]

def extract_triples_from_text(obs: str, inventory, location: str) -> List[Triple]:
    triples: List[Triple] = []

    # Ensure inventory is string for regex
    inv_str = ", ".join(inventory) if isinstance(inventory, (set, list)) else str(inventory)

    # Inventory triples
    for item in inv_str.split(","):
        item = item.strip()
        if item:
            triples.append((item, "carried_by", "agent"))

    # Location triple
    triples.append(("agent", "at", location))

    # Extract objects from obs (simplified heuristic)
    match_objs = re.findall(r"\b(\w+)\b", obs)
    for word in match_objs:
        if word.lower() in ["egg", "lamp", "key", "chest"]:
            triples.append((word, "in", location))

    # Extract directions
    if "north" in obs.lower():
        triples.append((location, "dir", "north"))
    if "south" in obs.lower():
        triples.append((location, "dir", "south"))
    if "east" in obs.lower():
        triples.append((location, "dir", "east"))
    if "west" in obs.lower():
        triples.append((location, "dir", "west"))

    return triples
