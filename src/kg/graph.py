from dataclasses import dataclass
from typing import List, Tuple, Iterable
import networkx as nx

Triple = Tuple[str, str, str]

@dataclass
class KGSnapshot:
    triples: List[Triple]
    step: int


class KnowledgeGraph:
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.history: List[KGSnapshot] = []

    def upsert_triples(self, triples: Iterable[Triple], step: int):
        """Insert or update triples into the graph."""
        for h, r, t in triples:
            self.G.add_node(h)
            self.G.add_node(t)
            self.G.add_edge(h, t, relation=r)
        self.history.append(KGSnapshot(list(triples), step))

    def to_list(self) -> List[Triple]:
        """Return all triples as a list of (head, relation, tail)."""
        return [(u, d["relation"], v) for u, v, d in self.G.edges(data=True)]

    # --- Convenience methods for policies ---
    def objects_at(self, location: str) -> List[str]:
        """Return all objects linked to a location via 'in' relation."""
        return [
            u for u, v, d in self.G.edges(data=True)
            if d["relation"] == "in" and v == location
        ]

    def directions(self, location: str) -> List[str]:
        """Return available directions from a location."""
        return [
            v for u, v, d in self.G.edges(data=True)
            if u == location and d["relation"] == "dir"
        ]

    @property
    def inventory(self) -> List[str]:
        """Return all items the agent currently has (via 'carried_by' relation)."""
        return [
            u for u, v, d in self.G.edges(data=True)
            if d["relation"] in {"has", "carried_by"} and v == "agent"
        ]
