from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Action:
    name: str
    params: Dict[str, Any]

    def to_text(self) -> str:
        if self.name == "go":    return f"go {self.params['direction']}"
        if self.name == "take":  return f"take {self.params['object']}"
        if self.name == "open":  return f"open {self.params['object']}"
        if self.name == "use_on":return f"use {self.params['object']} on {self.params['target']}"
        return self.name
