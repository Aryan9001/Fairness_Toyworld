from typing import List, Dict, Any
import json, os, time

class Recorder:
    def __init__(self, out_dir: str = "logs"):
        self.steps: List[Dict[str, Any]] = []
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def log(self, **kwargs):
        # auto-add action_name if only action_text is given
        if "action_text" in kwargs and "action_name" not in kwargs:
            kwargs["action_name"] = kwargs["action_text"].split()[0]
        self.steps.append(kwargs)

    def flush(self, run_name: str = None):
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = run_name or f"run-{ts}.jsonl"
        path = os.path.join(self.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for s in self.steps:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return path
