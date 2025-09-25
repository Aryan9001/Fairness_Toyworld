from src.policy.action_space import Action

class HeuristicPolicy:
    def __init__(self, bias_mode: bool = True):
        """
        bias_mode=True  -> biased (short-term greedy, egg only)
        bias_mode=False -> unbiased (explores long-term path: lamp + key + chest)
        """
        self.bias_mode = bias_mode

    def select_action(self, kg, location):
        inv = kg.inventory if hasattr(kg, "inventory") else set()
        room_objs = list(kg.objects_at(location))
        dirs = kg.directions(location)

        # --- Biased Agent: only wants egg ---
        if self.bias_mode:
            if "egg" in room_objs:
                return Action("take", {"object": "egg"})
            if "south" in dirs:
                return Action("go", {"direction": "south"})
            if "north" in dirs:
                return Action("go", {"direction": "north"})
            return Action("go", {"direction": dirs[0] if dirs else "north"})

        # --- Unbiased Agent: explore & solve puzzle ---
        else:
            # Collect items if available
            if "egg" in room_objs and "egg" not in inv:
                return Action("take", {"object": "egg"})
            if "lamp" in room_objs and "lamp" not in inv:
                return Action("take", {"object": "lamp"})
            if "key" in room_objs and "key" not in inv:
                return Action("take", {"object": "key"})

            # Move into the Attic once key is collected
            if location == "Hallway" and "east" in dirs:
                return Action("go", {"direction": "east"})

            # Solve Attic puzzle
            if location == "Attic" and "lamp" in inv and "key" in inv:
                return Action("open", {"object": "chest"})
            if location == "Attic" and "lamp" in inv:
                return Action("use_on", {"object": "lamp", "target": "room"})

            # Default exploration fallback
            if dirs:
                return Action("go", {"direction": dirs[0]})
            return Action("go", {"direction": "north"})
