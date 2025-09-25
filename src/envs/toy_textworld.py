from dataclasses import dataclass
import copy

@dataclass
class State:
    location: str
    inventory: set
    obs: str
    reward: int = 0


# Base template world (reset each episode)
BASE_WORLD = {
    "Garden": {"north": "Kitchen", "objects": ["egg"]},
    "Kitchen": {"north": "Hallway", "south": "Garden", "objects": ["lamp"]},
    "Hallway": {"east": "Attic", "south": "Kitchen", "objects": ["key"]},
    "Attic": {"west": "Hallway", "objects": ["chest"], "dark": True},
}


def initial_state() -> State:
    return State(
        location="Garden",
        inventory=set(),
        obs="You are in the Garden. You see an egg here. Paths: north -> Kitchen.",
        reward=0,
    )


def fresh_world() -> dict:
    """Return a new copy of the world for each episode so objects reset cleanly."""
    return copy.deepcopy(BASE_WORLD)


def step_fn(state: State, action: str, world: dict) -> State:
    """Apply an action to the world and return the next state."""
    loc = world[state.location]

    # --- Take object ---
    if action.startswith("take "):
        obj = action.split(" ", 1)[1]
        if obj in loc.get("objects", []):
            new_inv = set(state.inventory)
            new_inv.add(obj)
            loc["objects"].remove(obj)
            return State(
                location=state.location,
                inventory=new_inv,
                obs=f"You took the {obj}.",
                reward=1,
            )
        return State(state.location, set(state.inventory), f"No {obj} here.", reward=0)

    # --- Move ---
    if action.startswith("go "):
        direction = action.split(" ", 1)[1]
        if direction in loc:
            new_loc = loc[direction]
            return State(
                location=new_loc,
                inventory=set(state.inventory),
                obs=f"You are in {new_loc}. You see {world[new_loc].get('objects', [])}. "
                    f"Paths: {list(world[new_loc].keys())}",
                reward=0,
            )
        return State(state.location, set(state.inventory), "Can't go that way.", reward=0)

    # --- Use lamp ---
    if action == "use lamp" and "lamp" in state.inventory and state.location == "Attic":
        return State(
            location="Attic",
            inventory=set(state.inventory),
            obs="The lamp lights up the Attic. You see a chest here.",
            reward=1,
        )

    # --- Open chest ---
    if action == "open chest" and "key" in state.inventory and state.location == "Attic":
        return State(
            location="Attic",
            inventory=set(state.inventory),
            obs="The chest is open! Treasure inside.",
            reward=10,
        )

    return State(state.location, set(state.inventory), "Nothing happens.", reward=0)
