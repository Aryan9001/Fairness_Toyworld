import argparse
from src.envs.toy_textworld import initial_state, fresh_world, step_fn
from src.kg.graph import KnowledgeGraph
from src.kg.extractors import extract_triples_from_text
from src.policy.policy_stub import HeuristicPolicy
from src.explain.local import local_explanation
from src.explain.temporal import summarize_trajectory
from src.logger.recorder import Recorder


def main(bias_mode: bool = True, return_actions: bool = False):
    """Run one toy episode with explainability and logging."""
    state = initial_state()
    world = fresh_world()   # <<< NEW: fresh copy of world
    kg = KnowledgeGraph()
    policy = HeuristicPolicy(bias_mode=bias_mode)
    rec = Recorder()

    for step in range(12):
        # 1) Update KG
        triples = extract_triples_from_text(state.obs, state.inventory, state.location)
        kg.upsert_triples(triples, step=step)

        # 2) Choose action
        action = policy.select_action(kg, state.location)

        # 3) Explain
        reason = local_explanation(kg, action, state.location)

        # 4) Apply action (world now passed in)
        new_state = step_fn(state, action.to_text(), world)

        # carry forward inventory
        new_state.inventory = set(state.inventory).union(new_state.inventory)

        # 5) Log
        rec.log(
            step=step,
            location=state.location,
            next_location=new_state.location,
            obs=state.obs,
            kg_triples=kg.to_list(),
            action_text=action.to_text(),
            action_name=action.name,
            local_reason=reason,
            reward=new_state.reward,
        )

        # 6) Print
        if not return_actions:
            print(f"[step {step}] {action.to_text()} → {reason}")

        # 7) Stop if chest opened
        if "chest is open" in new_state.obs.lower():
            state = new_state
            break

        state = new_state

    # 8) Summarize
    if not return_actions:
        print("\n" + summarize_trajectory(rec.steps))

    log_path = rec.flush()
    if not return_actions:
        print(f"\nSaved audit log: {log_path}")
    else:
        return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Toy TextWorld Agent")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["biased", "unbiased"],
        default="biased",
        help="Choose agent mode: biased (short-term reward) or unbiased (long-term exploration)."
    )
    args = parser.parse_args()

    bias_mode = True if args.mode == "biased" else False
    main(bias_mode=bias_mode)
