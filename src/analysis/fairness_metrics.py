import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from src.envs.toy_textworld import initial_state, fresh_world, step_fn
from src.kg.extractors import extract_triples_from_text
from src.kg.graph import KnowledgeGraph
from src.policy.policy_stub import HeuristicPolicy
from src.config import N_RUNS


def gini(counts: Counter) -> float:
    arr = np.array(list(counts.values()))
    if arr.sum() == 0:
        return 0
    arr = np.sort(arr)
    n = len(arr)
    cum = np.cumsum(arr)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def disparate_impact(counts: Counter) -> float:
    short = counts.get("take egg", 0)
    long = counts.get("take lamp", 0) + counts.get("take key", 0) + counts.get("open chest", 0)
    if short == 0:
        return 1.0
    return long / short


def run_episode(bias_mode: bool, max_steps: int = 20):
    state = initial_state()
    world = fresh_world()   # fresh world copy per episode
    policy = HeuristicPolicy(bias_mode=bias_mode)
    kg = KnowledgeGraph()
    actions, rewards = [], []
    chest_opened = False

    for step in range(max_steps):
        triples = extract_triples_from_text(state.obs, state.inventory, state.location)
        kg.upsert_triples(triples, step)

        action = policy.select_action(kg, state.location)
        actions.append(action.to_text())

        new_state = step_fn(state, action.to_text(), world)
        new_state.inventory = set(state.inventory).union(new_state.inventory)
        state = new_state

        # intrinsic reward shaping (only for unbiased)
        reward = state.reward
        if not bias_mode and action.name in {"take", "open", "use_on"}:
            if action.to_text() in {"take lamp", "take key", "open chest", "use lamp"}:
                reward += 0.5
        rewards.append(reward)

        if "chest is open" in state.obs.lower():
            chest_opened = True
            break

    return actions, rewards, chest_opened, kg


def plot_results(results):
    os.makedirs("plots", exist_ok=True)
    modes = ["Biased", "Unbiased"]

    # --- 1. Action Distribution per Agent ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, mode in enumerate(modes):
        counts = results[mode]["counts"]
        ax[i].bar(counts.keys(), counts.values())
        ax[i].set_title(f"{mode} Agent - Action Distribution")
        ax[i].set_ylabel("Count")
        ax[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("plots/action_distribution.png")
    plt.show()

    # --- 2. Fairness Metrics Comparison ---
    metrics = ["avg_reward", "gini", "disparate_impact", "chest_open_rate"]
    labels = ["Avg Reward", "Gini Index", "Disparate Impact", "Chest Open %"]

    biased_vals = [results["Biased"][m] for m in metrics]
    unbiased_vals = [results["Unbiased"][m] for m in metrics]

    x = range(len(metrics))
    plt.figure(figsize=(8, 5))
    plt.bar(x, biased_vals, width=0.4, label="Biased", align="center")
    plt.bar([p + 0.4 for p in x], unbiased_vals, width=0.4, label="Unbiased", align="center")
    plt.xticks([p + 0.2 for p in x], labels)
    plt.title("Fairness Metrics Comparison")
    plt.legend()
    plt.savefig("plots/fairness_metrics.png")
    plt.show()


def evaluate():
    results = {}
    for mode in [True, False]:  # True = Biased, False = Unbiased
        episodes = N_RUNS
        all_actions, total_rewards, chest_count = [], [], 0
        kgs = []

        for ep in range(episodes):
            actions, rewards, opened, kg = run_episode(mode)
            all_actions.extend(actions)
            total_rewards.extend(rewards)
            if opened:
                chest_count += 1
            kgs.append(kg)

        counts = Counter(all_actions)
        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        gini_val = gini(counts)
        di_val = disparate_impact(counts)

        print(f"\n=== Running {'Biased' if mode else 'Unbiased'} Mode ===")
        print(f"Action counts: {counts}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Gini Index (action inequality): {gini_val:.2f}")
        print(f"Disparate Impact (long/short): {di_val:.2f}")
        print(f"Chest Opened in {chest_count}/{episodes} runs ({chest_count/episodes:.1%})")

        if kgs:
            triples = kgs[0].to_list()
            print("\nSample Knowledge Graph (from episode 1):")
            for h, r, t in triples:
                print(f"  ({h}) -[{r}]-> ({t})")

        results["Biased" if mode else "Unbiased"] = {
            "counts": counts,
            "avg_reward": avg_reward,
            "gini": gini_val,
            "disparate_impact": di_val,
            "chest_open_rate": chest_count / episodes,
        }

    # plot after both modes are done
    plot_results(results)


if __name__ == "__main__":
    evaluate()
