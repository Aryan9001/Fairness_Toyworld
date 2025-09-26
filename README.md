# Fairness & Explainability in Agentic AI (ToyWorld RL)

## Overview
This repository contains the implementation of a Fairness-aware, Explainable Reinforcement Learning environment designed as part of the SIT723/SIT792 Research Thesis at Deakin University.  
The project investigates how ethical components—fairness, explainability, and runtime oversight—can be embedded into Agentic AI systems for trustworthy governance.

We extend the ideas from Peng et al. (NeurIPS 2022) on inherently explainable RL and operationalize fairness techniques inspired by Jabbari et al. (2017) and Deng et al. (2023).  
Our sandbox environment demonstrates how biased (short-term) and unbiased (fairness-aware) agents behave differently under governance constraints.

---

## Features
- Toy TextWorld Environment with deterministic rooms (Garden, Kitchen, Hallway, Attic).
- Biased vs. Unbiased Agents  
  - Biased: optimizes for short-term reward (egg).  
  - Unbiased: fairness-aware with intrinsic reward shaping (lamp → key → chest).  
- Explainability Integration  
  - Natural Language rationales for actions.  
  - Knowledge Graph (KG) logs for symbolic accountability.  
- Fairness Metrics  
  - Action distribution  
  - Gini index  
  - Disparate impact  
  - Average reward  
  - Success (chest-open rate).  
- Governance Tools: runtime oversight, accountability artifacts, and participatory transparency.

---

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Aryan9001/Fairness_Toyworld.git
cd Fairness_Toyworld
```
###2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```
###3. Install Dependencies
```bash
pip install -r requirements.txt
```
###4. Run Agents
```bash
python -m src.run_toy --mode biased --episodes 10
python -m src.run_toy --mode unbiased --episodes 10
```
Example Output

Biased Agent
Action distribution skewed (90% eggs).
Gini Index ≈ 0.75.
Chest-Open Rate: 0%.

Unbiased Agent
Balanced distribution (egg, lamp, key, chest).
Gini Index ≈ 0.12.
Chest-Open Rate: 100%.

Logs include JSON files with:
Step-level actions + natural language explanations.
Knowledge Graph transitions.
Fairness metrics per run.

Author

Aryan Nandal (s224373399)
Bachelor of Software Engineering (Honours), Deakin University
Supervisor: Dr. Bahareh Nakisa
