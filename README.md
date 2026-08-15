# Deep-Q Learning Agent for Colossal Cave Adventure

This repository contains a PyTorch implementation of a Deep Q-Network (DQN) agent designed to learn and play the classic text-based interactive fiction game **Colossal Cave Adventure** (`Advent.z5`) using `dfrotz` in a Linux environment.

After training the neural network across specified episodes, the script automatically executes a deterministic walkthrough of the game using the learned policy.

---

## 📌 Overview

* **Environment Wrapper (`FrotzEnv`):** Wraps the `dfrotz` interpreter using `pexpect` to send text commands and parse game outputs non-interactively.
* **State Representation:** Converts unstructured game text responses into high-dimensional semantic embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
* **Deep Q-Network:** A multi-layer perceptron (MLP) mapping text embeddings to expected Q-values across a discrete set of classic interactive fiction actions.
* **Automatic Walkthrough:** Runs a post-training evaluation step ($\epsilon = 0$) that prints a step-by-step trace of the agent's actions and game responses.

---

## 🛠️ Prerequisites & Setup

### 1. System Dependencies (Ubuntu 24.04 LTS)

Ensure system packages including `frotz` (which provides `dfrotz`) are installed:

```bash
sudo apt update
sudo apt install -y frotz python3-pip python3-venv
```

### 2. Python Environment

Set up a virtual environment and install required libraries:

```bash
python3 -m venv dqn_env
source dqn_env/bin/activate
pip install torch sentence-transformers pexpect numpy
```

---

## 🚀 Usage

### 1. Game File Placement

Place your Z-machine game file (e.g., `Advent.z5`) in the repository root directory or specify its path at launch.

### 2. Run Training & Walkthrough

Execute the main script:

```bash
python3 main.py --game-path Advent.z5 --frotz-bin dfrotz
```

### Command Line Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--game-path` | Path to the Z-machine game file (`.z5`, `.z8`, etc.) | Auto-detects `Advent.z5` |
| `--frotz-bin` | Name or path of the Frotz executable | `dfrotz` |

---

## 🔁 Workflow

1. **Training Phase:** The agent explores the text environment via epsilon-greedy action selection, populating a replay memory buffer and updating the Q-network via Temporal Difference loss.
2. **Evaluation Phase (Walkthrough):** Upon completing the training loop, the agent switches to evaluation mode (`q_net.eval()`). It operates deterministically ($\epsilon = 0$) to generate and log a full step-by-step game walkthrough.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
