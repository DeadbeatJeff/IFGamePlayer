import argparse
import re
import random
import collections
from pathlib import Path
import shutil
import pexpect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

DEFAULT_GAME_PATH = Path("/home/jeff/ExpanDrive/Google Drive/Games/ZCode/advent.z5")

# --------------------------------------------------
# 1. Custom Text Adventure Environment Wrapper
# --------------------------------------------------
class FrotzEnv:
    def __init__(self, game_path=None, frotz_bin="dfrotz"):
        self.frotz_bin = shutil.which(frotz_bin) or frotz_bin
        if not shutil.which(frotz_bin) and not Path(frotz_bin).exists():
            raise FileNotFoundError(
                f"Frotz executable '{frotz_bin}' was not found. Install frotz or pass --frotz-bin with the correct path."
            )
        self.game_path = self._resolve_game_path(game_path)
        self.child = None

        # Action space: Basic navigation & interaction commands
        self.action_space = [
            "look", "inventory", "north", "south", "east", "west",
            "up", "down", "in", "out", "take all", "drop all",
            "open door", "examine lamp", "take lamp", "light lamp", "score"
        ]

    def _resolve_game_path(self, game_path=None):
        candidates = []

        if game_path is not None:
            candidates.append(Path(game_path))
        else:
            candidates.append(DEFAULT_GAME_PATH)

        default_names = ["Advent.z5", "advent.z5", "advent.z8", "Advent.z8"]
        for name in default_names:
            candidates.append(Path.cwd() / name)
            candidates.append(Path.cwd() / "IFGamePlayer" / name)

        root = Path.cwd()
        for pattern in ("*.z1", "*.z2", "*.z3", "*.z4", "*.z5", "*.z6", "*.z7", "*.z8"):
            candidates.extend(sorted(root.rglob(pattern)))

        seen = set()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                return str(resolved)

        raise FileNotFoundError(
            "No Z-machine game file was found. Place a game such as Advent.z5 in the project root or pass game_path explicitly."
        )

    def reset(self):
        if self.child and self.child.isalive():
            self.child.close()

        # Spawn dfrotz non-interactively using argument list to avoid shell quoting issues.
        self.child = pexpect.spawn(self.frotz_bin, [self.game_path], encoding='utf-8', timeout=2)
        self.child.expect('>')  # Wait for command prompt
        raw_output = self.child.before

        return self._clean_text(raw_output)

    def step(self, action_str):
        if not self.child or not self.child.isalive():
            raise RuntimeError("Environment terminal or not initialized.")

        # Send action to frotz
        self.child.sendline(action_str)
        self.child.expect('>')
        output = self._clean_text(self.child.before)

        # Retrieve score to build reward signal
        score = self._get_score()
        
        # Basic reward structure: score change or small step penalty
        reward = float(score) - getattr(self, 'last_score', 0)
        self.last_score = score
        if reward == 0:
            reward = -0.01  # Small penalty to discourage infinite loops

        done = False
        if "you have died" in output.lower() or "you win" in output.lower():
            done = True

        return output, reward, done, {}

    def _get_score(self):
        """Query score silently from game state."""
        try:
            self.child.sendline("score")
            self.child.expect('>')
            score_text = self.child.before
            match = re.search(r'score of (\d+)', score_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 0

    def _clean_text(self, text):
        """Remove control characters and standard prompt noise."""
        lines = text.splitlines()
        cleaned = [l.strip() for l in lines if l.strip() and not l.strip().startswith('>')]
        return " ".join(cleaned)

    def close(self):
        if self.child and self.child.isalive():
            self.child.close()

# --------------------------------------------------
# 2. DQN Neural Network & Replay Buffer
# --------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.fc(x)

ReplayBuffer = collections.deque(maxlen=2000)

# --------------------------------------------------
# 3. Training Loop
# --------------------------------------------------
def train_dqn(game_path=None, frotz_bin="dfrotz"):
    env = FrotzEnv(game_path=game_path, frotz_bin=frotz_bin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-trained NLP Encoder for state text representations
    encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    state_dim = 384  # Embedding dimension of all-MiniLM-L6-v2
    num_actions = len(env.action_space)

    # Hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    lr = 1e-3
    batch_size = 32
    episodes = 50
    max_steps_per_ep = 30

    q_net = QNetwork(state_dim, num_actions).to(device)
    target_net = QNetwork(state_dim, num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    for ep in range(episodes):
        state_text = env.reset()
        state_emb = encoder.encode(state_text, convert_to_tensor=True, device=device).detach()
        total_reward = 0

        for step in range(max_steps_per_ep):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randint(0, num_actions - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(state_emb.unsqueeze(0))
                    action_idx = torch.argmax(q_values).item()

            action_str = env.action_space[action_idx]
            next_state_text, reward, done, _ = env.step(action_str)
            next_state_emb = encoder.encode(next_state_text, convert_to_tensor=True, device=device).detach()

            # Save transition
            ReplayBuffer.append((state_emb, action_idx, reward, next_state_emb, done))
            state_emb = next_state_emb
            total_reward += reward

            # Train Network
            if len(ReplayBuffer) >= batch_size:
                batch = random.sample(ReplayBuffer, batch_size)
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)

                s_b = torch.stack(s_b).to(device)
                a_b = torch.tensor(a_b, dtype=torch.long, device=device).unsqueeze(1)
                r_b = torch.tensor(r_b, dtype=torch.float32, device=device).unsqueeze(1)
                ns_b = torch.stack(ns_b).to(device)
                d_b = torch.tensor(d_b, dtype=torch.float32, device=device).unsqueeze(1)

                current_q = q_net(s_b).gather(1, a_b)
                max_next_q = target_net(ns_b).max(1)[0].unsqueeze(1)
                target_q = r_b + (1 - d_b) * gamma * max_next_q

                loss = nn.MSELoss()(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Decay epsilon and sync target network
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if ep % 5 == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {ep + 1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")

    env.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on an IF game via Frotz.")
    parser.add_argument("--game-path", type=str, default=None, help="Path to the Z-machine game file, e.g. Advent.z5")
    parser.add_argument("--frotz-bin", type=str, default="dfrotz", help="Path or name of the dfrotz executable")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train_dqn(game_path=args.game_path, frotz_bin=args.frotz_bin)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing game asset: {exc}")