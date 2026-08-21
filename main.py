import re
import random
import argparse
from pathlib import Path
import shutil
import pexpect

DEFAULT_GAME_PATH = Path.home() / "Games" / "ZCode" / "advent.z5"
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class FrotzEnv:
    def __init__(self, game_path=None, frotz_bin="frotz"):
        self.frotz_bin = shutil.which(frotz_bin) or frotz_bin
        if not shutil.which(frotz_bin) and not Path(frotz_bin).exists():
            raise FileNotFoundError(f"Frotz executable '{frotz_bin}' was not found.")
        self.game_path = self._resolve_game_path(game_path)
        self.child = None

        self.action_space = [
            "look", "inventory", "north", "south", "east", "west",
            "up", "down", "in", "out", "take all", "drop all",
            "open door", "examine lamp", "take lamp", "light lamp"
        ]

    def _resolve_game_path(self, game_path=None):
        candidates = []
        if game_path is not None:
            candidates.append(Path(game_path))
        else:
            candidates.append(DEFAULT_GAME_PATH)

        for name in ["advent.z5", "Advent.z5", "advent.z8", "Advent.z8"]:
            candidates.append(Path.home() / "Games" / "ZCode" / name)
            candidates.append(Path.cwd() / name)

        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return str(resolved)

        raise FileNotFoundError(f"No Z-machine game file found at {DEFAULT_GAME_PATH} or alternative candidates.")

    def reset(self):
        if self.child and self.child.isalive():
            self.child.close()

        # Launch frotz with plain output (-p) and quiet mode (-q)
        # Use standard pty terminal dimensions to keep frotz stable
        self.child = pexpect.spawn(
            self.frotz_bin, 
            ["-p", "-q", self.game_path], 
            encoding='utf-8', 
            dimensions=(24, 80),
            timeout=2
        )
        self._wait_for_prompt()
        return self._clean_text(self.child.before)

    def _wait_for_prompt(self):
        # Match standard prompt > or fall back gracefully on timeout
        try:
            self.child.expect([r'>', pexpect.TIMEOUT], timeout=1.0)
        except pexpect.EOF:
            pass

    def step(self, action_str):
        if not self.child or not self.child.isalive():
            raise RuntimeError("Environment not initialized.")

        self.child.sendline(action_str)
        self._wait_for_prompt()
        output = self._clean_text(self.child.before)

        score = self._get_score()
        reward = float(score) - getattr(self, 'last_score', 0)
        self.last_score = score
        if reward == 0:
            reward = -0.01

        done = False
        if "you have died" in output.lower() or "you win" in output.lower():
            done = True

        return output, reward, done, {}

    def _get_score(self):
        try:
            self.child.sendline("score")
            self._wait_for_prompt()
            clean = ANSI_ESCAPE.sub('', self.child.before)
            match = re.search(r'score of (\d+)', clean, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 0

    def _clean_text(self, text):
        if not text:
            return ""
        clean = ANSI_ESCAPE.sub('', text)
        lines = clean.splitlines()
        cleaned = [l.strip() for l in lines if l.strip() and not l.strip().startswith('>')]
        return " ".join(cleaned)

    def close(self):
        if self.child and self.child.isalive():
            self.child.close()

class TabularQAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.99995, epsilon_min=0.01):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
        return self.q_table[state]

    def choose_action(self, state, greedy=False):
        q_vals = self.get_q_values(state)
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        
        max_v = max(q_vals)
        best_indices = [i for i, v in enumerate(q_vals) if v == max_v]
        return random.choice(best_indices)

    def update(self, state, action, reward, next_state, done):
        q_vals = self.get_q_values(state)
        next_q_vals = self.get_q_values(next_state)
        
        max_next_q = max(next_q_vals) if not done else 0.0
        target = reward + self.gamma * max_next_q
        q_vals[action] += self.alpha * (target - q_vals[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train(env, agent, episodes=70000, max_steps=30):
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action_idx = agent.choose_action(state)
            action_str = env.action_space[action_idx]
            
            next_state, reward, done, _ = env.step(action_str)
            agent.update(state, action_idx, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        if (ep + 1) % 1000 == 0 or ep == episodes - 1:
            print(f"Episode {ep + 1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}", flush=True)

def play_walkthrough(env, agent, max_steps=50):
    print("\n--- STARTING TRAINED WALKTHROUGH ---\n")
    state = env.reset()
    print(f"[Initial Observation]: {state}\n")
    total_reward = 0

    for step in range(1, max_steps + 1):
        action_idx = agent.choose_action(state, greedy=True)
        action_str = env.action_space[action_idx]

        next_state, reward, done, _ = env.step(action_str)
        total_reward += reward

        print(f"Step {step} | Action: '{action_str}' | Reward: {reward:.2f}")
        print(f"Game Response: {next_state}\n")

        state = next_state
        if done:
            break

    print(f"Walkthrough Finished | Cumulative Reward: {total_reward:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-path", type=str, default=None)
    parser.add_argument("--frotz-bin", type=str, default="frotz")
    parser.add_argument("--episodes", type=int, default=70000)
    args = parser.parse_args()

    env = FrotzEnv(game_path=args.game_path, frotz_bin=args.frotz_bin)
    agent = TabularQAgent(actions=env.action_space)

    train(env, agent, episodes=args.episodes, max_steps=30)
    play_walkthrough(env, agent, max_steps=50)
    env.close()
