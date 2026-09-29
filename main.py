import re
import random
import argparse
import json
from pathlib import Path
import shutil
import pexpect

DEFAULT_GAME_PATH = Path.home() / "Games" / "ZCode" / "advent.z5"
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
CHECKPOINT_FILE = "q_checkpoint.json"
WALKTHROUGH_FILE = "walkthrough.txt"

BASE_ACTIONS = [
    "look", "inventory", "north", "south", "east", "west",
    "up", "down", "in", "out", "take all", "drop all",
    "open door", "examine lamp", "take lamp", "light lamp"
]
ACTION_VERBS = (
    "take", "get", "drop", "open", "close", "examine", "read",
    "light", "extinguish", "unlock", "lock", "eat", "drink", "fill",
    "pour", "wave", "rub", "feed", "attack", "kill", "throw", "wear",
    "remove", "move", "turn on", "turn off"
)
GAME_NOUNS = (
    "lamp", "lantern", "keys", "key", "food", "bottle", "water", "oil",
    "grate", "door", "cage", "bird", "rod", "pillow", "snake", "fissure",
    "bridge", "clam", "oyster", "magazine", "dwarf", "dwarves", "pirate",
    "dragon", "bear", "eggs", "vase", "coins", "diamonds", "silver", "gold",
    "jewelry", "spices", "chain", "axe", "troll", "plant", "beanstalk",
    "batteries", "carpet", "sign", "message", "mirror", "helmet", "statue"
)

class FrotzEnv:
    def __init__(self, game_path=None, frotz_bin="frotz"):
        self.frotz_bin = shutil.which(frotz_bin) or frotz_bin
        if not shutil.which(frotz_bin) and not Path(frotz_bin).exists():
            raise FileNotFoundError(f"Frotz executable '{frotz_bin}' was not found.")
        self.game_path = self._resolve_game_path(game_path)
        self.child = None
        self._recent_nouns = []

        self.action_space = list(BASE_ACTIONS)

    def action_space_for(self, observation):
        """Return base commands plus verb-noun pairs named in the observation."""
        observation = observation or ""
        visible_nouns = [
            noun for noun in GAME_NOUNS
            if re.search(rf"\b{re.escape(noun)}\b", observation, re.IGNORECASE)
        ]
        for noun in visible_nouns:
            if noun in self._recent_nouns:
                self._recent_nouns.remove(noun)
            self._recent_nouns.append(noun)
        self._recent_nouns = self._recent_nouns[-12:]
        actions = list(self.action_space)
        actions.extend(
            f"{verb} {noun}"
            for noun in self._recent_nouns
            for verb in ACTION_VERBS
        )
        return list(dict.fromkeys(actions))

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

        self._recent_nouns = []
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
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.5, epsilon_decay=0.99995, epsilon_min=0.001):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {}
        return self.q_table[state]

    def choose_action(self, state, greedy=False, actions=None):
        available_actions = actions or self.actions
        q_vals = self.get_q_values(state)
        for action in available_actions:
            q_vals.setdefault(action, 0.0)
        if not greedy and random.random() < self.epsilon:
            return random.randrange(len(available_actions))
        
        available_values = [q_vals[action] for action in available_actions]
        max_v = max(available_values)
        best_indices = [i for i, value in enumerate(available_values) if value == max_v]
        return random.choice(best_indices)

    def update(self, state, action, reward, next_state, done, actions=None, next_actions=None):
        available_actions = actions or self.actions
        action_name = available_actions[action] if isinstance(action, int) else action
        q_vals = self.get_q_values(state)
        q_vals.setdefault(action_name, 0.0)
        next_available_actions = next_actions or self.actions
        next_q_vals = self.get_q_values(next_state)
        for next_action in next_available_actions:
            next_q_vals.setdefault(next_action, 0.0)
        
        max_next_q = max(next_q_vals[action] for action in next_available_actions) if not done else 0.0
        target = reward + self.gamma * max_next_q
        q_vals[action_name] += self.alpha * (target - q_vals[action_name])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, episode, path=CHECKPOINT_FILE):
        data = {
            'episode': episode,
            'epsilon': self.epsilon,
            'q_table': self.q_table
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"[Info]: Saved checkpoint at episode {episode} to {path}", flush=True)

    def load_checkpoint(self, path=CHECKPOINT_FILE):
        if Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
            saved_q_table = data.get('q_table', {})
            self.q_table = {}
            for state, values in saved_q_table.items():
                if isinstance(values, list):
                    self.q_table[state] = {
                        action: value
                        for action, value in zip(self.actions, values)
                    }
                else:
                    self.q_table[state] = values
            self.epsilon = min(data.get('epsilon', self.epsilon), self.epsilon)
            start_ep = data.get('episode', 0)
            print(f"[Info]: Resuming from episode {start_ep} (Epsilon: {self.epsilon:.4f})", flush=True)
            return start_ep
        return 0

def train(env, agent, episodes=210000, max_steps=30, checkpoint_interval=5000):
    start_ep = agent.load_checkpoint(CHECKPOINT_FILE)

    for ep in range(start_ep, episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            actions = env.action_space_for(state)
            action_idx = agent.choose_action(state, actions=actions)
            action_str = actions[action_idx]
            
            next_state, reward, done, _ = env.step(action_str)
            next_actions = env.action_space_for(next_state)
            agent.update(state, action_str, reward, next_state, done, actions, next_actions)
            
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()

        if (ep + 1) % 1000 == 0 or ep == episodes - 1:
            print(f"Episode {ep + 1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}", flush=True)

        if (ep + 1) % checkpoint_interval == 0:
            agent.save_checkpoint(ep + 1, CHECKPOINT_FILE)

    agent.save_checkpoint(episodes, CHECKPOINT_FILE)

def play_and_save_walkthrough(env, agent, max_steps=50, output_file=WALKTHROUGH_FILE):
    print("\n--- STARTING TRAINED WALKTHROUGH ---\n")
    state = env.reset()
    commands = []
    total_reward = 0

    for step in range(1, max_steps + 1):
        actions = env.action_space_for(state)
        action_idx = agent.choose_action(state, greedy=True, actions=actions)
        action_str = actions[action_idx]
        commands.append(action_str)

        next_state, reward, done, _ = env.step(action_str)
        total_reward += reward

        print(f"Step {step} | Action: '{action_str}' | Reward: {reward:.2f}")
        print(f"Game Response: {next_state}\n")

        state = next_state
        if done:
            break

    with open(output_file, "w") as f:
        f.write("\n".join(commands) + "\n")

    print(f"[Info]: Saved {len(commands)} walkthrough commands to {output_file}")
    print(f"Walkthrough Finished | Cumulative Reward: {total_reward:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-path", type=str, default=None)
    parser.add_argument("--frotz-bin", type=str, default="frotz")
    parser.add_argument("--episodes", type=int, default=500000)
    args = parser.parse_args()

    env = FrotzEnv(game_path=args.game_path, frotz_bin=args.frotz_bin)
    agent = TabularQAgent(actions=env.action_space)

    train(env, agent, episodes=args.episodes, max_steps=30, checkpoint_interval=5000)
    play_and_save_walkthrough(env, agent, max_steps=50, output_file=WALKTHROUGH_FILE)
    
    env.close()
