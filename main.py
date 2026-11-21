import random
import re
from collections import deque, defaultdict
from jericho import FrotzEnv

# --- 1. CONFIGURATION ---
ROM_PATH = "advent.z5"
TOTAL_EPISODES = 1000000
GAMMA = 0.99
LEARNING_RATE = 0.15

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 200000

FALLBACK_ACTIONS = [
    "north", "south", "east", "west", "up", "down", "in", "out",
    "ne", "nw", "se", "sw",
    "look", "inventory", "score",
    "get keys", "get lamp", "get food", "get bottle", "get cage", "get rod",
    "on", "light lamp", "unlock grate", "open grate",
    "xyzzy", "plugh", "plover", "fee", "fie", "foe", "foo",
    "wave rod", "pour water", "pour oil", "fill bottle",
    "drop keys", "drop lamp", "drop food", "drop bottle", "drop cage", "drop rod"
]

q_table = defaultdict(float)
state_checkpoint_buffer = []
state_checkpoint_keys = set()
action_cache = {}


def get_epsilon(episode):
    if episode >= EPS_DECAY_EPISODES:
        return EPS_END
    return EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPISODES)


def extract_obs_text(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if not isinstance(obs, str):
        return str(obs)
    return obs


def get_current_score(env, obs_text, step_score, fallback_score):
    if hasattr(env, "get_score"):
        try:
            ram_score = env.get_score()
            if isinstance(ram_score, (int, float)) and ram_score != 0:
                return int(ram_score)
        except Exception:
            pass

    if isinstance(step_score, (int, float)) and step_score != 0:
        return int(step_score)

    match = re.search(r"score[:\s]+(-?\d+)", obs_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return fallback_score


def get_state_key(obs_text):
    lines = [line.strip() for line in obs_text.strip().splitlines() if line.strip()]
    header = lines[0] if lines else obs_text.strip()
    return header[:60]


def get_room_key(obs_text):
    lines = [line.strip() for line in obs_text.strip().splitlines() if line.strip()]
    return (lines[0] if lines else obs_text.strip())[:50]


def get_actions_for_state(env, obs_text, state_key):
    if state_key in action_cache:
        return action_cache[state_key]

    valid = []
    if hasattr(env, "get_valid_actions"):
        try:
            valid = env.get_valid_actions() or []
        except Exception:
            valid = []

    actions = list(dict.fromkeys(list(valid) + FALLBACK_ACTIONS))
    action_cache[state_key] = actions
    return actions


def select_greedy_action(state_key, legal_actions):
    best_action = legal_actions[0]
    max_q = float("-inf")
    for action in legal_actions:
        q_val = q_table[(state_key, action)]
        if q_val > max_q:
            max_q = q_val
            best_action = action
    return best_action


def record_checkpoint(env, force=False):
    if not hasattr(env, "get_state"):
        return
    try:
        current_state = env.get_state()
    except Exception:
        return

    if isinstance(current_state, tuple):
        state_bytes_key = str(current_state).encode("utf-8", errors="ignore")
    elif hasattr(current_state, "tobytes"):
        state_bytes_key = current_state.tobytes()
    else:
        state_bytes_key = str(current_state).encode("utf-8", errors="ignore")

    if state_bytes_key not in state_checkpoint_keys:
        if len(state_checkpoint_buffer) >= 50 and not force:
            return
        state_checkpoint_buffer.append(current_state)
        state_checkpoint_keys.add(state_bytes_key)
        if len(state_checkpoint_buffer) > 50:
            old_state = state_checkpoint_buffer.pop(0)
            if isinstance(old_state, tuple):
                old_key = str(old_state).encode("utf-8", errors="ignore")
            elif hasattr(old_state, "tobytes"):
                old_key = old_state.tobytes()
            else:
                old_key = str(old_state).encode("utf-8", errors="ignore")
            state_checkpoint_keys.discard(old_key)


def train_agent():
    env = FrotzEnv(ROM_PATH)

    visited_rooms = set()
    global_best_score = 0

    print("Training started. Output will stream every 50 episodes:", flush=True)

    for episode in range(1, TOTAL_EPISODES + 1):
        current_max_steps = min(800, 300 + max(0, global_best_score) * 10)

        obs_text = extract_obs_text(env.reset())
        episode_score = get_current_score(env, obs_text, 0, 0)
        if episode_score > global_best_score:
            global_best_score = episode_score

        if state_checkpoint_buffer and random.random() < 0.35:
            saved_state = random.choice(state_checkpoint_buffer)
            try:
                env.set_state(saved_state)
                obs_text = extract_obs_text(env.get_observation() if hasattr(env, "get_observation") else obs_text)
                episode_score = get_current_score(env, obs_text, 0, episode_score)
            except Exception:
                pass

        epsilon = get_epsilon(episode)
        recent_actions = deque(maxlen=4)
        episode_rooms = set()
        step = 0

        for step in range(current_max_steps):
            state_key = get_state_key(obs_text)
            legal_actions = get_actions_for_state(env, obs_text, state_key)

            if random.random() < epsilon:
                action = random.choice(legal_actions)
            else:
                action = select_greedy_action(state_key, legal_actions)

            recent_actions.append(action)
            is_looping = (
                len(recent_actions) == 4
                and recent_actions[0] == recent_actions[2]
                and recent_actions[1] == recent_actions[3]
            )
            if is_looping:
                action = random.choice(legal_actions)

            next_obs, step_score, done, _ = env.step(action)
            next_obs_text = extract_obs_text(next_obs)
            new_score = get_current_score(env, next_obs_text, step_score, episode_score)

            room_key = get_room_key(next_obs_text)
            intrinsic_reward = 0.0
            is_new_room = room_key not in visited_rooms
            if is_new_room:
                visited_rooms.add(room_key)
                episode_rooms.add(room_key)
                intrinsic_reward = 0.1

            score_delta = new_score - episode_score
            reward = (score_delta * 10.0) + intrinsic_reward

            next_state_key = get_state_key(next_obs_text)
            next_legal = get_actions_for_state(env, next_obs_text, next_state_key)
            max_next_q = max([q_table[(next_state_key, a)] for a in next_legal]) if next_legal else 0.0

            current_q = q_table[(state_key, action)]
            q_table[(state_key, action)] = current_q + LEARNING_RATE * (
                reward + GAMMA * max_next_q - current_q
            )

            if new_score > global_best_score:
                global_best_score = new_score
                record_checkpoint(env, force=True)
            elif is_new_room and len(episode_rooms) > 4:
                record_checkpoint(env, force=False)

            episode_score = new_score
            obs_text = next_obs_text

            if done:
                break

        if episode % 50 == 0:
            print(
                f"[EPISODE {episode}] | Mapped Rooms: {len(visited_rooms)} | "
                f"Checkpoints: {len(state_checkpoint_buffer)} | Score: {episode_score} | "
                f"Best Score: {global_best_score} | Max Steps: {current_max_steps} | "
                f"Steps Run: {step + 1}",
                flush=True,
            )


if __name__ == "__main__":
    train_agent()