import gzip
import os
import pickle
import random
import re
from collections import deque, defaultdict
from jericho import FrotzEnv

# --- 1. CONFIGURATION ---
ROM_PATH = "advent.z5"
CHECKPOINT_FILE = "advent_q_checkpoint.pkl.gz"
WALKTHROUGH_FILE = "best_walkthrough.txt"

ADDITIONAL_EPISODES = 1_000_000  # Runs this many MORE episodes every time you launch
SAVE_EVERY_EPISODES = 2500

GAMMA = 0.99
LEARNING_RATE = 0.15

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 200_000

FALLBACK_ACTIONS = [
    "north", "south", "east", "west", "up", "down", "in", "out",
    "ne", "nw", "se", "sw",
    "look", "inventory", "score",
    "get keys", "get lamp", "get food", "get bottle", "get cage", "get rod",
    "get bird", "get gold", "get diamonds", "get silver", "get jewelry", "get coins",
    "on", "light lamp", "unlock grate", "open grate", "catch bird", "free bird",
    "xyzzy", "plugh", "plover", "fee", "fie", "foe", "foo",
    "wave rod", "pour water", "pour oil", "fill bottle",
    "drop keys", "drop lamp", "drop food", "drop bottle", "drop cage", "drop rod",
    "drop bird", "drop gold", "drop diamonds", "drop silver", "drop jewelry", "drop coins"
]

q_table = defaultdict(float)
# Stores tuples of (z_machine_state, obs_text, score) so Go-Explore restores cleanly
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
            if isinstance(ram_score, (int, float)):
                return int(ram_score)
        except Exception:
            pass

    if isinstance(step_score, (int, float)):
        return int(step_score)

    match = re.search(r"score[:\s]+(-?\d+)", obs_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return fallback_score


def get_inventory_signature(env):
    """Distinguishes identical rooms when holding different key items."""
    if not hasattr(env, "get_inventory"):
        return ""
    try:
        inv = env.get_inventory()
        if not inv:
            return "empty"
        names = sorted(str(item.name).lower() for item in inv if hasattr(item, "name"))
        return ",".join(names)[:80]
    except Exception:
        return ""


def get_state_key(env, obs_text):
    lines = [line.strip() for line in obs_text.strip().splitlines() if line.strip()]
    header = lines[0] if lines else obs_text.strip()
    inv_sig = get_inventory_signature(env)
    return f"{header[:60]}|{inv_sig}"


def get_room_key(obs_text):
    lines = [line.strip() for line in obs_text.strip().splitlines() if line.strip()]
    return (lines[0] if lines else obs_text.strip())[:50]


def get_actions_for_state(env, state_key):
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


def _state_to_bytes_key(state_obj):
    if isinstance(state_obj, tuple):
        return str(state_obj).encode("utf-8", errors="ignore")
    if hasattr(state_obj, "tobytes"):
        return state_obj.tobytes()
    return str(state_obj).encode("utf-8", errors="ignore")


def record_checkpoint(env, obs_text, score, force=False):
    if not hasattr(env, "get_state"):
        return
    try:
        current_state = env.get_state()
    except Exception:
        return

    state_bytes_key = _state_to_bytes_key(current_state)

    if state_bytes_key not in state_checkpoint_keys:
        if len(state_checkpoint_buffer) >= 75 and not force:
            return
        state_checkpoint_buffer.append((current_state, obs_text, score))
        state_checkpoint_keys.add(state_bytes_key)
        if len(state_checkpoint_buffer) > 75:
            old_state, _, _ = state_checkpoint_buffer.pop(0)
            state_checkpoint_keys.discard(_state_to_bytes_key(old_state))


def save_training_state(filepath, episode, visited_rooms, global_best_score):
    """Atomically saves Q-table, Go-Explore snapshots, and progress to disk."""
    payload = {
        "episode": episode,
        "q_table": dict(q_table),
        "state_checkpoint_buffer": state_checkpoint_buffer,
        "visited_rooms": visited_rooms,
        "global_best_score": global_best_score,
    }
    tmp_path = filepath + ".tmp"
    with gzip.open(tmp_path, "wb", compresslevel=3) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, filepath)


def load_training_state(filepath):
    """Loads saved Q-table and Go-Explore snapshots if available."""
    global q_table, state_checkpoint_buffer, state_checkpoint_keys

    if not os.path.exists(filepath):
        return 0, set(), 0

    with gzip.open(filepath, "rb") as f:
        payload = pickle.load(f)

    q_table = defaultdict(float, payload.get("q_table", {}))
    state_checkpoint_buffer = payload.get("state_checkpoint_buffer", [])
    state_checkpoint_keys = {
        _state_to_bytes_key(entry[0]) for entry in state_checkpoint_buffer
    }
    last_episode = payload.get("episode", 0)
    visited_rooms = payload.get("visited_rooms", set())
    global_best_score = payload.get("global_best_score", 0)

    return last_episode, visited_rooms, global_best_score


def save_walkthrough(score, episode, action_history):
    with open(WALKTHROUGH_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Best Score: {score} (Episode {episode})\n")
        for act in action_history:
            f.write(f"{act}\n")


def train_agent():
    env = FrotzEnv(ROM_PATH)

    last_episode, visited_rooms, global_best_score = load_training_state(CHECKPOINT_FILE)
    start_episode = last_episode + 1
    end_episode = last_episode + ADDITIONAL_EPISODES

    if last_episode > 0:
        print(
            f"Resumed from {CHECKPOINT_FILE} at episode {start_episode} | "
            f"Q-entries: {len(q_table)} | Mapped Rooms: {len(visited_rooms)} | "
            f"Checkpoints: {len(state_checkpoint_buffer)} | Best Score: {global_best_score}",
            flush=True,
        )
    else:
        print(
            f"No checkpoint found. Starting fresh for {ADDITIONAL_EPISODES:,} episodes.",
            flush=True,
        )

    try:
        for episode in range(start_episode, end_episode + 1):
            current_max_steps = min(800, 300 + max(0, global_best_score) * 10)

            obs_text = extract_obs_text(env.reset())
            episode_score = get_current_score(env, obs_text, 0, 0)
            if episode_score > global_best_score:
                global_best_score = episode_score

            action_history = []
            restored_from_snapshot = False

            if state_checkpoint_buffer and random.random() < 0.35:
                saved_state, saved_obs, saved_score = random.choice(state_checkpoint_buffer)
                try:
                    env.set_state(saved_state)
                    obs_text = saved_obs
                    episode_score = get_current_score(env, obs_text, saved_score, saved_score)
                    restored_from_snapshot = True
                except Exception:
                    pass

            epsilon = get_epsilon(episode)
            recent_actions = deque(maxlen=4)
            episode_rooms = set()
            step = 0

            for step in range(current_max_steps):
                state_key = get_state_key(env, obs_text)
                legal_actions = get_actions_for_state(env, state_key)

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

                action_history.append(action)
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

                next_state_key = get_state_key(env, next_obs_text)
                next_legal = get_actions_for_state(env, next_state_key)
                max_next_q = (
                    max([q_table[(next_state_key, a)] for a in next_legal])
                    if next_legal
                    else 0.0
                )

                current_q = q_table[(state_key, action)]
                q_table[(state_key, action)] = current_q + LEARNING_RATE * (
                    reward + GAMMA * max_next_q - current_q
                )

                if new_score > global_best_score:
                    global_best_score = new_score
                    record_checkpoint(env, next_obs_text, new_score, force=True)
                    save_training_state(
                        CHECKPOINT_FILE, episode, visited_rooms, global_best_score
                    )
                    if not restored_from_snapshot:
                        save_walkthrough(global_best_score, episode, action_history)
                    print(
                        f"*** NEW BEST SCORE: {global_best_score} at Episode {episode} "
                        f"(Checkpoint saved) ***",
                        flush=True,
                    )
                elif is_new_room and len(episode_rooms) > 4:
                    record_checkpoint(env, next_obs_text, new_score, force=False)

                episode_score = new_score
                obs_text = next_obs_text

                if done:
                    break

            if episode % 50 == 0:
                print(
                    f"[EPISODE {episode}] | Mapped Rooms: {len(visited_rooms)} | "
                    f"Checkpoints: {len(state_checkpoint_buffer)} | Q-States: {len(q_table)} | "
                    f"Score: {episode_score} | Best: {global_best_score} | "
                    f"Eps: {epsilon:.3f} | Steps: {step + 1}",
                    flush=True,
                )

            if episode % SAVE_EVERY_EPISODES == 0:
                save_training_state(
                    CHECKPOINT_FILE, episode, visited_rooms, global_best_score
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user! Saving checkpoint before exit...", flush=True)
        save_training_state(CHECKPOINT_FILE, episode, visited_rooms, global_best_score)
        print(f"Saved cleanly at episode {episode} to {CHECKPOINT_FILE}.", flush=True)


if __name__ == "__main__":
    train_agent()