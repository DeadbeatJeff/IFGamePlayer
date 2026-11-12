import os
import random
import re
from collections import deque, defaultdict
from jericho import FrotzEnv

# --- 1. CONFIGURATION ---
ROM_PATH = "advent.z5"  # Use the compiled version 5 file
MAX_STEPS = 10000
TOTAL_EPISODES = 1000000
GAMMA = 0.99
LEARNING_RATE = 0.1

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 200000

FALLBACK_ACTIONS = [
    "north", "south", "east", "west", "up", "down", "in", "out",
    "look", "inventory", "score", "help", "take inventory",
    "take keys", "take lantern", "take food", "take bottle",
    "light lamp", "unlock door", "open door", "examine room", "sw", "se", "nw", "ne"
]

q_table = defaultdict(float)
state_checkpoint_buffer = []
state_checkpoint_keys = set()

def get_epsilon(episode):
    if episode >= EPS_DECAY_EPISODES:
        return EPS_END
    return EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPISODES)

def parse_score_from_text(obs, fallback_score):
    if isinstance(obs, tuple):
        obs = obs[0]
    if not isinstance(obs, str):
        obs = str(obs)
    match = re.search(r'score[:\s]+(\d+)', obs, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return fallback_score

def get_state_key(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if not isinstance(obs, str):
        obs = str(obs)
    return obs[:60].strip()

def extract_dynamic_actions(obs, legal_actions):
    if isinstance(obs, tuple):
        obs = obs[0]
    if not isinstance(obs, str):
        obs = str(obs)
        
    if legal_actions and legal_actions != FALLBACK_ACTIONS:
        return legal_actions
    
    words = re.findall(r'\b[a-z]{3,}\b', obs.lower())
    common_ignore = {'the', 'and', 'you', 'are', 'with', 'that', 'this', 'from', 'have'}
    nouns = [w for w in set(words) if w not in common_ignore]
    
    dynamic_actions = list(FALLBACK_ACTIONS)
    for noun in nouns[:5]:
        dynamic_actions.extend([f"take {noun}", f"examine {noun}", f"open {noun}", f"use {noun}"])
        
    return list(set(dynamic_actions))

def select_greedy_action(state_key, legal_actions):
    best_action = legal_actions[0]
    max_q = float('-inf')
    for action in legal_actions:
        q_val = q_table[(state_key, action)]
        if q_val > max_q:
            max_q = q_val
            best_action = action
    return best_action

def train_agent():
    env = FrotzEnv(ROM_PATH)
    
    visited_rooms = set()
    global_best_score = 0

    for episode in range(1, TOTAL_EPISODES + 1):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        if state_checkpoint_buffer and random.random() < 0.3:
            saved_state = random.choice(state_checkpoint_buffer)
            env.set_state(saved_state)

        epsilon = get_epsilon(episode)
        episode_score = 0
        recent_actions = deque(maxlen=4)
        episode_rooms = set()

        for step in range(MAX_STEPS):
            base_actions = env.get_valid_actions()
            legal_actions = extract_dynamic_actions(obs, base_actions)
            if not legal_actions:
                legal_actions = FALLBACK_ACTIONS

            state_key = get_state_key(obs)

            if random.random() < epsilon:
                action = random.choice(legal_actions)
            else:
                action = select_greedy_action(state_key, legal_actions)

            recent_actions.append(action)
            is_looping = len(recent_actions) == 4 and recent_actions[0] == recent_actions[2] and recent_actions[1] == recent_actions[3]
            if is_looping:
                action = random.choice(legal_actions)

            next_obs, score, done, info = env.step(action)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            
            if score == 0:
                score = parse_score_from_text(next_obs, episode_score)

            current_room = env.get_world_state().room if hasattr(env, 'get_world_state') else str(next_obs)[:50]
            intrinsic_reward = 0.0
            if current_room not in visited_rooms:
                visited_rooms.add(current_room)
                episode_rooms.add(current_room)
                intrinsic_reward = 0.1

            reward = (score - episode_score) + intrinsic_reward
            
            next_state_key = get_state_key(next_obs)
            next_legal = env.get_valid_actions()
            if not next_legal:
                next_legal = FALLBACK_ACTIONS
            
            max_next_q = max([q_table[(next_state_key, a)] for a in next_legal]) if next_legal else 0.0
            
            current_q = q_table[(state_key, action)]
            new_q = current_q + LEARNING_RATE * (reward + GAMMA * max_next_q - current_q)
            q_table[(state_key, action)] = new_q

            episode_score = score
            obs = next_obs

            if score > global_best_score:
                global_best_score = score
                current_state = env.get_state()
                
                if isinstance(current_state, tuple):
                    state_bytes_key = str(current_state).encode('utf-8')
                elif hasattr(current_state, 'tobytes'):
                    state_bytes_key = current_state.tobytes()
                else:
                    state_bytes_key = str(current_state).encode('utf-8')
                
                if state_bytes_key not in state_checkpoint_keys:
                    state_checkpoint_buffer.append(current_state)
                    state_checkpoint_keys.add(state_bytes_key)
                    if len(state_checkpoint_buffer) > 50:
                        old_state = state_checkpoint_buffer.pop(0)
                        if isinstance(old_state, tuple):
                            old_key = str(old_state).encode('utf-8')
                        elif hasattr(old_state, 'tobytes'):
                            old_key = old_state.tobytes()
                        else:
                            old_key = str(old_state).encode('utf-8')
                        state_checkpoint_keys.discard(old_key)

            if done:
                break

        if episode % 1000 == 0:
            print(f"[EPISODE {episode}] | Mapped Rooms: {len(visited_rooms)} | Checkpoints: {len(state_checkpoint_buffer)} | Score: {episode_score} | Steps: {step+1}")

if __name__ == "__main__":
    train_agent()