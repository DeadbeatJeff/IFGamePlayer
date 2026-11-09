import os
import random
import re
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from jericho import FrotzEnv

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
ROM_PATH = "advent.z5"
MAX_STEPS = 300
TOTAL_EPISODES = 500000

# Epsilon schedule with a persistent floor so exploration never dies out
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 200000

# Standard fallback action vocabulary for unsupported ROMs where get_valid_actions() fails
FALLBACK_ACTIONS = [
    "north", "south", "east", "west", "up", "down", "in", "out",
    "look", "inventory", "score", "help", "take inventory",
    "take keys", "take lantern", "take food", "take bottle",
    "light lamp", "unlock door", "open door", "examine room", "sw", "se", "nw", "ne"
]

# --- 2. CURRICULUM STATE BUFFER ---
state_checkpoint_buffer = []
state_checkpoint_keys = set()

def get_epsilon(episode):
    if episode >= EPS_DECAY_EPISODES:
        return EPS_END
    return EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPISODES)

def parse_score_from_text(obs, fallback_score):
    """Fallback text parser for unsupported ROMs where RAM score tracking is disabled."""
    match = re.search(r'score[:\s]+(\d+)', obs, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return fallback_score

def train_agent():
    env = FrotzEnv(ROM_PATH)
    
    visited_rooms = set()
    global_best_score = 0

    for episode in range(1, TOTAL_EPISODES + 1):
        env.reset()
        
        # Curriculum Learning: 30% chance to spawn from a saved deep-state checkpoint
        if state_checkpoint_buffer and random.random() < 0.3:
            saved_state = random.choice(state_checkpoint_buffer)
            env.set_state(saved_state)

        epsilon = get_epsilon(episode)
        episode_score = 0
        
        recent_actions = deque(maxlen=4)
        episode_rooms = set()

        for step in range(MAX_STEPS):
            # Fetch valid actions; use FALLBACK_ACTIONS if Jericho returns empty for unsupported ROM
            legal_actions = env.get_valid_actions()
            if not legal_actions:
                legal_actions = FALLBACK_ACTIONS

            # Epsilon-greedy action selection with loop avoidance
            if random.random() < epsilon:
                action = random.choice(legal_actions)
            else:
                action = select_greedy_action(env, legal_actions)

            # --- ACTION REPETITION / LOOP PENALTY ---
            recent_actions.append(action)
            is_looping = len(recent_actions) == 4 and recent_actions[0] == recent_actions[2] and recent_actions[1] == recent_actions[3]
            if is_looping:
                action = random.choice(legal_actions)

            # Step environment
            obs, score, done, info = env.step(action)
            
            # If RAM score tracking is disabled for advent.z5, parse score from observation text
            if score == 0:
                score = parse_score_from_text(obs, episode_score)

            # --- INTRINSIC NOVELTY REWARD ---
            current_room = env.get_world_state().room if hasattr(env, 'get_world_state') else obs[:50]
            intrinsic_reward = 0.0
            if current_room not in visited_rooms:
                visited_rooms.add(current_room)
                episode_rooms.add(current_room)
                intrinsic_reward = 0.1

            reward = (score - episode_score) + intrinsic_reward
            episode_score = score

            if score > global_best_score or len(episode_rooms) > 5:
                global_best_score = max(global_best_score, score)
                current_state = env.get_state()
                
                # Safely generate a unique key regardless of whether state is a tuple, array, or bytes
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

        if episode % 100 == 0:
            print(f"[EPISODE {episode}] | Mapped Rooms: {len(visited_rooms)} | Checkpoints: {len(state_checkpoint_buffer)} | Score: {episode_score} | Steps: {step+1}")

def select_greedy_action(env, legal_actions):
    return random.choice(legal_actions)

if __name__ == "__main__":
    train_agent()