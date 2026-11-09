import os
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from jericho import FrotzEnv

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
ROM_PATH = "advent.z5"
MAX_STEPS = 300
TOTAL_EPISODES = 500000

# Epsilon schedule with a persistent floor so exploration never completely dies
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 200000  # Linear decay over 200k episodes, then holds at 0.05

# --- 2. CURRICULUM STATE BUFFER ---
state_checkpoint_buffer = []

def get_epsilon(episode):
    if episode >= EPS_DECAY_EPISODES:
        return EPS_END
    return EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPISODES)

def train_agent():
    env = FrotzEnv(ROM_PATH)
    
    visited_rooms = set()
    global_best_score = 0

    for episode in range(1, TOTAL_EPISODES + 1):
        env.reset()
        
        # Curriculum Learning: 30% chance to spawn from a saved deep-state checkpoint if available
        if state_checkpoint_buffer and random.random() < 0.3:
            saved_state = random.choice(state_checkpoint_buffer)
            env.set_state(saved_state)

        epsilon = get_epsilon(episode)
        episode_score = 0
        
        recent_actions = deque(maxlen=4)
        episode_rooms = set()

        for step in range(MAX_STEPS):
            legal_actions = env.get_actions()
            if not legal_actions:
                break

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
            
            # --- INTRINSIC NOVELTY REWARD ---
            current_room = env.get_world_state().room if hasattr(env, 'get_world_state') else obs[:50]
            intrinsic_reward = 0.0
            if current_room not in visited_rooms:
                visited_rooms.add(current_room)
                episode_rooms.add(current_room)
                intrinsic_reward = 0.1

            reward = (score - episode_score) + intrinsic_reward
            episode_score = score

            if score > global_best_score or len(episode_rooms) > 15:
                global_best_score = max(global_best_score, score)
                current_state_bytes = env.get_state()
                if current_state_bytes not in state_checkpoint_buffer:
                    state_checkpoint_buffer.append(current_state_bytes)
                    if len(state_checkpoint_buffer) > 50:
                        state_checkpoint_buffer.pop(0)

            if done:
                break

        if episode % 1000 == 0:
            print(f"[EPISODE {episode}] | Mapped Rooms: {len(visited_rooms)} | Transitions: {len(state_checkpoint_buffer)} | Score: {episode_score} | Steps: {step+1}")

def select_greedy_action(env, legal_actions):
    return random.choice(legal_actions)

if __name__ == "__main__":
    train_agent()