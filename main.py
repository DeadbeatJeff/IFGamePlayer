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
MAX_STEPS = 600
TOTAL_EPISODES = 500000
GAMMA = 0.99
LEARNING_RATE = 1e-4

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

# --- 2. DEEP REINFORCEMENT RELEVANCE NETWORK (DRRN) ---
class TextEmbeddingEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, 64)
        
    def text_to_tensor(self, text):
        if not isinstance(text, str):
            text = str(text)
        byte_tokens = list(text.encode('utf-8', errors='ignore')[:100])
        if not byte_tokens:
            byte_tokens = [0]
        return torch.tensor(byte_tokens, dtype=torch.long)

    def forward(self, text):
        tokens = self.text_to_tensor(text).unsqueeze(0)
        embedded = self.embedding(tokens).mean(dim=1)
        return torch.relu(self.fc(embedded))

class DRRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = TextEmbeddingEncoder()
        self.action_encoder = TextEmbeddingEncoder()
        self.scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.action_cache = {}

    def forward(self, state_text, action_text):
        s_emb = self.state_encoder(state_text)
        
        if action_text in self.action_cache:
            a_emb = self.action_cache[action_text]
        else:
            a_emb = self.action_encoder(action_text)
            self.action_cache[action_text] = a_emb
            
        combined = torch.cat([s_emb, a_emb], dim=1)
        return self.scorer(combined)

# Initialize Q-Network and Optimizer
q_net = DRRN()
optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# --- 3. CURRICULUM STATE BUFFER ---
state_checkpoint_buffer = []
state_checkpoint_keys = set()

def get_epsilon(episode):
    if episode >= EPS_DECAY_EPISODES:
        return EPS_END
    return EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPISODES)

def parse_score_from_text(obs, fallback_score):
    match = re.search(r'score[:\s]+(\d+)', obs, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return fallback_score

def select_greedy_action(state_text, legal_actions):
    q_net.eval()
    with torch.no_grad():
        best_action = legal_actions[0]
        max_q = float('-inf')
        for action in legal_actions:
            q_val = q_net(state_text, action).item()
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
        
        # Curriculum Learning: 30% chance to spawn from a saved deep-state checkpoint
        if state_checkpoint_buffer and random.random() < 0.3:
            saved_state = random.choice(state_checkpoint_buffer)
            env.set_state(saved_state)

        epsilon = get_epsilon(episode)
        episode_score = 0
        
        recent_actions = deque(maxlen=4)
        episode_rooms = set()

        for step in range(MAX_STEPS):
            legal_actions = env.get_valid_actions()
            if not legal_actions:
                legal_actions = FALLBACK_ACTIONS

            if random.random() < epsilon:
                action = random.choice(legal_actions)
            else:
                action = select_greedy_action(obs, legal_actions)

            recent_actions.append(action)
            is_looping = len(recent_actions) == 4 and recent_actions[0] == recent_actions[2] and recent_actions[1] == recent_actions[3]
            if is_looping:
                action = random.choice(legal_actions)

            next_obs, score, done, info = env.step(action)
            
            if score == 0:
                score = parse_score_from_text(next_obs, episode_score)

            current_room = env.get_world_state().room if hasattr(env, 'get_world_state') else next_obs[:50]
            intrinsic_reward = 0.0
            if current_room not in visited_rooms:
                visited_rooms.add(current_room)
                episode_rooms.add(current_room)
                intrinsic_reward = 0.1

            reward = (score - episode_score) + intrinsic_reward
            
            # --- Q-NETWORK UPDATE (DRRN STEP) ---
            q_net.train()
            optimizer.zero_grad()
            
            # Compute next state max Q under no_grad to decouple graph creation
            with torch.no_grad():
                next_legal = env.get_valid_actions()
                if not next_legal:
                    next_legal = FALLBACK_ACTIONS
                eval_actions = next_legal[:10] if len(next_legal) > 10 else next_legal
                next_q_values = [q_net(next_obs, a).item() for a in eval_actions]
                max_next_q = max(next_q_values) if next_q_values else 0.0
                target_q = reward + (0.0 if done else GAMMA * max_next_q)
                target_tensor = torch.tensor([[target_q]], dtype=torch.float32)

            # Compute current Q in training mode
            current_q = q_net(obs, action)

            loss = loss_fn(current_q, target_tensor)
            loss.backward()
            optimizer.step()

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

        if episode % 100 == 0:
            print(f"[EPISODE {episode}] | Mapped Rooms: {len(visited_rooms)} | Checkpoints: {len(state_checkpoint_buffer)} | Score: {episode_score} | Steps: {step+1}")

if __name__ == "__main__":
    train_agent()