#!/usr/bin/env python3
"""
IFGamePlayer - Jericho-Backed RL Game Engine & Map Analyzer
"""

import hashlib
import json
import os
import re
import random
import signal
import sys
from typing import Dict, List, Optional, Set, Tuple, Any

try:
    import jericho
except ImportError:
    print("[Error] Jericho is not installed. Run: pip install jericho")
    sys.exit(1)


# =====================================================================
# 1. DATA STRUCTURES & GRAPH REPRESENTATION
# =====================================================================

class RoomNode:
    def __init__(self, room_id: str, description: str):
        self.room_id = room_id
        self.description = description
        self.items: Set[str] = set()
        self.exits: Dict[str, str] = {}
        self.blocked_actions: Set[str] = set()
        self.lifetime_visits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "description": self.description,
            "items": sorted(list(self.items)),
            "exits": self.exits,
            "blocked_actions": sorted(list(self.blocked_actions)),
            "lifetime_visits": self.lifetime_visits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoomNode":
        node = cls(room_id=data["room_id"], description=data.get("description", ""))
        node.items = set(data.get("items", []))
        node.exits = data.get("exits", {})
        node.blocked_actions = set(data.get("blocked_actions", []))
        node.lifetime_visits = data.get("lifetime_visits", 0)
        return node


class WorldGraph:
    def __init__(self):
        self.nodes: Dict[str, RoomNode] = {}
        self.current_room_id: Optional[str] = None

    def get_or_create_node(self, room_id: str, description: str) -> RoomNode:
        if room_id not in self.nodes:
            self.nodes[room_id] = RoomNode(room_id, description)
        elif description and self.nodes[room_id].description in ["Unknown Area", ""]:
            self.nodes[room_id].description = description
        return self.nodes[room_id]

    def add_transition(self, from_id: str, action: str, to_id: str):
        if from_id in self.nodes:
            self.nodes[from_id].exits[action] = to_id

    def total_transitions(self) -> int:
        return sum(len(node.exits) for node in self.nodes.values())

    def total_unique_items(self) -> int:
        items = set()
        for node in self.nodes.values():
            items.update(node.items)
        return len(items)

    def print_summary(self):
        total_rooms = len(self.nodes)
        total_edges = self.total_transitions()
        total_items = self.total_unique_items()
        print("=" * 65)
        print("           IFGamePlayer World Graph Statistics")
        print("=" * 65)
        print(f" Total Unique Rooms Mapped : {total_rooms}")
        print(f" Total Mapped Transitions  : {total_edges}")
        print(f" Unique Items Discovered   : {total_items}")
        print("=" * 65)
        print(f"\n{'ID':<12} | {'Visits':<8} | {'Exits':<6} | {'Title / Items'}")
        print("-" * 65)
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.lifetime_visits, reverse=True)
        for n in sorted_nodes:
            desc = n.description[:30]
            items = f" [Items: {', '.join(n.items)}]" if n.items else ""
            print(f"{n.room_id:<12} | {n.lifetime_visits:<8,d} | {len(n.exits):<6d} | {desc}{items}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_room_id": self.current_room_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldGraph":
        graph = cls()
        graph.current_room_id = data.get("current_room_id")
        nodes_data = data.get("nodes", {})
        for nid, ndict in nodes_data.items():
            graph.nodes[nid] = RoomNode.from_dict(ndict)
        return graph


# =====================================================================
# 2. DISK SERIALIZATION & LOGGING
# =====================================================================

class GraphStorage:
    @staticmethod
    def save_graph(graph: WorldGraph, filepath: str) -> None:
        temp_path = f"{filepath}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(graph.to_dict(), f, indent=2)
            os.replace(temp_path, filepath)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def load_graph(filepath: str) -> WorldGraph:
        if not os.path.exists(filepath):
            return WorldGraph()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorldGraph.from_dict(data)
        except Exception:
            return WorldGraph()


class EpisodeLogger:
    def __init__(self, log_path: str = "recent_episodes.txt"):
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("=== IFGamePlayer Episode Log ===\n\n")

    def append_episode(self, episode_num: int, steps: int, score: int, actions: List[str], rooms_found: int):
        summary_actions = " -> ".join(actions[:15]) + (f" ... [{len(actions)-15} more]" if len(actions) > 15 else "")

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"Episode #{episode_num} | Steps: {steps} | Score: {score} | Mapped Rooms: {rooms_found}\n")
                f.write(f"Command Sequence: {summary_actions}\n")
                f.write("-" * 70 + "\n")
                f.flush()
        except Exception as e:
            print(f"[Logger] Error writing log: {e}", file=sys.stderr)


# =====================================================================
# 3. JERICHO AGENT INTERFACE
# =====================================================================

class JerichoAgent:
    CARDINAL_DIRECTIONS = [
        "north", "south", "east", "west", 
        "ne", "nw", "se", "sw", 
        "up", "down", "in", "out", "enter", "exit", "climb"
    ]

    def __init__(self, save_path: str = "advent_world_graph.json"):
        self.save_path = save_path
        self.graph = GraphStorage.load_graph(self.save_path)
        self.episode_visits: Dict[str, int] = {}
        self.last_action: Optional[str] = None

    def process_step(self, env: jericho.FrotzEnv, observation: str) -> str:
        # Query RAM directly for player location and surrounding objects
        
        try:
            player_loc = env.get_player_location()
        except (AttributeError, ValueError):
            player_loc = None

        if player_loc:
            room_title = player_loc.name
            room_num = player_loc.num
            room_id = f"room_{room_num}"
        else:
            room_title = "Unknown Area"
            room_id = hashlib.md5(observation[:30].encode()).hexdigest()[:10]

        node = self.graph.get_or_create_node(room_id, room_title)

        # Retrieve room items directly from Z-Machine object tree
        try:
            items = [obj.name for obj in env.get_surrounding_objects() if obj != player_loc]
            node.items.update(items)
        except Exception:
            pass

        node.lifetime_visits += 1
        self.episode_visits[room_id] = self.episode_visits.get(room_id, 0) + 1

        # Track graph state transitions
        if self.graph.current_room_id and self.last_action:
            if self.graph.current_room_id != room_id:
                self.graph.add_transition(
                    from_id=self.graph.current_room_id,
                    action=self.last_action,
                    to_id=room_id
                )
            else:
                prev_node = self.graph.nodes[self.graph.current_room_id]
                prev_node.blocked_actions.add(self.last_action)

        self.graph.current_room_id = room_id

        # Get candidates (Jericho provides valid candidate actions or fallback to directions)
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            valid_actions = [a for a in self.CARDINAL_DIRECTIONS if a not in node.blocked_actions]

        action = self._select_action(node, valid_actions)
        self.last_action = action
        return action

    def _select_action(self, node: RoomNode, candidate_actions: List[str]) -> str:
        if random.random() < 0.15:
            return random.choice(candidate_actions)

        unvisited = [a for a in candidate_actions if a not in node.exits and a not in node.blocked_actions]
        if unvisited:
            return random.choice(unvisited)

        known = [a for a in candidate_actions if a in node.exits]
        if known:
            known.sort(key=lambda a: self.episode_visits.get(node.exits[a], 0))
            return known[0]

        return random.choice(candidate_actions)

    def close(self):
        GraphStorage.save_graph(self.graph, self.save_path)


# =====================================================================
# 4. MAIN EXECUTION LOOP
# =====================================================================

def run_jericho_episode(rom_path: str, agent: JerichoAgent, max_steps: int = 300) -> Tuple[bool, int, int, List[str]]:
    env = jericho.FrotzEnv(rom_path)
    obs, info = env.reset()

    executed_actions = []
    game_won = False
    step_count = 0
    final_score = 0

    for step in range(1, max_steps + 1):
        step_count = step
        action = agent.process_step(env, obs)
        executed_actions.append(action)

        obs, reward, done, info = env.step(action)
        final_score = info.get("score", 0)

        if done or env.game_over():
            if final_score >= 350:
                game_won = True
            break

    env.close()
    return game_won, step_count, final_score, executed_actions


if __name__ == "__main__":
    if "--view" in sys.argv or "-v" in sys.argv:
        graph = GraphStorage.load_graph("advent_world_graph.json")
        graph.print_summary()
        sys.exit(0)

    ROM_PATH = "advent.z5"
    if not os.path.exists(ROM_PATH):
        print(f"[Error] Could not find '{ROM_PATH}' in current directory.")
        sys.exit(1)

    MAX_EPISODES = 500000
    STEPS_PER_EPISODE = 300
    SAVE_PATH = "advent_world_graph.json"

    agent = JerichoAgent(save_path=SAVE_PATH)
    logger = EpisodeLogger(log_path="recent_episodes.txt")

    def handle_signal(signum, frame):
        print("\n[Runner] Gracefully saving world graph...")
        agent.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("============================================================")
    print(" IFGamePlayer - Jericho Environment Interface")
    print(f" Target Game ROM   : {ROM_PATH}")
    print(f" Saved Graph Rooms : {len(agent.graph.nodes)}")
    print("============================================================\n")

    for episode in range(1, MAX_EPISODES + 1):
        agent.episode_visits.clear()
        agent.graph.current_room_id = None
        agent.last_action = None

        won, steps_used, score, actions = run_jericho_episode(
            rom_path=ROM_PATH,
            agent=agent,
            max_steps=STEPS_PER_EPISODE
        )

        logger.append_episode(
            episode_num=episode,
            steps=steps_used,
            score=score,
            actions=actions,
            rooms_found=len(agent.graph.nodes)
        )
        agent.close()

        print(
            f"[EPISODE {episode:5d}] | "
            f"Mapped Rooms: {len(agent.graph.nodes):3d} | "
            f"Transitions: {agent.graph.total_transitions():4d} | "
            f"Score: {score:3d} | "
            f"Steps: {steps_used:3d}"
        )
        sys.stdout.flush()

        if won:
            print(f"\n🏆 VICTORY DETECTED on Episode {episode}! 🏆")
            sys.exit(0)