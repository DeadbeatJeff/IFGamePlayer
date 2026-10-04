#!/usr/bin/env python3
"""
view_map.py - Terminal & Graphviz Inspection Tool for IFGamePlayer World Graph
Run on sdf.org to view mapped rooms, exits, items, and topology metrics.
"""

import json
import sys
import os

def load_graph(filepath: str = "advent_world_graph.json") -> dict:
    if not os.path.exists(filepath):
        print(f"Error: Save file '{filepath}' not found.")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def print_summary(data: dict):
    nodes = data.get("nodes", {})
    total_rooms = len(nodes)
    total_edges = sum(len(n.get("exits", {})) for n in nodes.values())
    
    all_items = set()
    for n in nodes.values():
        all_items.update(n.get("items", []))

    print("=" * 60)
    print("           IFGamePlayer World Graph Statistics")
    print("=" * 60)
    print(f" Total Unique Rooms Mapped : {total_rooms}")
    print(f" Total Directional Exits   : {total_edges}")
    print(f" Unique Items Discovered   : {len(all_items)} -> {sorted(list(all_items))}")
    print("=" * 60)

def print_room_list(data: dict):
    nodes = data.get("nodes", {})
    print(f"\n{'ID':<12} | {'Visits':<8} | {'Exits':<6} | {'Title / Items'}")
    print("-" * 65)
    
    # Sort rooms by lifetime visit count
    sorted_nodes = sorted(nodes.values(), key=lambda x: x.get("lifetime_visits", 0), reverse=True)
    
    for n in sorted_nodes:
        nid = n.get("room_id", "???")
        visits = n.get("lifetime_visits", 0)
        exits_count = len(n.get("exits", {}))
        desc = n.get("description", "Unknown")[:30]
        items = f" [Items: {', '.join(n.get('items'))}]" if n.get("items") else ""
        
        print(f"{nid:<12} | {visits:<8,d} | {exits_count:<6d} | {desc}{items}")

def export_dot(data: dict, dot_filepath: str = "cave_map.dot"):
    nodes = data.get("nodes", {})
    with open(dot_filepath, "w", encoding="utf-8") as f:
        f.write("digraph ColossalCaveMap {\n")
        f.write("  rankdir=LR;\n")
        f.write('  node [shape=box, style=rounded, fontname="Helvetica"];\n\n')

        for nid, node in nodes.items():
            label = f"{node.get('description', nid)[:25]}\\n(visits: {node.get('lifetime_visits', 0)})"
            f.write(f'  "{nid}" [label="{label}"];\n')

        f.write("\n")
        for nid, node in nodes.items():
            for dir_name, target_id in node.get("exits", {}).items():
                f.write(f'  "{nid}" -> "{target_id}" [label="{dir_name}"];\n')

        f.write("}\n")
    print(f"\n[Export] Graphviz DOT file written to '{dot_filepath}'.")

if __name__ == "__main__":
    filepath = "advent_world_graph.json"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        filepath = sys.argv[1]

    data = load_graph(filepath)
    print_summary(data)
    print_room_list(data)

    if "--dot" in sys.argv or "-d" in sys.argv:
        export_dot(data)