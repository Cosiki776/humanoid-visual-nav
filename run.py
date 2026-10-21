#!/usr/bin/env python3
"""Export GT, test humanoid execution, or run bounded language-model planning."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import bootstrap
import numpy as np
from planning import ModelPlanner, smoke_plan, validate_plan


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["export", "smoke", "agent", "replay"], default="export")
    p.add_argument("--scene", default="apt_0")
    p.add_argument("--instruction")
    p.add_argument("--target", help="Exact GT object ID for smoke test")
    p.add_argument("--start", help="Roadmap start node (defaults to a point far from target)")
    p.add_argument("--plan", type=Path, help="JSON plan for replay")
    p.add_argument("--annotations", type=Path, default=bootstrap.ROOT / "annotations.json")
    p.add_argument("--output", type=Path)
    p.add_argument("--max-replans", type=int, default=2)
    p.add_argument("--radius", type=float, default=0.3)
    args = p.parse_args()
    if args.mode == "agent" and not args.instruction:
        p.error("agent mode requires --instruction")
    if args.mode == "replay" and not args.plan:
        p.error("replay mode requires --plan")
    if args.max_replans < 0 or not 0.1 <= args.radius <= 1:
        p.error("Invalid max-replans or radius")
    planner = ModelPlanner() if args.mode == "agent" else None
    output = args.output or bootstrap.ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    from world import World, vector
    world, writer = None, None
    result = {"mode": args.mode, "instruction": args.instruction, "status": "initializing",
              "events": [], "limitation": "kinematic humanoid; circular NavMesh clearance, not full-body collision certification"}
    try:
        world = World(args.scene, args.radius)
        annotations = json.loads(args.annotations.read_text()) if args.annotations.exists() else {}
        scene = world.build_map(annotations)
        save(output / "scene.json", scene)
        targets = [o for o in scene["objects"] if o["goal_nodes"]]
        if not targets:
            raise RuntimeError("No reachable furniture targets")
        target = next((o for o in targets if o["id"] == args.target), None) if args.target else next(
            (o for o in targets if o["category"] == "chair"), targets[0])
        if target is None:
            raise ValueError("Requested target has no reachable goal nodes")
        replay = json.loads(args.plan.read_text()) if args.mode == "replay" else None
        start = args.start or (replay["route"][0] if replay else max(scene["nodes"], key=lambda n:
            min(np.linalg.norm(np.array(scene["nodes"][n])-scene["nodes"][g]) for g in target["goal_nodes"])))
        if start not in scene["nodes"]:
            raise ValueError("Unknown start node")
        scene["current_node"] = start
        save(output / "scene.json", scene)
        print(f"GT: {len(scene['objects'])} objects, {len(scene['nodes'])} nodes, {len(scene['edges'])} edges", flush=True)
        print(f"start={start}; example target={target['id']}; output={output}", flush=True)
        if args.mode == "export":
            result["status"] = "exported"
            return
        import imageio.v2 as imageio
        from executor import Executor
        writer = imageio.get_writer(str(output / "walk.mp4"), fps=15, macro_block_size=16)
        executor = Executor(world, scene["nodes"][start], writer.append_data)
        feedback = None
        for attempt in range(args.max_replans+1):
            scene["current_node"] = start
            try:
                plan = (planner.plan(args.instruction, scene, start, feedback) if planner else
                        replay if replay else smoke_plan(scene, start, target["id"]))
                if isinstance(plan, dict) and plan.get("status") == "needs_clarification":
                    result.update(status="needs_clarification", message=plan.get("message"))
                    break
                validate_plan(plan, scene, start)
            except (ValueError, KeyError, TypeError) as exc:
                feedback = {"status": "invalid_plan", "reason": str(exc)}
                result["events"].append(feedback)
                result["status"] = "invalid_plan"
                if not planner:
                    raise
                continue
            save(output / f"plan_{attempt}.json", plan)
            result["events"].append({"attempt": attempt, "plan": plan})
            blocked = False
            for node in plan["route"][1:]:
                previous = start
                event = executor.execute_segment(scene["nodes"][node])
                event["target_node"] = node
                result["events"].append(event)
                print(json.dumps(event), flush=True)
                if event["status"] != "succeeded":
                    blocked = True
                    # Replanning starts from ACTUAL pose, never the last intended waypoint.
                    start = f"current_{attempt}"
                    scene["nodes"][start] = vector(executor.position)
                    scene["edges"] = [e for e in scene["edges"] if set(e) != {previous, node}]
                    for key, pos in list(scene["nodes"].items()):
                        if key != start and np.linalg.norm(executor.position-pos) <= 1.65 and world.direct(executor.position, pos):
                            scene["edges"].append([start, key])
                    feedback = event
                    result["status"] = event["status"]
                    break
                start = node
            if not blocked:
                destination = scene["nodes"][plan["route"][-1]]
                error = float(np.linalg.norm(executor.position-destination))
                result.update(status="succeeded" if error < 0.1 else "failed_arrival",
                              target_object=plan["target_object"], final_position=vector(executor.position),
                              final_error_m=error)
                for _ in range(30):
                    executor.stop()
                break
            if not planner:
                break
        save(output / "trajectory.json", executor.trace)
        save(output / "scene_final.json", scene)
        imageio.imwrite(output / "final.png", world.frame(executor.position))
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if writer is not None:
            writer.close()
        if world is not None:
            world.close()
        save(output / "result.json", result)
        print(f"RESULT: {result['status']} -> {output}", flush=True)
    if result["status"] not in {"succeeded", "needs_clarification"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
