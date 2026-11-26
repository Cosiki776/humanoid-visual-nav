"""Render labelled target views for manual color verification (no model API)."""
import json
from pathlib import Path

import bootstrap
import magnum as mn
import numpy as np
from PIL import Image, ImageDraw
from simulation.world import World


def main():
    output = bootstrap.ROOT / 'outputs' / 'color_review'
    output.mkdir(parents=True, exist_ok=True)
    world = World('apt_0')
    try:
        annotations = json.loads((bootstrap.ROOT / 'legacy_gt/annotations.json').read_text())
        scene = world.build_map(annotations)
        targets = [o for o in scene['objects'] if o['goal_nodes']]
        assert all(o['attributes'].get('color') == annotations[o['id']]['color'] for o in targets)
        for index, obj in enumerate(targets):
            center = (np.array(obj['bbox_min']) + obj['bbox_max']) / 2
            goals = [np.array(scene['nodes'][n]) for n in obj['goal_nodes']]
            sheet = Image.new('RGB', (1536, 550), 'white')
            draw = ImageDraw.Draw(sheet)
            for view, point in enumerate((goals[0], goals[-1])):
                eye = point.copy()
                eye[1] = max(center[1] + 0.7, 1.3)
                world.sim.get_agent(0).scene_node.transformation = mn.Matrix4.look_at(
                    mn.Vector3(eye), mn.Vector3(center), mn.Vector3.y_axis())
                frame = world.sim.get_sensor_observations()['rgb'][:, :, :3]
                sheet.paste(Image.fromarray(frame), (view * 768, 35))
            draw.text((10, 10), obj['id'], fill='black')
            sheet.save(output / f'{index:02d}.jpg')
        (output / 'targets.json').write_text(json.dumps(targets, indent=2))
        print(output, flush=True)
    finally:
        world.close()


if __name__ == '__main__':
    main()
