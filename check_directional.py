"""Independent simulator smoke test; no detector or cloud model is loaded."""
import json
from pathlib import Path
from visual_nav.habitat_io import HabitatIO
from visual_nav.directional import direction_vector, move_straight


if __name__=='__main__':
    output=Path('outputs/directional_smoke_20261106')
    output.mkdir(exist_ok=False)
    sim=HabitatIO('apt_0',output,realtime=False)
    try:
        start=sim.position.copy()
        forward=sim._executor.controller.prev_orientation
        direction=next((d for d in ('forward','left','right','backward')
            if sim._world.direct(start,start+direction_vector(forward,d)*.4)),None)
        if direction is None:raise RuntimeError('No short open direction at test start')
        first=move_straight(sim._executor,sim._world,direction,.3)
        print(json.dumps(first),flush=True)
        assert first['status']=='succeeded' and abs(first['displacement_m']-.3)<.04
        second=move_straight(sim._executor,sim._world,'forward',20)
        print(json.dumps(second),flush=True)
        assert second['status']=='blocked' and second['displacement_m']<20
        (output/'result.json').write_text(json.dumps([first,second],indent=2))
    finally:
        sim.close()
