"""Visual surface based stopping geometry; never uses simulator object metadata."""
import numpy as np

MIN_GAP = .35
MAX_GAP = .65
PREFERRED_GAP = .5


def arrival_geometry(position, instance):
    p = np.asarray(position)[[0, 2]]
    lo, hi = np.asarray(instance['bbox_min'])[[0, 2]], np.asarray(instance['bbox_max'])[[0, 2]]
    gap = float(np.linalg.norm(np.maximum(np.maximum(lo-p, p-hi), 0)))
    points = np.asarray(instance.get('points', []))
    surface = float(np.quantile(np.linalg.norm(points[:, [0, 2]]-p, axis=1), .05)) if points.ndim == 2 and len(points) else None
    return dict(estimated_boundary_gap_m=gap, observed_surface_gap_m=surface,
                allowed_gap_m=[MIN_GAP, MAX_GAP], preferred_gap_m=PREFERRED_GAP,
                close_enough=bool(MIN_GAP <= gap <= MAX_GAP and surface is not None and surface <= MAX_GAP+.10))


def boundary_samples(instance, height):
    """Sample offsets from all observed sides and rounded corners of the visual bbox."""
    lo, hi = np.asarray(instance['bbox_min'])[[0, 2]], np.asarray(instance['bbox_max'])[[0, 2]]
    for gap in (PREFERRED_GAP, .4, .6):
        for x in np.linspace(lo[0], hi[0], max(3, min(20, int((hi[0]-lo[0])/.25)+2))):
            for z in (lo[1]-gap, hi[1]+gap):
                yield np.array([x, height, z])
        for z in np.linspace(lo[1], hi[1], max(3, min(20, int((hi[1]-lo[1])/.25)+2))):
            for x in (lo[0]-gap, hi[0]+gap):
                yield np.array([x, height, z])
        for ix in (0, 1):
            for iz in (0, 1):
                corner=np.array([lo[0] if ix==0 else hi[0],lo[1] if iz==0 else hi[1]])
                for angle in (np.pi/6, np.pi/4, np.pi/3):
                    p=corner+gap*np.array([(-1 if ix==0 else 1)*np.cos(angle),(-1 if iz==0 else 1)*np.sin(angle)])
                    yield np.array([p[0],height,p[1]])
