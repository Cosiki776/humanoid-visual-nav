"""Finite geometric search coverage; never infer semantic absence from one frame."""
import math
import numpy as np


class CoverageSearch:
    def __init__(self, points, direct, radius=1.2, route_distance=None):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(self.points):
            raise ValueError('No geometric search samples')
        self.direct, self.radius = direct, radius
        self.covered, self.blocked = set(), set()
        self.failures = {}
        self.scans = []
        self.partial = []
        self.active = None
        self.repeat_visits = {}
        self.route_distance = route_distance
        self.neighbors = {}

    def next_view(self, position):
        """Skip sweeps when all nearby reachable samples were already checked."""
        position = np.asarray(position)
        pending = set(range(len(self.points)))-self.covered-self.blocked
        if not any(np.linalg.norm(self.points[i]-position) <= self.radius and
                   self.direct(position, self.points[i]) for i in pending):
            return None
        scan = next((s for s in self.partial if np.linalg.norm(position-s['position']) < .25), None)
        observed = scan['directions'] if scan else {}
        return next((direction*math.pi/2 for direction in range(4) if direction not in observed), None)

    def record_view(self, position, yaw, observation_id):
        position = np.asarray(position)
        # A full sweep must be taken at the same physical location, not accumulated along a path.
        scan = next((s for s in self.partial if np.linalg.norm(position-s['position']) < .25), None)
        if scan is None:
            scan = dict(position=position.copy(), directions={})
            self.partial.append(scan)
        direction = int(round(yaw/(math.pi/2))) % 4
        scan['directions'][direction] = observation_id
        if len(scan['directions']) != 4:
            return
        evidence = dict(position=scan['position'].tolist(), observations=list(scan['directions'].values()))
        # Do not mark a nearby sample across a wall as searched.
        for index, point in enumerate(self.points):
            if np.linalg.norm(point-position) <= self.radius and self.direct(position, point):
                self.covered.add(index)
                self.blocked.discard(index)
        self.scans.append(evidence)
        if self.active is not None and self.active not in self.covered:
            cell = tuple(np.round(position/.25).astype(int))
            key = (self.active, cell)
            self.repeat_visits[key] = self.repeat_visits.get(key, 0)+1
            if self.repeat_visits[key] >= 3:
                # A controller/search cycle is unresolved, never evidence of target absence.
                self.blocked.add(self.active)
        self.partial = [item for item in self.partial if item is not scan]

    def next_goal(self, position):
        pending = set(range(len(self.points)))-self.covered-self.blocked
        if not pending:
            return None
        # Keep a goal across intermediate keyframes; nearest-goal switching can oscillate across walls.
        if self.active not in pending:
            options = []
            for i in sorted(pending):
                distance = self.route_distance(self.points[i]) if self.route_distance else float(np.linalg.norm(self.points[i]-position))
                if distance is None or not np.isfinite(distance):
                    self.blocked.add(i)
                    continue
                if i not in self.neighbors:
                    self.neighbors[i] = {j for j, point in enumerate(self.points)
                        if np.linalg.norm(point-self.points[i]) <= self.radius and self.direct(self.points[i], point)}
                gain = len(self.neighbors[i] & pending)
                options.append((-gain/(1+distance), distance, i))
            if not options:
                return None
            self.active = min(options)[2]
        return self.points[self.active].copy()

    def failed(self, goal):
        index = int(np.argmin(np.linalg.norm(self.points-np.asarray(goal), axis=1)))
        self.failures[index] = self.failures.get(index, 0)+1
        if self.failures[index] >= 2:
            self.blocked.add(index)

    def summary(self):
        pending = len(set(range(len(self.points)))-self.covered-self.blocked)
        return dict(total_samples=len(self.points), covered_samples=len(self.covered),
                    blocked_samples=len(self.blocked), pending_samples=pending,
                    completed_scans=len(self.scans), complete=pending == 0 and not self.blocked,
                    active_goal=self.points[self.active].tolist() if self.active is not None else None,
                    selection='uncovered geometric samples per route metre; not semantic absence',
                    exhausted=pending == 0, radius_m=self.radius,
                    meaning='finite reachable geometric viewpoint coverage, not proof of semantic absence')
