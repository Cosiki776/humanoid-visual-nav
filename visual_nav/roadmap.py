"""Geometry-only fallback graph for conservative humanoid corner tracking."""
import heapq
import numpy as np


class GeometricRoadmap:
    def __init__(self, pathfinder, island, direct, floor, spacing=.4):
        self.direct = direct
        lo, hi = pathfinder.get_bounds()
        points = []
        for x in np.arange(lo[0], hi[0], spacing):
            for z in np.arange(lo[2], hi[2], spacing):
                query = np.array([x, floor, z])
                point = np.asarray(pathfinder.snap_point(query, island))
                if np.isfinite(point).all() and np.linalg.norm(point-query) < .12:
                    points.append(point)
        self.points = np.asarray(points).reshape(-1, 3)
        self.edges = {i: [] for i in range(len(points))}
        for i, point in enumerate(self.points):
            distances = np.linalg.norm(self.points-point, axis=1)
            for j in np.flatnonzero((distances > .01) & (distances <= 1.0)):
                if j > i and direct(point, self.points[j]):
                    self.edges[i].append((int(j), float(distances[j])))
                    self.edges[j].append((i, float(distances[j])))

    def path(self, start, end):
        start, end = np.asarray(start), np.asarray(end)
        if self.direct(start, end):
            return [start, end]
        points = np.concatenate((self.points, [start, end]))
        src, dst = len(points)-2, len(points)-1
        edges = {i: list(neighbors) for i, neighbors in self.edges.items()}
        edges[src], edges[dst] = [], []
        for index in (src, dst):
            distances = np.linalg.norm(self.points-points[index], axis=1)
            for other in np.flatnonzero(distances <= 1.5):
                if self.direct(points[index], points[other]):
                    edges[index].append((int(other), float(distances[other])))
                    edges[other].append((index, float(distances[other])))
        distances, previous, queue = {src: 0.}, {}, [(0., src)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node]:
                continue
            if node == dst:
                route = [dst]
                while route[-1] != src:
                    route.append(previous[route[-1]])
                return [points[i] for i in reversed(route)]
            for nxt, length in edges[node]:
                candidate = distance+length
                if candidate < distances.get(nxt, float('inf')):
                    distances[nxt], previous[nxt] = candidate, node
                    heapq.heappush(queue, (candidate, nxt))
        return None
