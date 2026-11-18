"""Road network graph construction, position-aware routing, and shortest-path caching.

Follows PathSnap ML v1 specification:
- Directed and undirected graph support (default directed).
- Preserves all original node IDs and arc IDs.
- Calculates distance between projected positions on road segments, not just endpoint nodes.
- Handles parallel edges, zero-length edges, disconnected components, and same-arc routing.
- Reconstructs intermediate arc sequences.
- Thread-safe / deterministic shortest path caching with trip/network awareness.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArcRecord:
    """Immutable representation of a road segment arc."""
    arc_id: int
    from_node_id: int
    to_node_id: int
    length_m: float
    from_x: float
    from_y: float
    to_x: float
    to_y: float


@dataclass(frozen=True)
class CandidatePosition:
    """Projected candidate position on an arc."""
    arc_id: int
    fraction: float  # Projection parameter along arc in [0.0, 1.0]
    projected_x: float
    projected_y: float
    distance_to_arc_m: float = 0.0
    obs_id: int | None = None
    p_match: float = 1.0


class RoadNetworkGraph:
    """Road network graph with position-aware shortest path routing and intermediate path reconstruction."""

    def __init__(
        self,
        nodes: pd.DataFrame | dict[str, Any],
        arcs: pd.DataFrame | dict[str, Any],
        directed: bool = True,
        network_id: str | int = "default",
    ):
        """Construct the road graph from nodes and arcs tables.

        Args:
            nodes: DataFrame or dict with node coordinates (supports columns 'x', 'y' in metres,
                   or 'lon', 'lat', or row index as node_id).
            arcs: DataFrame or dict with 'from_node_id', 'to_node_id', and optional 'length_m'.
                  Row index or 'arc_id' serves as arc_id.
            directed: If True, travel is allowed only from from_node to to_node.
            network_id: Identifier used to partition cache entries.
        """
        self.directed = directed
        self.network_id = str(network_id)
        self.nodes: dict[int, tuple[float, float]] = {}
        self.arcs: dict[int, ArcRecord] = {}
        # Adjacency list: node_id -> list of (neighbor_node_id, weight_m, arc_id)
        self.adj: dict[int, list[tuple[int, float, int]]] = {}
        # Node-to-node shortest path cache: (from_node, to_node) -> (distance, [arc_ids])
        self._node_path_cache: dict[tuple[int, int], tuple[float, list[int]]] = {}
        # Position-aware route cache: (from_arc, round(f1, 4), to_arc, round(f2, 4)) -> (distance, [arc_ids])
        self._position_path_cache: dict[tuple[int, float, int, float], tuple[float, list[int]]] = {}

        self._load_nodes(nodes)
        self._load_arcs(arcs)

    def _load_nodes(self, nodes_data: pd.DataFrame | dict[str, Any]) -> None:
        """Parse and store node coordinates."""
        if not isinstance(nodes_data, pd.DataFrame):
            nodes_df = pd.DataFrame(nodes_data)
        else:
            nodes_df = nodes_data

        # Determine coordinate column names
        x_col = "x" if "x" in nodes_df.columns else ("lon" if "lon" in nodes_df.columns else nodes_df.columns[0])
        y_col = "y" if "y" in nodes_df.columns else ("lat" if "lat" in nodes_df.columns else nodes_df.columns[1])
        id_col = "node_id" if "node_id" in nodes_df.columns else None

        for idx, row in nodes_df.iterrows():
            nid = int(row[id_col]) if id_col else int(idx)
            x_val = float(row[x_col])
            y_val = float(row[y_col])
            self.nodes[nid] = (x_val, y_val)
            self.adj[nid] = []

    def _load_arcs(self, arcs_data: pd.DataFrame | dict[str, Any]) -> None:
        """Parse and store arcs with geometric lengths."""
        if not isinstance(arcs_data, pd.DataFrame):
            arcs_df = pd.DataFrame(arcs_data)
        else:
            arcs_df = arcs_data

        from_col = "from_node_id" if "from_node_id" in arcs_df.columns else arcs_df.columns[0]
        to_col = "to_node_id" if "to_node_id" in arcs_df.columns else arcs_df.columns[1]
        id_col = "arc_id" if "arc_id" in arcs_df.columns else None
        len_col = "length_m" if "length_m" in arcs_df.columns else ("length" if "length" in arcs_df.columns else None)

        for idx, row in arcs_df.iterrows():
            aid = int(row[id_col]) if id_col else int(idx)
            u = int(row[from_col])
            v = int(row[to_col])

            # Ensure nodes exist in adj
            if u not in self.nodes:
                self.nodes[u] = (0.0, 0.0)
                self.adj[u] = []
            if v not in self.nodes:
                self.nodes[v] = (0.0, 0.0)
                self.adj[v] = []

            u_x, u_y = self.nodes[u]
            v_x, v_y = self.nodes[v]

            # Compute Euclidean length in projected coordinates if not provided
            if len_col and not pd.isna(row[len_col]):
                length_m = max(0.0, float(row[len_col]))
            else:
                length_m = math.hypot(v_x - u_x, v_y - u_y)

            arc_rec = ArcRecord(
                arc_id=aid,
                from_node_id=u,
                to_node_id=v,
                length_m=length_m,
                from_x=u_x,
                from_y=u_y,
                to_x=v_x,
                to_y=v_y,
            )
            self.arcs[aid] = arc_rec

            # Add to adjacency list (supports parallel edges naturally)
            self.adj[u].append((v, length_m, aid))
            if not self.directed:
                self.adj[v].append((u, length_m, aid))

    def project_point_to_arc(self, px: float, py: float, arc_id: int) -> CandidatePosition:
        """Project a point (px, py) onto the straight line segment of arc_id.

        Calculates the clamped projection fraction f in [0.0, 1.0] and perpendicular distance.
        Handles zero-length arcs safely without division by zero.
        """
        if arc_id not in self.arcs:
            raise KeyError(f"Arc ID {arc_id} not found in road network.")

        arc = self.arcs[arc_id]
        dx = arc.to_x - arc.from_x
        dy = arc.to_y - arc.from_y
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq <= 1e-12:
            # Zero-length arc: projection is the start node
            fraction = 0.0
            proj_x = arc.from_x
            proj_y = arc.from_y
            dist_m = math.hypot(px - proj_x, py - proj_y)
        else:
            # Scalar projection fraction
            t = ((px - arc.from_x) * dx + (py - arc.from_y) * dy) / seg_len_sq
            fraction = min(max(t, 0.0), 1.0)
            proj_x = arc.from_x + fraction * dx
            proj_y = arc.from_y + fraction * dy
            dist_m = math.hypot(px - proj_x, py - proj_y)

        return CandidatePosition(
            arc_id=arc_id,
            fraction=fraction,
            projected_x=proj_x,
            projected_y=proj_y,
            distance_to_arc_m=dist_m,
        )

    def shortest_node_path(self, start_node: int, target_node: int) -> tuple[float, list[int]]:
        """Compute shortest path distance and sequence of arc IDs between two nodes using Dijkstra's algorithm.

        Returns:
            (distance_m, [arc_id_1, arc_id_2, ...])
            If target_node is unreachable from start_node, returns (float('inf'), []).
        """
        if start_node == target_node:
            return 0.0, []

        cache_key = (start_node, target_node)
        if cache_key in self._node_path_cache:
            return self._node_path_cache[cache_key]

        if start_node not in self.adj or target_node not in self.adj:
            return float("inf"), []

        # Min-heap: (current_dist, current_node)
        heap: list[tuple[float, int]] = [(0.0, start_node)]
        distances: dict[int, float] = {start_node: 0.0}
        # Predecessor mapping: node -> (prev_node, arc_id)
        predecessors: dict[int, tuple[int, int]] = {}

        found = False
        while heap:
            curr_dist, u = heapq.heappop(heap)

            if curr_dist > distances.get(u, float("inf")):
                continue

            if u == target_node:
                found = True
                break

            for v, weight, aid in self.adj.get(u, []):
                new_dist = curr_dist + weight
                if new_dist < distances.get(v, float("inf")):
                    distances[v] = new_dist
                    predecessors[v] = (u, aid)
                    heapq.heappush(heap, (new_dist, v))

        if not found:
            res = (float("inf"), [])
            self._node_path_cache[cache_key] = res
            return res

        # Backtrack path of arcs
        curr = target_node
        path_arcs: list[int] = []
        while curr != start_node:
            prev_node, aid = predecessors[curr]
            path_arcs.append(aid)
            curr = prev_node
        path_arcs.reverse()

        res = (distances[target_node], path_arcs)
        self._node_path_cache[cache_key] = res
        return res

    def network_distance(
        self,
        from_candidate: CandidatePosition,
        to_candidate: CandidatePosition,
    ) -> tuple[float, list[int]]:
        """Calculate the position-aware road-network shortest path distance and intermediate arc sequence.

        CRITICAL REQUIREMENT:
        Calculates distance between projected positions on road segments, not just endpoint nodes.
        Preserves directionality, parallel edges, zero-length edges, same-arc routing, and repeated road visits.

        Args:
            from_candidate: Candidate position at time t.
            to_candidate: Candidate position at time t+1.

        Returns:
            (distance_m, intermediate_arc_sequence)
            If unreachable or physically impossible, returns (float('inf'), []).
        """
        c1 = from_candidate
        c2 = to_candidate

        # Cache key rounded to 4 decimals (~0.1mm precision)
        cache_key = (c1.arc_id, round(c1.fraction, 4), c2.arc_id, round(c2.fraction, 4))
        if cache_key in self._position_path_cache:
            return self._position_path_cache[cache_key]

        if c1.arc_id not in self.arcs or c2.arc_id not in self.arcs:
            res = (float("inf"), [])
            self._position_path_cache[cache_key] = res
            return res

        arc1 = self.arcs[c1.arc_id]
        arc2 = self.arcs[c2.arc_id]

        # Case 1: Same arc routing
        if c1.arc_id == c2.arc_id:
            if c2.fraction >= c1.fraction:
                # Forward motion along the same directed arc
                dist = (c2.fraction - c1.fraction) * arc1.length_m
                res = (dist, [c1.arc_id])
                self._position_path_cache[cache_key] = res
                return res
            else:
                # Backward motion along the same directed arc
                if not self.directed:
                    # In undirected mode, vehicle can move in either direction
                    dist = (c1.fraction - c2.fraction) * arc1.length_m
                    res = (dist, [c1.arc_id])
                    self._position_path_cache[cache_key] = res
                    return res
                else:
                    # In directed mode, vehicle must travel to end of arc, loop through network back to start, and reach fraction
                    tail_dist = (1.0 - c1.fraction) * arc1.length_m
                    cycle_dist, cycle_arcs = self.shortest_node_path(arc1.to_node_id, arc1.from_node_id)
                    head_dist = c2.fraction * arc2.length_m

                    if math.isinf(cycle_dist):
                        res = (float("inf"), [])
                    else:
                        total_dist = tail_dist + cycle_dist + head_dist
                        # Intermediate route: [arc1] + cycle_arcs + [arc1]
                        route = [arc1.arc_id] + cycle_arcs + [arc1.arc_id]
                        res = (total_dist, route)
                    self._position_path_cache[cache_key] = res
                    return res

        # Case 2: Different arcs
        if self.directed:
            # Must exit arc1 at arc1.to_node_id, traverse network to arc2.from_node_id, enter arc2 up to c2.fraction
            tail_dist = (1.0 - c1.fraction) * arc1.length_m
            head_dist = c2.fraction * arc2.length_m

            if arc1.to_node_id == arc2.from_node_id:
                # Directly contiguous segments
                total_dist = tail_dist + head_dist
                res = (total_dist, [arc1.arc_id, arc2.arc_id])
            else:
                node_dist, intermediate_arcs = self.shortest_node_path(arc1.to_node_id, arc2.from_node_id)
                if math.isinf(node_dist):
                    res = (float("inf"), [])
                else:
                    total_dist = tail_dist + node_dist + head_dist
                    route = [arc1.arc_id] + intermediate_arcs + [arc2.arc_id]
                    res = (total_dist, route)

            self._position_path_cache[cache_key] = res
            return res
        else:
            # Undirected mode: evaluate all 4 endpoint combinations
            candidates_endpoints = [
                # from end of arc1 to start of arc2
                (1.0 - c1.fraction, arc1.to_node_id, c2.fraction, arc2.from_node_id),
                # from end of arc1 to end of arc2
                (1.0 - c1.fraction, arc1.to_node_id, 1.0 - c2.fraction, arc2.to_node_id),
                # from start of arc1 to start of arc2
                (c1.fraction, arc1.from_node_id, c2.fraction, arc2.from_node_id),
                # from start of arc1 to end of arc2
                (c1.fraction, arc1.from_node_id, 1.0 - c2.fraction, arc2.to_node_id),
            ]
            best_dist = float("inf")
            best_route: list[int] = []

            for tail_frac, u_node, head_frac, v_node in candidates_endpoints:
                t_dist = tail_frac * arc1.length_m
                h_dist = head_frac * arc2.length_m
                if u_node == v_node:
                    d = t_dist + h_dist
                    arcs_seq = [arc1.arc_id, arc2.arc_id]
                else:
                    n_dist, i_arcs = self.shortest_node_path(u_node, v_node)
                    d = t_dist + n_dist + h_dist
                    arcs_seq = [arc1.arc_id] + i_arcs + [arc2.arc_id]

                if d < best_dist:
                    best_dist = d
                    best_route = arcs_seq

            res = (best_dist, best_route)
            self._position_path_cache[cache_key] = res
            return res

    def clear_cache(self) -> None:
        """Clear shortest path memoization caches."""
        self._node_path_cache.clear()
        self._position_path_cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Return cache sizes for monitoring and testing."""
        return {
            "node_cache_size": len(self._node_path_cache),
            "position_cache_size": len(self._position_path_cache),
        }
