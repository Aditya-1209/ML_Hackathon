from collections import OrderedDict
import math

import networkx as nx


class Router:
    def __init__(self,nodes,arcs,metadata):
        self.arcs = {row.arc_id: row for row in arcs.itertuples()}
        self.directed = metadata["graph_directed"]
        self.graph = nx.DiGraph() if self.directed else nx.Graph()
        self.graph.add_nodes_from(nodes.node_id)
        for row in arcs.itertuples():
            current = self.graph.get_edge_data(row.from_node_id,row.to_node_id)
            if current is None or row.length_m < current["weight"]:
                self.graph.add_edge(row.from_node_id,row.to_node_id,weight=row.length_m,arc_id=row.arc_id)
        self.cache = OrderedDict()

    def between(self,left,right,limit):
        a,b = self.arcs[left.candidate_arc_id],self.arcs[right.candidate_arc_id]
        best = (math.inf,[])
        if left.candidate_arc_id == right.candidate_arc_id:
            delta = (right.projection_fraction-left.projection_fraction)*a.length_m
            if delta >= -1e-8 or not self.directed:
                return (abs(delta),[int(left.candidate_arc_id)]) if abs(delta) <= limit else (math.inf,[])
        exits = [(int(a.to_node_id),(1-left.projection_fraction)*a.length_m)]
        entries = [(int(b.from_node_id),right.projection_fraction*b.length_m)]
        if not self.directed:
            exits.append((int(a.from_node_id),left.projection_fraction*a.length_m))
            entries.append((int(b.to_node_id),(1-right.projection_fraction)*b.length_m))
        cap = max(1000,int(math.ceil(limit/1000))*1000)
        for source,offset_a in exits:
            key = (source,cap)
            if key not in self.cache:
                self.cache[key] = nx.single_source_dijkstra(self.graph,source,cutoff=cap,weight="weight")
                if len(self.cache)>64:
                    self.cache.popitem(last=False)
            distances,paths = self.cache[key]
            for target,offset_b in entries:
                distance = offset_a + distances.get(target,math.inf) + offset_b
                if distance < best[0] and distance <= limit:
                    path = paths[target]
                    arc_ids = [int(left.candidate_arc_id)]
                    arc_ids += [int(self.graph[u][v]["arc_id"]) for u,v in zip(path,path[1:])]
                    arc_ids.append(int(right.candidate_arc_id))
                    best = (distance,[arc for i,arc in enumerate(arc_ids) if i==0 or arc!=arc_ids[i-1]])
        return best if best[0] <= limit else (math.inf,[])
