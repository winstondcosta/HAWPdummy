import itertools

from roof_graph_processing.core.roof import UndirectedEdge, Graph

from roof_graph_processing.core.geometry.line_helper import check_if_edges_intersect, \
    get_angle_between_two_line_segments


def edge_intersect_with_edge_list(ue: UndirectedEdge, ue_dict: dict):
    for ue_id in ue_dict:
        ue_ = ue_dict[ue_id]
        if check_if_edges_intersect(ue_, ue):
            return True
    return False


def get_min_diff_among_angles_list(angle, angles_list):
    min_diff = 1e9
    for angle_ in angles_list:
        if abs(angle - angle_) < min_diff:
            min_diff = abs(angle - angle_)
    return min_diff


def get_undirected_edges_from_graph(graph: Graph, edge_meta_list: dict):
    roof_surface_undirected_edges = {}

    for roof_surface in graph.surface_list:
        for p_idx in range(len(roof_surface.ordered_vertices)):
            start_vertex = roof_surface.ordered_vertices[p_idx]
            end_vertex = roof_surface.ordered_vertices[(p_idx + 1) % len(roof_surface.ordered_vertices)]

            edge_tuple = tuple(sorted([start_vertex.vertex_id, end_vertex.vertex_id]))

            if edge_tuple not in edge_meta_list:
                edge_meta = {"pred_score": 1.0}  # Default value

            else:
                edge_meta = edge_meta_list[edge_tuple]  # Retrieve the meta

            ue = UndirectedEdge(start_vertex,
                                end_vertex,
                                meta=edge_meta)

            roof_surface_undirected_edges[ue.edge_id] = ue

    # Will return this
    dangling_undirected_edges = graph.dangling_undirected_edges

    for dangling_ue_id in dangling_undirected_edges:
        dangling_ue = dangling_undirected_edges[dangling_ue_id]
        v1, v2 = dangling_ue.v1, dangling_ue.v2

        edge_tuple = tuple(sorted([v1.vertex_id, v2.vertex_id]))

        if edge_tuple not in edge_meta_list:
            edge_meta = {"pred_score": 1.0}  # Default

        else:
            edge_meta = edge_meta_list[edge_tuple]  # Retrieve the meta

        dangling_undirected_edges[dangling_ue_id].meta = edge_meta

    graph.roof_surface_undirected_edges = roof_surface_undirected_edges  # With updated meta
    graph.dangling_undirected_edges = dangling_undirected_edges  # With updated meta


def get_undirected_edges_from_sub_graphs(sub_graphs, edge_meta_list, include_dangling_edges=False):
    undirected_edges = {}

    for graph in sub_graphs:
        get_undirected_edges_from_graph(graph, edge_meta_list)

        undirected_edges.update(graph.roof_surface_undirected_edges)

        if include_dangling_edges:
            undirected_edges.update(graph.dangling_undirected_edges)

    return undirected_edges


def get_edge_angles_in_graph(vertices: dict, adjacency_list: dict):
    angles = []

    for v1_id in adjacency_list:
        v1 = vertices[v1_id]

        adj_vertices = adjacency_list[v1_id]
        if len(adj_vertices) < 2:
            continue

        for v2_id, v3_id in itertools.combinations(adj_vertices, 2):

            v2 = vertices[v2_id]
            v3 = vertices[v3_id]

            ue12 = UndirectedEdge(v1, v2)
            ue13 = UndirectedEdge(v1, v3)

            angles.append(get_angle_between_two_line_segments(ue12, ue13))

            del ue12
            del ue13

    return angles
