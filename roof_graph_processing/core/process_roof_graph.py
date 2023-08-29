import numpy as np

from roof_graph_processing.core.geometry.common_helper import euclidean_distance

from roof_graph_processing.core.geometry.line_helper import get_foot_of_perpendicular, \
    get_order_vertex_list_based_on_line_segment_formation, \
    angle_wrt_x_axis, distance_between_two_line_segments, distance_between_two_line_segments_using_vertices, \
    get_points_of_intersection, edge_to_vector, get_angle_between_two_directed_edges

from roof_graph_processing.core.geometry.surface_helper import does_vertex_lie_within_line_segment, \
    does_point_lie_within_a_polygon, distance_between_point_and_polygon, surface_area

from roof_graph_processing.core.geometry.graph_helper import find_sub_graphs, extract_all_unique_loops, \
    check_if_a_graph_completely_lies_within_another, \
    subdivide_adjacency_matrix, purify_adjacency_list

from roof_graph_processing.core.roof import Vertex, UndirectedEdge, Edge, Surface, Graph


def remove_intersecting_lines(vertices: dict, undirected_edges: dict):
    modified_vertices = vertices
    modified_undirected_edges = {}

    ue_to_poI_map, poI_to_ue_map = get_points_of_intersection(undirected_edges)

    for ue_id in ue_to_poI_map:
        intersecting_points = ue_to_poI_map[ue_id]  # Potential new vertices

        ue = undirected_edges[ue_id]

        # Populate the vertices of the original edge
        v1, v2 = ue.v1, ue.v2
        if v1.vertex_id not in modified_vertices:
            modified_vertices[v1.vertex_id] = v1

        if v2.vertex_id not in modified_vertices:
            modified_vertices[v2.vertex_id] = v2

        if not intersecting_points:  # Empty list, this edge does not intersect with any other edge
            modified_undirected_edges[ue.edge_id] = ue
            continue

        # Populate the vertices of intersecting_points
        for v in intersecting_points:
            if v.vertex_id not in modified_vertices:
                modified_vertices[v.vertex_id] = v

        poI_vertex_id_list = [v.vertex_id for v in intersecting_points]

        ordered_vertex_id_list = get_order_vertex_list_based_on_line_segment_formation([ue.v1.vertex_id,
                                                                                        ue.v2.vertex_id],
                                                                                       poI_vertex_id_list,
                                                                                       modified_vertices)

        num_ordered_vertices = len(ordered_vertex_id_list)

        # Divide the undirected edge at the points of intersection into smaller undirected edges
        for v_idx in range(num_ordered_vertices - 1):
            v_st_id = ordered_vertex_id_list[v_idx]
            v_st = modified_vertices[v_st_id]

            v_en_id = ordered_vertex_id_list[(v_idx + 1) % num_ordered_vertices]
            v_en = modified_vertices[v_en_id]

            new_ue = UndirectedEdge(v_st, v_en)
            new_ue.meta = ue.meta  # Preserve the edge meta
            modified_undirected_edges[new_ue.edge_id] = new_ue

    return modified_vertices, modified_undirected_edges


def fuse_junctions_and_create_graph(vertices: dict, undirected_edges: dict, epsilon=12):

    completed_vertices = {}

    # Will contain groups of points which are very close to each other, indicating they are actually the same point
    vertex_to_cluster_centers = {}

    cluster_centers = {}

    vertex_ids = list(vertices.keys())
    num_vertices = len(vertex_ids)

    for v1_idx in range(num_vertices):
        v1_uuid = vertex_ids[v1_idx]
        if v1_uuid in completed_vertices:
            continue

        v1 = vertices[v1_uuid]
        centroid_x, centroid_y = v1.x, v1.y
        count = 1

        completed_vertices[v1.vertex_id] = True

        cluster = [v1.vertex_id]

        for v2_idx in range(v1_idx + 1, num_vertices):
            v2_uuid = vertex_ids[v2_idx]
            v2 = vertices[v2_uuid]
            if v2_uuid in completed_vertices:
                continue

            if euclidean_distance(v1, v2) < epsilon:  # Some threshold
                completed_vertices[v2.vertex_id] = True
                centroid_x += v2.x
                centroid_y += v2.y
                count += 1
                cluster.append(v2.vertex_id)

        if count > 1:
            cluster_center = Vertex(centroid_x // count, centroid_y // count)
            cluster_center.meta = v1.meta  # Retaining the meta data

        else:
            cluster_center = v1  # Only one in the cluster, i.e. v1 is mapped to no other point

        cluster_centers[cluster_center.vertex_id] = cluster_center

        for vertex_uuid in cluster:
            vertex_to_cluster_centers[vertex_uuid] = cluster_center.vertex_id

    adjacency_list = {}
    modified_vertices = {}
    undirected_edges_modified = {}

    vertex_keys_list = list(vertices.keys())
    for vertex_idx in range(len(vertex_keys_list)):
        original_vertex_id = vertex_keys_list[vertex_idx]
        cluster_center_vertex_id = vertex_to_cluster_centers[original_vertex_id]
        cluster_center = cluster_centers[cluster_center_vertex_id]

        if cluster_center.vertex_id not in modified_vertices:
            modified_vertices[cluster_center.vertex_id] = cluster_center

    for edge_id in undirected_edges:
        v1, v2 = undirected_edges[edge_id].v1, undirected_edges[edge_id].v2

        mod_v1, mod_v2 = \
            cluster_centers[vertex_to_cluster_centers[v1.vertex_id]], \
            cluster_centers[vertex_to_cluster_centers[v2.vertex_id]]

        if mod_v1.vertex_id not in modified_vertices:
            modified_vertices[mod_v1.vertex_id] = mod_v1

        if mod_v2.vertex_id not in modified_vertices:
            modified_vertices[mod_v2.vertex_id] = mod_v2

        if mod_v1.vertex_id not in adjacency_list:
            adjacency_list[mod_v1.vertex_id] = []

        if mod_v2.vertex_id not in adjacency_list:
            adjacency_list[mod_v2.vertex_id] = []

        adjacency_list[mod_v1.vertex_id].append(mod_v2.vertex_id)
        adjacency_list[mod_v2.vertex_id].append(mod_v1.vertex_id)

        undirected_edge = UndirectedEdge(mod_v1, mod_v2)
        undirected_edge.meta = undirected_edges[edge_id].meta  # Updating meta data
        undirected_edges_modified[undirected_edge.edge_id] = undirected_edge

    vertices = modified_vertices
    undirected_edges = undirected_edges_modified

    return vertices, undirected_edges


def fuse_junctions_with_edges_and_create_graph(vertices, undirected_edges, epsilon=12):

    undirected_edge_to_vertex_map = {}  # map to store the undirected edge to nearby vertex of fusion correlations
    for vertex_id in vertices:
        vertex = vertices[vertex_id]
        for undirected_edge_id in undirected_edges:
            undirected_edge = undirected_edges[undirected_edge_id]
            if vertex_id in (undirected_edge.v1.vertex_id, undirected_edge.v2.vertex_id):
                # If the vertex is a part of the edge itself, no need to proceed
                continue

            foot_of_perp_vertex = get_foot_of_perpendicular(vertex, undirected_edge)

            if not does_vertex_lie_within_line_segment(foot_of_perp_vertex, undirected_edge):
                # If the foot of perpendicular does not lie within the undirected edge, no need to proceed
                continue

            if euclidean_distance(foot_of_perp_vertex, vertex) < epsilon:
                # This means we have found a nearby vertex to corresponding edge
                if undirected_edge.edge_id not in undirected_edge_to_vertex_map:
                    undirected_edge_to_vertex_map[undirected_edge.edge_id] = []

                undirected_edge_to_vertex_map[undirected_edge.edge_id].append(vertex.vertex_id)

    modified_undirected_edges = undirected_edges

    for redundant_edge_id in undirected_edge_to_vertex_map:
        redundant_edge = undirected_edges[redundant_edge_id]
        fusing_vertex_id_list = undirected_edge_to_vertex_map[redundant_edge_id]

        # Existing edge with redundant_edge_id will be divided to form 2 new edges
        # If existing edge had two vertices (v1, v2), the new edges will have vertices as follows:
        # new_edge1 will be formed by (v1, fusing_vertex)
        # new_edge2 will be formed by (fusing_vertex, v2)

        edge_meta_data = modified_undirected_edges[redundant_edge_id].meta  # Need to preserve the meta data
        del modified_undirected_edges[redundant_edge_id]

        ordered_vertex_id_list = get_order_vertex_list_based_on_line_segment_formation([redundant_edge.v1.vertex_id,
                                                                                        redundant_edge.v2.vertex_id],
                                                                                       fusing_vertex_id_list, vertices)
        for v_num in range(0, len(ordered_vertex_id_list) - 1):
            v1 = vertices[ordered_vertex_id_list[v_num]]
            v2 = vertices[ordered_vertex_id_list[v_num + 1]]
            undirected_edge = UndirectedEdge(v1, v2)
            undirected_edge.meta = edge_meta_data  # Adding the same meta data
            modified_undirected_edges[undirected_edge.edge_id] = undirected_edge

    return vertices, modified_undirected_edges


def remove_line_with_lower_probability(edge1: Edge, edge2: Edge, undirected_edges: dict, longEdge_to_shortEdge: dict):
    """
        Removes longer line with lower probability
        Args:
            edge1: Longer edge with lower probability that will be deleted
            edge2: Shorted edge with higher probability
            undirected_edges: map of edge_id to corresponding objects of class Edge
            longEdge_to_shortEdge: Map of longer edges to shorter edges that need to be corrected

    """
    new_edge_id = "-".join(sorted([edge2.end.vertex_id, edge1.end.vertex_id]))
    # Adding new edge to avoid discontinuity
    if new_edge_id not in undirected_edges:
        # assigning default edge score of 1.0 for newly created edges
        new_edge = UndirectedEdge(edge1.end, edge2.end, meta={'pred_score': 1.0})
        undirected_edges[new_edge.edge_id] = new_edge

    edge1_id_1 = edge1.start.vertex_id + "-" + edge1.end.vertex_id
    edge1_id_2 = edge1.end.vertex_id + "-" + edge1.start.vertex_id

    _ = longEdge_to_shortEdge.pop(edge1_id_1, [])
    _ = longEdge_to_shortEdge.pop(edge1_id_2, [])
    for edge_id in [edge1_id_1, edge1_id_2]:
        if edge_id in undirected_edges:
            _ = undirected_edges.pop(edge_id)


def remove_overlapping_lines(vertices: dict, undirected_edges: dict, angle_tolerance_in_degrees=5.0):
    """
    Merges line segments which subtend a very small angle (within the tolerance)
    Modifies vertices in-place
    Args:
        vertices: map of vertex_id to corresponding objects of class Vertex
        undirected_edges: map of edge_id to corresponding objects of class Edge
        angle_tolerance_in_degrees: The threshold of angle between two edges within which
                                    they are considered overlapping

    Returns: vertices, undirected_edges

    """

    # Map having the following structure
    # Key: the edge_id of the longer edge
    # Value: the list of edge_id(s) of the smaller edges
    # The core idea is to combine smaller edges with the longer edges
    # The long edge (key) and all edges in the value, must have the start vertex as common
    mp_longEdge_to_shortEdge = {}

    # Temporary directed edges directory to store the edges (directed edges)
    tmp_directed_edges = {}
    total_edges = list(undirected_edges.keys())
    for ue1_id in total_edges:
        if ue1_id not in undirected_edges:
            continue
        ue1 = undirected_edges[ue1_id]
        score1 = ue1.meta['pred_score']
        for ue2_id in total_edges:
            # Deleting based on score so rechecking if ue1_id is present
            if ue2_id not in undirected_edges or ue1_id not in undirected_edges:
                continue
            ue2 = undirected_edges[ue2_id]
            score2 = ue2.meta['pred_score']
            if ue1.edge_id == ue2.edge_id:
                continue

            common_vertex = None  # Common vertex of either undirected edge
            ue1_end_vertex = None  # End point of one undirected edge
            ue2_end_vertex = None  # End point of the other undirected edge

            # Now check if they share a common vertex
            if ue1.v1.vertex_id == ue2.v1.vertex_id:
                common_vertex = ue1.v1
                ue1_end_vertex = ue1.v2
                ue2_end_vertex = ue2.v2

            elif ue1.v1.vertex_id == ue2.v2.vertex_id:
                common_vertex = ue1.v1
                ue1_end_vertex = ue1.v2
                ue2_end_vertex = ue2.v1

            elif ue1.v2.vertex_id == ue2.v1.vertex_id:
                common_vertex = ue1.v2
                ue1_end_vertex = ue1.v1
                ue2_end_vertex = ue2.v2

            elif ue1.v2.vertex_id == ue2.v2.vertex_id:
                common_vertex = ue1.v2
                ue1_end_vertex = ue1.v1
                ue2_end_vertex = ue2.v1

            if common_vertex is None or ue1_end_vertex is None or ue2_end_vertex is None:
                # The edges do not share a common vertex
                continue

            edge1 = Edge(common_vertex, ue1_end_vertex)
            edge2 = Edge(common_vertex, ue2_end_vertex)

            angle_1_2 = get_angle_between_two_directed_edges(edge1, edge2)

            if angle_1_2 < angle_tolerance_in_degrees:
                # Populate the directed edges
                tmp_directed_edges[edge1.edge_id] = edge1
                tmp_directed_edges[edge2.edge_id] = edge2

                # These 2 edges will be merged
                if edge1.edge_length() > edge2.edge_length():
                    if score2 > score1 and edge2.edge_length() > 0.75 * edge1.edge_length():
                        remove_line_with_lower_probability(edge1, edge2, undirected_edges, mp_longEdge_to_shortEdge)
                        continue

                    if edge1.edge_id not in mp_longEdge_to_shortEdge:
                        mp_longEdge_to_shortEdge[edge1.edge_id] = []

                    if edge2.edge_id not in mp_longEdge_to_shortEdge[edge1.edge_id]:
                        mp_longEdge_to_shortEdge[edge1.edge_id].append(edge2.edge_id)

    # Sort the edge_ids with edges arranged in descending order of edge lengths
    mp_longEdge_to_shortEdge = {k: v for k, v in sorted(mp_longEdge_to_shortEdge.items(),
                                                        key=lambda item: tmp_directed_edges[item[0]].edge_length(),
                                                        reverse=True)}

    # Combine w.r.t a longer edge combine the smaller edges

    done_edge_ids = {edge_id: False for edge_id in list(mp_longEdge_to_shortEdge.keys())}

    # Combined dictionary
    combined_mp_longEdge_to_shortEdge = {}

    # Breadth first search on the following map to combine the edges
    for edge_id in mp_longEdge_to_shortEdge:
        if done_edge_ids[edge_id]:
            continue

        cur = [edge_id]
        done_edge_ids[edge_id] = True
        edges_to_combine = set()
        while cur:
            next_iter = []
            for cur_edge_id in cur:
                edges_to_combine.add(cur_edge_id)
                if cur_edge_id in mp_longEdge_to_shortEdge:
                    next_iter = next_iter + mp_longEdge_to_shortEdge[cur_edge_id]

            cur = list(set(next_iter))

            # Set all the edge ids to done
            for cur_edge_id in cur:
                done_edge_ids[cur_edge_id] = True

            edges_to_combine.union(set(next_iter))

        edges_to_combine = list(edges_to_combine)
        if edge_id in edges_to_combine:
            edges_to_combine.remove(edge_id)

        combined_mp_longEdge_to_shortEdge[edge_id] = edges_to_combine

    edges_parsed = set()
    # Logic to combine: All the shorter edges in the list (value) will become aligned with the longer edge (key)
    for long_edge_id in combined_mp_longEdge_to_shortEdge:
        # If edge is parsed already in reverse then it is a repeat
        if long_edge_id in edges_parsed:
            continue

        # This is the longest edge, and all the edges in the list will combine with this
        long_edge = tmp_directed_edges[long_edge_id]
        long_edge_id_reverse = long_edge.end.vertex_id + "-" + long_edge.start.vertex_id

        # If long edge is already deleted as shorter edge do not continue with the overlapping since there
        # is no edge present
        if long_edge_id not in undirected_edges and long_edge_id_reverse not in undirected_edges:
            continue

        # The following will contain all the vertices that shall lie on the long edge after the combination
        # Post the combination, the long edge shall be divided into smaller edges
        vertices_on_long_edge = []

        # This for loop is since the dict contains both a-b and b-a as keys resulting in issues. So solving both in go
        # to solve this issue
        for long_edge_sub_id in [long_edge_id, long_edge_id_reverse]:
            if long_edge_sub_id not in combined_mp_longEdge_to_shortEdge:
                continue
            long_edge = tmp_directed_edges[long_edge_sub_id]
            long_edge_vector = edge_to_vector(long_edge)
            edges_parsed.add(long_edge_sub_id)
            for combining_edge_id in combined_mp_longEdge_to_shortEdge[long_edge_sub_id]:
                combining_edge = tmp_directed_edges[combining_edge_id]
                combining_edge_length = combining_edge.edge_length()

                # Create a new vertex and hence a new edge
                new_vertex = Vertex(long_edge.start.x + combining_edge_length * long_edge_vector.x,
                                    long_edge.start.y + combining_edge_length * long_edge_vector.y)

                vertices[new_vertex.vertex_id] = new_vertex
                vertices_on_long_edge.append(new_vertex.vertex_id)

                # Delete the old undirected edge
                old_ue = UndirectedEdge(combining_edge.start, combining_edge.end)
                if old_ue.edge_id in undirected_edges:
                    del undirected_edges[old_ue.edge_id]
                del old_ue

                # This can cause (in some cases) a disconnections, hence a new edge needs to be created
                connecting_ue = UndirectedEdge(combining_edge.end, new_vertex)
                undirected_edges[connecting_ue.edge_id] = connecting_ue

        # Sort the vertices that are on the long edge along the line segment
        ordered_vertex_list = get_order_vertex_list_based_on_line_segment_formation([long_edge.start.vertex_id,
                                                                                     long_edge.end.vertex_id],
                                                                                    vertices_on_long_edge,
                                                                                    vertices)

        long_ue = UndirectedEdge(long_edge.start, long_edge.end)
        del undirected_edges[long_ue.edge_id]
        del long_ue

        # Create undirected edges based on the sorted vertices list
        for v_idx in range(len(ordered_vertex_list) - 1):
            v1 = vertices[ordered_vertex_list[v_idx]]
            v2 = vertices[ordered_vertex_list[v_idx + 1]]

            ue = UndirectedEdge(v1, v2)
            undirected_edges[ue.edge_id] = ue

    # Some vertices has become obsolete and some new have been added
    # Will take care of all those scenarios here
    vertices = {}
    for ue_id in undirected_edges:
        ue = undirected_edges[ue_id]
        vertices[ue.v1.vertex_id] = ue.v1
        vertices[ue.v2.vertex_id] = ue.v2

    return vertices, undirected_edges


def identify_disjoint_graphs(adjacency_list: dict, vertices: dict):
    sub_graphs = []

    # The following module removes self directing and repeated edges to speed up computation
    vertices, adjacency_list = purify_adjacency_list(vertices, adjacency_list)

    list_of_sub_graph_vertices, list_of_sub_graph_adjacency_list = subdivide_adjacency_matrix(adjacency_list, vertices)

    for sub_graph_vertices, sub_graph_adjacency_list in zip(list_of_sub_graph_vertices,
                                                            list_of_sub_graph_adjacency_list):

        graph = Graph(sub_graph_adjacency_list, sub_graph_vertices)
        sub_graphs.append(graph)

    return sub_graphs


def get_central_roof_graph(roof_graph_list, img_dims=(800, 800)):

    # Module which selects the central roof graph among all the detected graphs
    central_roof_graph = None  # By default
    min_dist_centroid_img_center = np.inf

    img_center = Vertex(img_dims[0] // 2, img_dims[1] // 2)

    # Check if the central point is enclosed by any graph's outer contour
    # if so return that as the central roof graph

    for roof_graph in roof_graph_list:
        roof_outer_contour = roof_graph.outer_contour

        if does_point_lie_within_a_polygon(img_center, roof_outer_contour):
            return roof_graph

    # This means that the central point is not enclosed by any graph's outer contour
    # In this case, the idea is to search for the nearest contour

    for roof_graph in roof_graph_list:
        roof_outer_contour = roof_graph.outer_contour

        distance_point_to_poly = distance_between_point_and_polygon(img_center, roof_outer_contour)

        if distance_point_to_poly < min_dist_centroid_img_center:
            min_dist_centroid_img_center = distance_point_to_poly
            central_roof_graph = roof_graph

    return central_roof_graph


def find_roof_surfaces(graph: Graph):  # SG
    vertices = graph.vertex_list
    connectivity_matrix = graph.connectivity_matrix
    num_vertices = graph.num_vertices

    theta = []  # Angle of each edge connecting every pair of vertices wrt x-axis.

    for i in range(num_vertices):
        row_theta = []
        for j in range(num_vertices):
            y_diff = vertices[j].y - vertices[i].y
            x_diff = vertices[j].x - vertices[i].x
            row_theta.append(angle_wrt_x_axis(x_diff, y_diff))

        theta.append(row_theta)

    sub_graphs_list = find_sub_graphs(num_vertices, connectivity_matrix)

    for sub_graph in sub_graphs_list:
        new_vertices = []
        for point in sub_graph:
            new_vertices.append(vertices[point])

        new_adj_matrix = []
        new_theta = []

        for i in sub_graph:
            adj_matrix_row = []
            adj_theta_row = []

            for j in sub_graph:
                adj_matrix_row.append(connectivity_matrix[i][j])
                adj_theta_row.append(theta[i][j])

            new_adj_matrix.append(adj_matrix_row)
            new_theta.append(adj_theta_row)

        graph_outer_contour, new_points_index_loops = extract_all_unique_loops(new_adj_matrix, new_vertices, new_theta)

        outer_contour_points = []

        for p_idx in graph_outer_contour:
            outer_contour_points.append(vertices[p_idx])

        if len(outer_contour_points) <= 2:
            continue

        outer_surface = Surface(outer_contour_points)

        graph.outer_contour = outer_surface

        for points_index_loop in new_points_index_loops:
            points_loop = []
            for p_idx in points_index_loop:
                vertex = vertices[p_idx]

                points_loop.append(vertex)

            if len(points_loop) <= 2:
                continue

            surface = Surface(points_loop)
            graph.surface_list.append(surface)


def remove_repeated_edges(graph: Graph):
    edges_parsed_so_far = {}  # Vertex tuple to Edge
    # Starting with the outer contour

    for edge in graph.outer_contour.edge_list:
        edges_parsed_so_far[tuple([edge.start.vertex_id, edge.end.vertex_id])] = edge

    # Will iterate over the surface list and remove repeated edges
    for roof_surface in graph.surface_list:
        for edge_idx in range(len(roof_surface.edge_list)):

            roof_edge = roof_surface.edge_list[edge_idx]
            if tuple([roof_edge.start.vertex_id, roof_edge.end.vertex_id]) in edges_parsed_so_far:
                roof_surface.edge_list[edge_idx] = edges_parsed_so_far[tuple([roof_edge.start.vertex_id,
                                                                              roof_edge.end.vertex_id])]

            else:
                edges_parsed_so_far[tuple([roof_edge.start.vertex_id, roof_edge.end.vertex_id])] = roof_edge


def identify_disjoint_sub_graphs(roof_graph_list):
    # Check if a disjoint graph completely overlaps another disjoint graph
    # If the above situation is satisfied, merge them as one bigger graph
    # This is required as a the smaller disjoint graph may be an internal structure (say a dormer) within the rooftop

    roof_graphs = [roof_graph for roof_graph in roof_graph_list if surface_area(roof_graph.outer_contour) > 0]

    roof_graph_list = roof_graphs

    num_overlaps = -1
    while num_overlaps:
        num_overlaps = 0
        # Merged roof graphs which will be deleted
        merged_roof_graphs = []

        # Those roof graphs created as a result of a merge and will be added to the list of roof graphs
        new_roof_graphs = []

        roof_graph_list.sort(key=lambda x: surface_area(x.outer_contour))

        num_roof_graphs = len(roof_graph_list)
        for i in range(num_roof_graphs):
            roof_graph1 = roof_graph_list[i]
            for j in range(i + 1, num_roof_graphs):
                roof_graph2 = roof_graph_list[j]

                combined_roof_graph = check_if_a_graph_completely_lies_within_another(roof_graph1, roof_graph2)

                if combined_roof_graph is None:
                    continue

                else:
                    num_overlaps += 1

                    merged_roof_graphs.append(roof_graph1)
                    merged_roof_graphs.append(roof_graph2)

                    new_roof_graphs.append(combined_roof_graph)

                    break

            if num_overlaps >= 1:
                break

        #  Delete already merged roof graphs
        for merged_roof_graph in merged_roof_graphs:
            if merged_roof_graph in roof_graph_list:
                roof_graph_list.remove(merged_roof_graph)

        # Add newly formed combined roof graphs as a result from merges
        for new_roof_graph in new_roof_graphs:
            if new_roof_graph not in roof_graph_list:
                roof_graph_list.append(new_roof_graph)

    return roof_graph_list
