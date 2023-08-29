import math

from ...core.roof import Graph

from .common_helper import first_index_of_satisfied_predicate, negation_predicate, mod, find_max_index, \
    find_min_index, set_intersection, flatten_list, are_sets_equal, splice

from .surface_helper import polygon_area, does_point_lie_within_a_polygon


def create_adjacency_list_from_undirected_edges(undirected_edges, preserve_meta=False):
    adjacency_list = {}
    meta_dict = {}

    for edge_id in undirected_edges:
        undirected_edge = undirected_edges[edge_id]
        v1 = undirected_edge.v1
        v2 = undirected_edge.v2

        if v1.vertex_id not in adjacency_list:
            adjacency_list[v1.vertex_id] = []

        if v2.vertex_id not in adjacency_list:
            adjacency_list[v2.vertex_id] = []

        adjacency_list[v1.vertex_id].append(v2.vertex_id)
        adjacency_list[v2.vertex_id].append(v1.vertex_id)

        edge_tuple = tuple(sorted([v1.vertex_id, v2.vertex_id]))

        meta_dict[edge_tuple] = undirected_edge.meta

    if preserve_meta:
        return adjacency_list, meta_dict

    else:
        return adjacency_list


def purify_adjacency_list(vertices: dict, adjacency_list: dict):
    new_vertices = {}
    new_adjacency_list = {}

    # check for self directing edges
    for vertex_id in adjacency_list:
        adj_vertices = []
        for adj_vertex_id in adjacency_list[vertex_id]:
            if adj_vertex_id == vertex_id:
                # Self directing edge
                continue

            if adj_vertex_id in adj_vertices:
                # Repeated edge
                continue

            adj_vertices.append(adj_vertex_id)

        if len(adj_vertices) > 0:
            new_adjacency_list[vertex_id] = adj_vertices
            new_vertices[vertex_id] = vertices[vertex_id]

    return new_vertices, new_adjacency_list


def subdivide_adjacency_matrix(adjacency_list: dict, vertices: dict):
    vertex_clusters = []

    traversed = {vertex_id: False for vertex_id in list(adjacency_list.keys())}

    for vertex_id in traversed:
        if traversed[vertex_id]:
            continue

        temp_traversed = {vertex_id: False for vertex_id in list(adjacency_list.keys())}

        # bfs traversal
        next_layer = [vertex_id]
        while len(next_layer) > 0:
            temp_next_layer = []

            for next_node in next_layer:
                temp_traversed[next_node] = True
                for adjacent_node in adjacency_list[next_node]:
                    if not temp_traversed[adjacent_node]:
                        temp_next_layer.append(adjacent_node)

            next_layer = temp_next_layer

        # List of nodes which form a connected graph
        sub_graph_vertices = [vertex_id for vertex_id in temp_traversed if temp_traversed[vertex_id]]

        vertex_clusters.append(sub_graph_vertices)

        # Update traversed
        for vertex_idx in temp_traversed:
            traversed[vertex_idx] = temp_traversed[vertex_idx] or traversed[vertex_idx]

    list_of_sub_graph_adjacency_list, list_of_sub_graph_vertices = [], []

    # Divide adjacency list into sub graphs
    for vertex_cluster in vertex_clusters:
        sub_graph_adjacency_list = {}
        sub_graph_vertices = {}
        for vertex_id in vertex_cluster:
            sub_graph_adjacency_list[vertex_id] = adjacency_list[vertex_id]

        for vertex_id in sub_graph_adjacency_list:
            sub_graph_vertices[vertex_id] = vertices[vertex_id]

        list_of_sub_graph_vertices.append(sub_graph_vertices)
        list_of_sub_graph_adjacency_list.append(sub_graph_adjacency_list)

    return list_of_sub_graph_vertices, list_of_sub_graph_adjacency_list


def find_sub_graphs(num_vertices, connectivity_matrix):  # SG

    included_points = [False for _ in range(num_vertices)]
    sub_graph_list = []

    while sum(included_points) < len(included_points):
        sub_graph = []
        next_not_included_point_index = first_index_of_satisfied_predicate(included_points,
                                                                           negation_predicate)

        list_of_points = [next_not_included_point_index]

        missing_points_to_add = True

        while missing_points_to_add:
            new_points_added = False

            for index in list_of_points:
                sub_graph.append(index)
                included_points[index] = True

            new_list_of_points = []

            for i in list_of_points:
                for j in range(num_vertices):
                    if connectivity_matrix[i][j] != 0 and not included_points[j]:
                        new_list_of_points.append(j)
                        included_points[j] = True
                        new_points_added = True

            if not new_points_added:
                missing_points_to_add = False

            list_of_points = new_list_of_points

        sub_graph_list.append(sorted(sub_graph))

    return sub_graph_list


def trace_surface(loop, adj_matrix, theta):  # SG
    surface_loop = loop.copy()
    num_points = len(adj_matrix)

    closed = False

    while not closed:
        vertices = []
        angles = []

        for k in range(num_points):
            if k == surface_loop[len(surface_loop) - 2] or adj_matrix[surface_loop[len(surface_loop) - 1]][k] == 0:
                continue

            else:
                angles.append(mod(
                    theta[surface_loop[len(surface_loop) - 1]][k]
                    - theta[surface_loop[len(surface_loop) - 1]][surface_loop[len(surface_loop) - 2]],
                    2 * math.pi
                ))

                vertices.append(k)

        angle_index = find_min_index(angles)

        if vertices[angle_index] in surface_loop:
            closed = True

            if vertices[angle_index] != surface_loop[0]:
                loop_new_first_point = surface_loop.index(vertices[angle_index])
                # Sliced at the new first point so that array starts with that point
                surface_loop = surface_loop[loop_new_first_point:]

        else:
            surface_loop.append(vertices[angle_index])

    return surface_loop


def extract_sub_groups_of_loops(all_loops):  # SG
    sub_loops = []
    loop_included = [False for _ in range(len(all_loops))]

    while sum(loop_included) < len(loop_included):
        tmp_sub_group = []
        next_not_included_loop = first_index_of_satisfied_predicate(
            loop_included,
            negation_predicate
        )

        lst_loops = [all_loops[next_not_included_loop]]
        lst_loop_indexes = [next_not_included_loop]
        missing_loops_to_group = True

        while missing_loops_to_group:
            new_loop_grouped = False

            for i in range(len(lst_loops)):
                lst = lst_loops[i]
                tmp_sub_group.append(lst)
                loop_included[lst_loop_indexes[i]] = True

            new_lst_loops = []
            new_lst_loop_indexes = []

            for i in range(len(lst_loops)):
                list1 = lst_loops[i]
                for j in range(len(all_loops)):
                    list2 = all_loops[j]

                    if not loop_included[j] and len(set_intersection(set(list1), set(list2))) >= 2:
                        new_lst_loops.append(list2)
                        new_lst_loop_indexes.append(j)
                        loop_included[j] = True
                        new_loop_grouped = True

            if not new_loop_grouped:
                missing_loops_to_group = False

            lst_loops = new_lst_loops
            lst_loop_indexes = new_lst_loop_indexes

        sub_loops.append(tmp_sub_group)

    point_groups = []
    for i in range(len(sub_loops)):
        point_group = set(flatten_list(sub_loops[i]))
        point_groups.append(point_group)

    hinge_points = []
    point_groups_array = point_groups.copy()

    nb_point_groups = len(point_groups)

    for i in range(nb_point_groups - 1):
        for j in range(i + 1, nb_point_groups):
            group_intersection = set_intersection(point_groups_array[i], point_groups_array[j])
            if len(group_intersection):
                hinge_point = list(group_intersection)[0]
                if hinge_point not in hinge_points:
                    hinge_points.append(hinge_point)

    hinge_point_count = []

    for i in range(nb_point_groups):
        pg = point_groups[i]
        set_i = set_intersection(pg, set(hinge_points))

        hinge_point_count.append(len(set_i))

    return sub_loops, hinge_point_count


def remove_repeated_loops(all_loops, points):  # SG
    sub_loops, hinge_point_count = extract_sub_groups_of_loops(all_loops)
    all_unique_loops_sgs = []

    for lsg in sub_loops:
        temp_sg_unique_loops = []
        for l1 in lsg:
            l1_included = False
            for l2 in temp_sg_unique_loops:
                if are_sets_equal(set(l1), set(l2)):
                    l1_included = True

            if not l1_included:
                temp_sg_unique_loops.append(l1)

        all_unique_loops_sgs.append(temp_sg_unique_loops)

    all_unique_loops = []

    for i in range(len(all_unique_loops_sgs)):
        lsg = all_unique_loops_sgs[i]
        if len(lsg) == 1 or hinge_point_count[i] >= 2:
            all_unique_loops.extend(lsg)

        else:
            areas = []
            for loop in lsg:
                points_in_list = []
                for j in loop:
                    points_in_list.append(points[j])

                x_list = []
                y_list = []

                for point in points_in_list:
                    x_list.append(point.x)
                    y_list.append(point.y)

                areas.append(polygon_area(x_list, y_list))

            envelop_index = find_max_index(areas)

            # Need to implement splice or use some python equivalent to splice of ts/js
            lsg = splice(lsg, envelop_index, 1)

            all_unique_loops.extend(lsg)

    return all_unique_loops


def extract_all_unique_loops(adj_matrix, points, theta):  # SG

    max_roof_surface_area = 0
    graph_outer_contour = []

    number_of_points = len(points)
    all_loops = []

    for i in range(number_of_points):
        for j in range(number_of_points):
            if adj_matrix[i][j] == 0:
                continue

            loop = [i, j]
            extracted_surface = trace_surface(loop, adj_matrix, theta)

            x_list = []
            y_list = []

            for k in extracted_surface:
                x_list.append(points[k].x)
                y_list.append(points[k].y)

            roof_surface_area = polygon_area(x_list, y_list)
            if roof_surface_area > max_roof_surface_area:
                max_roof_surface_area = roof_surface_area
                graph_outer_contour = extracted_surface

            all_loops.append(extracted_surface)

    return graph_outer_contour, remove_repeated_loops(all_loops, points)


def merge_roof_graphs(outer_graph: Graph, inner_graph: Graph):
    #  If outer graph completely engulfs inner graph
    merged_graph_adjacency_list = {**outer_graph.adjacency_list.copy(), **inner_graph.adjacency_list.copy()}
    merged_graph_vertices = {**outer_graph.vertices.copy(), **inner_graph.vertices.copy()}

    merged_graph = Graph(merged_graph_adjacency_list, merged_graph_vertices)

    if len(inner_graph.surface_list) > 0:
        merged_graph_surface_list = outer_graph.surface_list + inner_graph.surface_list

    else:
        merged_graph_surface_list = outer_graph.surface_list + [inner_graph.outer_contour]

    merged_graph.surface_list = merged_graph_surface_list

    merged_graph.outer_contour = outer_graph.outer_contour

    return merged_graph


def check_if_a_graph_completely_lies_within_another(graph1: Graph, graph2: Graph):
    # To check if either of graph1/graph2 completely lies within the other

    # Will only check w.r.t the outer contours
    graph1_outer_contour = graph1.outer_contour
    graph2_outer_contour = graph2.outer_contour

    graph1_lies_within_graph2 = graph2_outer_contour.contains(graph1_outer_contour)

    if graph1_lies_within_graph2:
        return merge_roof_graphs(graph2, graph1)

    graph2_lies_within_graph1 = graph1_outer_contour.contains(graph2_outer_contour)

    if graph2_lies_within_graph1:
        return merge_roof_graphs(graph1, graph2)

    return None
