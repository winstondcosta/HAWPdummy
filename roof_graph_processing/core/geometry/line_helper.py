import math
from typing import Optional

from ...core.roof import Vertex, UndirectedEdge, Edge, Surface

from .common_helper import euclidean_distance, mod


def signed_polygon_area(x: list, y: list):  # SG
    """
    Shoelace formula to get the area of a polygon given the cartesian coordinates (x,y) of each vertex
    link: https://en.wikipedia.org/wiki/Shoelace_formula
    """
    length_x = len(x)
    length_y = len(y)

    try:
        assert length_x == length_y, "x and y coordinates have different set of points"

        assert length_x != 0, "zero point surface"

    except Exception as e:
        print("ERROR: " + str(e))
        return 0

    correction = x[length_x - 1] * y[0] - y[length_y - 1] * x[0]

    sum1 = 0
    for i in range(length_x - 1):
        sum1 += x[i] * y[i + 1]

    sum2 = 0
    for i in range(length_y - 1):
        sum2 += y[i] * x[i + 1]

    main_area = sum1 - sum2

    return 0.5 * (main_area + correction)


def check_orientation(surface: Surface):
    x = []
    y = []

    for vertex in surface.ordered_vertices:
        x.append(vertex.x)
        y.append(vertex.y)

    val = signed_polygon_area(x, y)

    if val == 0:
        return "COLLINEAR"

    else:
        if val > 0:
            return "CLOCKWISE"

        else:
            return "ANTICLOCKWISE"


def does_vertex_lie_within_line_segment(v: Vertex, ue: UndirectedEdge):
    # Only way a point say v can lie within a line segment formed by two vertices (v1, v2) is as follows:
    # dist(v, v1) + dist(v, v2) is equal to dist(v1, v2)
    return abs(euclidean_distance(v, ue.v1) + euclidean_distance(v, ue.v2) - euclidean_distance(ue.v1, ue.v2)) < 1e-8


def get_line_equation(ue: UndirectedEdge):
    # y = mx + c
    ue_m = (ue.v1.y - ue.v2.y) / (ue.v1.x - ue.v2.x + 1e-8)  # Calculate slope of line segment
    ue_c = ue.v1.y - ue_m * ue.v1.x  # y_intercept = y1 - slope * x1 = y2 - slope * x2
    return ue_m, ue_c


def get_point_of_intersection(ue1: UndirectedEdge, ue2: UndirectedEdge):
    # y = ue1_m * x + ue1_c  First equation
    ue1_m, ue1_c = get_line_equation(ue1)
    # y = ue2_m * x + ue2_c  Second line equation
    ue2_m, ue2_c = get_line_equation(ue2)

    x_intersection = (ue1_c - ue2_c) / (ue2_m - ue1_m + 1e-8)
    y_intersection = ue1_m * x_intersection + ue1_c

    return Vertex(x_intersection, y_intersection)


def get_foot_of_perpendicular(v: Vertex, ue: UndirectedEdge):
    # Calculate equation of undirected edge: y = mx + c
    # ue_m : slope, ue_c : y_intercept
    ue_m = (ue.v1.y - ue.v2.y) / (ue.v1.x - ue.v2.x + 1e-8)  # Calculate slope of line segment
    ue_c = ue.v1.y - ue_m * ue.v1.x  # y_intercept = y1 - slope * x1 = y2 - slope * x2

    # Calculate equation of line perpendicular to the undirected edge and passing through vertex V
    # perp_m : slope, perp_c = y
    perp_m = -1 / (ue_m + 1e-8)  # Slope of the perpendicular line
    perp_c = v.y - perp_m * v.x  # y_intercept = y - slope * x

    # Let us say there are 2 line equations: y = m1*x + c1 and y = m2*x + c2
    # The point of intersection will be:
    # x_inter = (c2 - c1)/(m1 - m2)
    # y_inter = m1 * x_inter + c1 = m2 * x_inter + c2
    x_foot = (perp_c - ue_c) / (ue_m - perp_m + 1e-8)
    y_foot = ue_m * x_foot + ue_c

    # Instantiate a vertex and return the foot of the perpendicular to the undirected edge
    return Vertex(x_foot, y_foot)


def get_order_vertex_list_based_on_line_segment_formation(old_vertex_id_list: list,
                                                          fusing_vertices_id_list: list,
                                                          vertices: dict):
    total_number_of_vertices = len(old_vertex_id_list) + len(fusing_vertices_id_list)

    ordered_vertex_list = {0: old_vertex_id_list[0],
                           2 ** total_number_of_vertices: old_vertex_id_list[-1]}

    for fusing_vertex_id in fusing_vertices_id_list:
        fusing_vertex = vertices[fusing_vertex_id]  # Get the fusing vertex
        ordered_vertex_list_keys = sorted(ordered_vertex_list)
        for v_num in range(len(ordered_vertex_list_keys) - 1):
            v1_id = ordered_vertex_list[ordered_vertex_list_keys[v_num]]
            v2_id = ordered_vertex_list[ordered_vertex_list_keys[v_num + 1]]

            # Two ends of the potential new edge
            v1 = vertices[v1_id]
            v2 = vertices[v2_id]

            ue_v1_v2 = UndirectedEdge(v1, v2)

            foot_of_perp = get_foot_of_perpendicular(fusing_vertex, ue_v1_v2)

            if not does_vertex_lie_within_line_segment(foot_of_perp, ue_v1_v2):
                # This means this does not fall within the new edge
                continue

            else:
                # Vertex will lie within
                ordered_vertex_list[(ordered_vertex_list_keys[v_num] + ordered_vertex_list_keys[v_num + 1]) // 2] = \
                    fusing_vertex_id
                break

    ordered_vertex_list_keys = sorted(ordered_vertex_list)

    final_ordered_list = []
    for v_num in ordered_vertex_list_keys:
        vertex_id = ordered_vertex_list[v_num]
        final_ordered_list.append(vertex_id)

    return final_ordered_list


def check_if_edges_intersect(ue1: UndirectedEdge, ue2: UndirectedEdge):  # GFG
    ue1_v1, ue1_v2 = ue1.v1, ue1.v2  # p1, q1
    ue2_v1, ue2_v2 = ue2.v1, ue2.v2  # p2, q2

    orient_1 = check_orientation(Surface([ue1_v1, ue1_v2, ue2_v1], order_anticlockwise=False))
    orient_2 = check_orientation(Surface([ue1_v1, ue1_v2, ue2_v2], order_anticlockwise=False))
    orient_3 = check_orientation(Surface([ue2_v1, ue2_v2, ue1_v1], order_anticlockwise=False))
    orient_4 = check_orientation(Surface([ue2_v1, ue2_v2, ue1_v2], order_anticlockwise=False))

    if orient_1 != orient_2 and orient_3 != orient_4:
        return True

    if orient_1 == "COLLINEAR" and does_vertex_lie_within_line_segment(ue2_v1, UndirectedEdge(ue1_v1, ue1_v2)):
        return True

    if orient_2 == "COLLINEAR" and does_vertex_lie_within_line_segment(ue2_v2, UndirectedEdge(ue1_v1, ue1_v2)):
        return True

    if orient_3 == "COLLINEAR" and does_vertex_lie_within_line_segment(ue1_v1, UndirectedEdge(ue2_v1, ue2_v2)):
        return True

    if orient_4 == "COLLINEAR" and does_vertex_lie_within_line_segment(ue1_v2, UndirectedEdge(ue2_v1, ue2_v2)):
        return True

    return False


def angle_wrt_x_axis(y, x):  # SG
    return mod(math.atan2(y, x), 2 * math.pi)


def distance_between_vertex_and_line_segment(v: Vertex, ue: UndirectedEdge):
    v_foot_of_perp = get_foot_of_perpendicular(v, ue)
    if does_vertex_lie_within_line_segment(v_foot_of_perp, ue):
        return euclidean_distance(v_foot_of_perp, v)

    return min(euclidean_distance(v, ue.v1), euclidean_distance(v, ue.v2))


def distance_between_two_line_segments(ue1: UndirectedEdge, ue2: UndirectedEdge):  # Stack Overflow
    # For reference: https://stackoverflow.com/questions/541150/connect-two-line-segments/11427699#11427699
    if check_if_edges_intersect(ue1, ue2):
        return 0.0

    dists = [
        distance_between_vertex_and_line_segment(ue1.v1, ue2),
        distance_between_vertex_and_line_segment(ue1.v2, ue2),
        distance_between_vertex_and_line_segment(ue2.v1, ue1),
        distance_between_vertex_and_line_segment(ue2.v2, ue1)
    ]

    return min(dists)


def distance_between_two_line_segments_using_vertices(ue1: UndirectedEdge, ue2: UndirectedEdge):
    if ue1.v1.vertex_id == ue2.v1.vertex_id and ue1.v2.vertex_id == ue2.v2.vertex_id:
        return 0.0

    if ue1.v1.vertex_id == ue2.v2.vertex_id and ue1.v2.vertex_id == ue2.v1.vertex_id:
        return 0.0

    dist1 = euclidean_distance(ue1.v1, ue2.v1) + euclidean_distance(ue1.v2, ue2.v2)
    dist2 = euclidean_distance(ue1.v1, ue2.v2) + euclidean_distance(ue1.v2, ue2.v1)

    return min(dist1, dist2)


def get_points_of_intersection(undirected_edges):
    ue_to_poI_map = {}  # Undirected edge id to points of intersection
    poI_to_ue_map = {}  # Points of intersection to undirected edges

    undirected_edge_ids = list(undirected_edges.keys())

    for ue_idx_1 in range(len(undirected_edge_ids)):
        ue1_id = undirected_edge_ids[ue_idx_1]
        ue1 = undirected_edges[ue1_id]
        end_vertices1 = [ue1.v1.vertex_id, ue1.v2.vertex_id]

        if ue1_id not in ue_to_poI_map:
            ue_to_poI_map[ue1_id] = []

        for ue_idx_2 in range(ue_idx_1 + 1, len(undirected_edge_ids)):
            ue2_id = undirected_edge_ids[ue_idx_2]
            ue2 = undirected_edges[ue2_id]
            # Dont do for edges already having one vertex as intersection
            if ue2.v1.vertex_id in end_vertices1 or ue2.v2.vertex_id in end_vertices1:
                continue
            if ue2_id not in ue_to_poI_map:
                ue_to_poI_map[ue2_id] = []

            poI = get_point_of_intersection(ue1, ue2)

            if does_vertex_lie_within_line_segment(poI, ue1) and does_vertex_lie_within_line_segment(poI, ue2):

                # populate the map from undirected edges to the point of intersection list
                ue_to_poI_map[ue1_id].append(poI)
                ue_to_poI_map[ue2_id].append(poI)

                # populate the map from point of intersection to the undirected edge list
                if poI.vertex_id not in poI_to_ue_map:
                    poI_to_ue_map[poI.vertex_id] = []

                if ue1.edge_id not in poI_to_ue_map[poI.vertex_id]:
                    poI_to_ue_map[poI.vertex_id].append(ue1)

                if ue2.edge_id not in poI_to_ue_map[poI.vertex_id]:
                    poI_to_ue_map[poI.vertex_id].append(ue2)

    return ue_to_poI_map, poI_to_ue_map


def get_angle_between_two_line_segments(ue1: UndirectedEdge, ue2: UndirectedEdge):
    ue1_angle, ue2_angle = ue1.edge_angle(), ue2.edge_angle()

    diff = abs(ue1_angle - ue2_angle)

    while diff >= 180:
        diff -= 180

    if diff < 0:
        diff += 180

    return diff


def edge_to_vector(e: Edge):
    """
    Converts a directed edge into a unit vector with a direction from start ----> end
    Args:
        e: Edge with start

    Returns: The unit vector as an object of class Vertex

    """
    edge_vector = Vertex(e.end.x - e.start.x, e.end.y - e.start.y)
    edge_vector_length = max(math.sqrt(edge_vector.x ** 2 + edge_vector.y ** 2), 1e-8)
    edge_vector.x, edge_vector.y = edge_vector.x / edge_vector_length, edge_vector.y / edge_vector_length
    return edge_vector


def get_angle_between_two_directed_edges(e1: Edge, e2: Edge):
    """
    Calculates the angle between two Edge objects which have a direction from start ----> end
    Args:
        e1: Edge object 1
        e2: Edge object 2

    Returns: angle in degrees
    """

    edge_vector1 = edge_to_vector(e1)
    edge_vector2 = edge_to_vector(e2)

    dot_product = edge_vector1.x * edge_vector2.x + edge_vector1.y * edge_vector2.y
    dot_product = min(1, max(dot_product, -1))
    angle_in_radians = math.acos(dot_product)

    return angle_in_radians * 180.0 / math.pi


def rotate_vertex(free_point: Vertex, anchor_point: Vertex, angle_in_degrees, mutate_free_point=False) -> \
        Optional[Vertex]:
    """
    Method to rotate the free_point vertex to a new vertex after rotation

    :param Vertex free_point: The point to rotate
    :param Vertex anchor_point: The anchor point about which the free point is to be rotated
    :param float angle_in_degrees: The angle about which the free point is to be rotated
    :param bool mutate_free_point: If true, will mutate free_point vertex, else will create a new Vertex instance
    :return Optional[Vertex] new_free_point: if mutate_free_point is true, it will overwrite the free_point's x and y
                                             values (Note: This is pass by reference) else, it will create a new vertex
                                             with the newly calculated coordinates and return that object
    """
    # Translation operation making the anchor the origin
    vector_ = Vertex(free_point.x - anchor_point.x, free_point.y - anchor_point.y)

    # Convert angle in degrees to radian
    angle_in_radians = (math.pi * angle_in_degrees / 180)

    # Calculate sin and cos
    sin_theta = math.sin(angle_in_radians)
    cos_theta = math.cos(angle_in_radians)

    # Rotate the vector anticlockwise
    vector_x = vector_.x * cos_theta + vector_.y * sin_theta
    vector_y = -vector_.x * sin_theta + vector_.y * cos_theta

    # Translating origin back to 0,0
    new_point_x = vector_x + anchor_point.x
    new_point_y = vector_y + anchor_point.y

    # Change the free_point vertex
    if mutate_free_point:
        free_point.x = new_point_x
        free_point.y = new_point_y
        return

    new_free_point = Vertex(new_point_x, new_point_y, meta=free_point.meta)

    return new_free_point


def move_free_point_anchor_point_gets_fixed(free_v: Vertex, anchor_v: Vertex, fixed_v: Vertex, support_v: Vertex,
                                            mutate_free_point=False) -> Optional[Vertex]:
    """
    This method moves free_v along the line joining (support_v and free_v) such that the angle at the anchor_v
    is modified and become orthogonal

    :param Vertex free_v: free vertex connected to anchor_point
    :param Vertex anchor_v: anchor_point
    :param Vertex fixed_v: fixed vertex connected to anchor point (anchor_v)
    :param Vertex support_v: the other vertex connected to the free vertex (free_v)
    :param bool mutate_free_point: If true, will mutate free vertex (free_v), else will create a new Vertex instance
    :return Optional[Vertex]: if mutate_free_point is true, it will overwrite the free_point's x and y values
                              (Note: This is pass by reference) else, it will create a new vertex with the newly
                              calculated coordinates and return that object
    """
    # Line_1 represented as a1x + b1y = c1
    # perpendicular to line btw anchor_v and fixed_v, passing through anchor_v
    a1 = fixed_v.x - anchor_v.x + 1e-6
    b1 = fixed_v.y - anchor_v.y + 1e-6
    c1 = a1 * anchor_v.x + b1 * anchor_v.y

    # Line_2 represented as a2x + b2y = c2
    # Line btw free_v and support_v
    a2 = support_v.y - free_v.y
    b2 = free_v.x - support_v.x + 1e-6
    c2 = a2 * support_v.x + b2 * support_v.y

    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return free_v
    else:
        x_new = (b2 * c1 - b1 * c2) / determinant
        y_new = (a1 * c2 - a2 * c1) / determinant
        # Change the free_point vertex
        if mutate_free_point:
            free_v.x = x_new
            free_v.y = y_new
            return
        # Instantiate a vertex and return
        return Vertex(x_new, y_new, meta=free_v.meta)


def foot_of_perpendicular_to_line_joining_anchor_and_fixed_vertices(free_v: Vertex, anchor_v: Vertex, fixed_v: Vertex,
                                                                    mutate_anchor_point=False) -> Optional[Vertex]:
    """
    This method generates foot of perpendicular from free_v to the line joining (anchor_v and fixed_v)

    :param Vertex free_v: free vertex connected to anchor_point
    :param Vertex anchor_v: anchor_point
    :param Vertex fixed_v: fixed vertex connected to anchor point (anchor_v)
    :param bool mutate_anchor_point: If True anchor point is moved to foot to perpendicular point else new vertex is
                                     created
    :return Optional[Vertex]: if mutate_anchor_point is true, it will overwrite the anchor_point's x and y values
                              else, it will create a new vertex with the newly calculated coordinates and return that
                              object
    """
    # Calculate equation of undirected edge: y = mx + c
    # ue_m : slope, ue_c : y_intercept
    ue_m = (fixed_v.y - anchor_v.y) / (fixed_v.x - anchor_v.x + 1e-8)  # Calculate slope of line segment
    ue_c = fixed_v.y - ue_m * fixed_v.x  # y_intercept = y1 - slope * x1 = y2 - slope * x2

    # Calculate equation of line perpendicular to the undirected edge and passing through vertex free_v
    # perp_m : slope, perp_c = y
    perp_m = -1 / (ue_m + 1e-8)  # Slope of the perpendicular line
    perp_c = free_v.y - perp_m * free_v.x  # y_intercept = y - slope * x

    # Let us say there are 2 line equations: y = m1*x + c1 and y = m2*x + c2
    # The point of intersection will be:
    # x_inter = (c2 - c1)/(m1 - m2)
    # y_inter = m1 * x_inter + c1 = m2 * x_inter + c2
    x_foot = (perp_c - ue_c) / (ue_m - perp_m + 1e-8)
    y_foot = ue_m * x_foot + ue_c

    # Change the anchor point vertex
    if mutate_anchor_point:
        anchor_v.x = x_foot
        anchor_v.y = y_foot
        return

    # Instantiate a vertex and return the foot of the perpendicular to the edge joining anchor_v, fixed_v
    return Vertex(x_foot, y_foot, meta=anchor_v.meta)
