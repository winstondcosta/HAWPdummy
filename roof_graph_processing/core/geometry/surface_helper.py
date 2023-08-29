from ...core.roof import Vertex, Edge, UndirectedEdge, Surface

from .line_helper import does_vertex_lie_within_line_segment, get_foot_of_perpendicular, signed_polygon_area, \
    get_point_of_intersection

from .common_helper import euclidean_distance


def polygon_area(x: list, y: list):  # SG
    return abs(signed_polygon_area(x, y))


def surface_area(surface: Surface):
    x_list = [v.x for v in surface.ordered_vertices]
    y_list = [v.y for v in surface.ordered_vertices]

    return polygon_area(x_list, y_list)


def does_point_lie_within_a_polygon(v: Vertex, surface: Surface):  # GFG
    # v: An object of class Vertex
    # surface: An object of class Surface

    ordered_vertices = surface.ordered_vertices  # Ordered vertex signifies a polygon

    num_points_in_surface = len(ordered_vertices)

    if num_points_in_surface < 3:
        return False

    extreme_v = Vertex(1e9, v.y)
    count, i = 0, 0

    vertex_ray = UndirectedEdge(v, extreme_v)

    while True:
        next_point_idx = (i + 1) % num_points_in_surface

        surface_edge = UndirectedEdge(ordered_vertices[i], ordered_vertices[next_point_idx])

        poI = get_point_of_intersection(surface_edge, vertex_ray)

        if does_vertex_lie_within_line_segment(poI, vertex_ray) \
                and does_vertex_lie_within_line_segment(poI, surface_edge):
            count = count + 1

        i = next_point_idx

        if i == 0:
            break

    return count % 2 == 1


def distance_between_point_and_polygon(v: Vertex, surface: Surface):
    # Defining a distance between a point and a polygon

    distance_of_vertex_to_polygon = 1e9

    for v_poly_id in range(len(surface.ordered_vertices)):

        v_poly = surface.ordered_vertices[v_poly_id]
        distance_of_vertex_to_polygon = min(distance_of_vertex_to_polygon,
                                            euclidean_distance(v, v_poly))

        v_poly_next_id = (v_poly_id + 1) % len(surface.ordered_vertices)
        v_poly_next = surface.ordered_vertices[v_poly_next_id]

        v_foot_of_perpendicular = get_foot_of_perpendicular(v, UndirectedEdge(v_poly, v_poly_next))

        if does_vertex_lie_within_line_segment(v, UndirectedEdge(v_poly, v_poly_next)):
            distance_of_vertex_to_polygon = min(distance_of_vertex_to_polygon,
                                                euclidean_distance(v, v_foot_of_perpendicular))

    return distance_of_vertex_to_polygon


def get_polygon_centroid(surface: Surface):
    sum_x, sum_y = 0, 0

    for vertex in surface.ordered_vertices:
        sum_x += vertex.x
        sum_y += vertex.y

    centroid = Vertex(sum_x / (len(surface.ordered_vertices) + 1e-8),
                      sum_y / (len(surface.ordered_vertices) + 1e-8))

    return centroid


def inflate_polygon(surface: Surface, inflation_distance=-3.0):
    shapely_polygon = surface.convert_surface_to_shapely_polygon().buffer(inflation_distance)

    x = list(shapely_polygon.exterior.xy)[0]
    y = list(shapely_polygon.exterior.xy)[1]

    inflated_ordered_vertices = []

    for xc, yc in zip(x, y):
        inflated_ordered_vertices.append(Vertex(int(xc), int(yc)))

    if not inflated_ordered_vertices:
        polygon_centroid = get_polygon_centroid(surface)
        inflated_ordered_vertices = [polygon_centroid for _ in range(len(surface.ordered_vertices))]

    inflated_surface = Surface(inflated_ordered_vertices)

    return inflated_surface


def get_longest_edge(surface: Surface) -> Edge:
    """
    Finds the longest edge

    :param Surface surface:
    :return Edge: longest edge
    """
    edge_list = surface.edge_list

    # Find the index of the longest edge
    longest_edge_idx = -1
    longest_edge_length = 0
    for i in range(len(edge_list)):
        if edge_list[i].edge_length() > longest_edge_length:
            longest_edge_idx = i
            longest_edge_length = edge_list[i].edge_length()

    return edge_list[longest_edge_idx]
