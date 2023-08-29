import copy
import logging
from roof_graph_processing.core.roof import Graph, Vertex
from roof_graph_processing.core.geometry.common_helper import find_angle_btw_vertices, euclidean_distance
from roof_graph_processing.core.geometry.line_helper import rotate_vertex, move_free_point_anchor_point_gets_fixed, \
    foot_of_perpendicular_to_line_joining_anchor_and_fixed_vertices
from roof_graph_processing.core.geometry.surface_helper import get_longest_edge


logger = logging.getLogger("Post-processing")


# NOTE: Currently this method is implemented assuming one moves in anti-clock-wise direction
def is_reflex(v1: Vertex, v2: Vertex, v3: Vertex) -> bool:
    """
    This method calculates whether angle subtended at v2 vertex is reflex or not. It assumes v1, v2, v3 are in
    anti-clock wise order

    :param Vertex v1: vertex
    :param Vertex v2: vertex
    :param Vertex v3: vertex
    :return bool: True (if angle subtended by v2 is reflex) else False
    """
    x1_diff = v2.x - v1.x
    y1_diff = v2.y - v1.y
    x2_diff = v3.x - v2.x
    y2_diff = v3.y - v2.y
    if abs(x1_diff) > abs(y1_diff):
        if x1_diff > 0:
            if y2_diff > 0:
                return True
        else:
            if y2_diff < 0:
                return True
    else:
        if y1_diff > 0:
            if x2_diff < 0:
                return True
        else:
            if x2_diff > 0:
                return True
    return False


def angle_subtended_by_vertex_to_the_roof(angle: float, reflex_info: bool) -> float:
    """
    This method calculates the original angle subtended by vertex of the roof (in degrees)

    :param float angle: rotation angle in order to make the angle orthogonal
    :param bool reflex_info: whether the angle subtended by vertex is reflex or not
    :return float: original angle subtended by vertex (in degrees)
    """
    if reflex_info:
        return 270 - angle
    else:
        return angle + 90


def check_for_restoration(v1: Vertex, new_v: Vertex, orig_v: Vertex, v2: Vertex) -> bool:
    """
    This method checks for the change in edge length between modified vertex and its adjacent unmodified connected
    vertex, based on which modified vertex is restored to its original place.

    :param Vertex v1: vertex connected to the modified_vertex
    :param Vertex new_v: vertex after orthogonal correction
    :param Vertex orig_v: vertex before orthogonal correction
    :param Vertex v2: other vertex connected to the modified_vertex
    :return bool: True if restored else False
    """
    ed_free_anchor_vertices = euclidean_distance(orig_v, v2)
    ed_btw_changed_vertices = euclidean_distance(orig_v, new_v)
    ed_new = euclidean_distance(v1, new_v)
    ed_old = euclidean_distance(v1, orig_v)
    if ed_old <= ed_free_anchor_vertices:
        t_ratio = 1 / 3  # experimental
    else:
        t_ratio = 1 / 4  # experimental
    if ((ed_old > ed_new > (1 - t_ratio) * ed_old) and ed_btw_changed_vertices < ed_old) or \
            ((ed_old < ed_new < (1 + t_ratio) * ed_old) and ed_btw_changed_vertices < ed_old):
        return False
    new_v.x = orig_v.x
    new_v.y = orig_v.y
    return True


def update_modified_angles(v1: Vertex, adj_vertices: dict, id_to_vertex: dict, rotation_angle_dic: dict,
                           fixed_vertices: dict) -> None:
    """
    This method checks for the change in edge length between modified vertex and its adjacent unmodified connected
    vertex, based on which modified vertex is restored to its original place.

    :param Vertex v1: vertex connected to the changed vertex
    :param dict adj_vertices: dict which maps each vertex id to its adjacent vertices'
    :param dict id_to_vertex: dict which maps vertex id to vertex object
    :param dict rotation_angle_dic: dict which contains the rotation angle for each vertex to make perimeter orthogonal
    :param dict fixed_vertices: dict which contains whether given vertex is fixed or not
    :return None:
    """
    adj_vertices_list = adj_vertices[v1.vertex_id]
    angle_subtended = find_angle_btw_vertices(id_to_vertex[adj_vertices_list[0]], v1,
                                              id_to_vertex[adj_vertices_list[1]])
    rotation_angle_dic[v1.vertex_id] = angle_subtended - 90
    if angle_subtended == 90 or angle_subtended == 180:
        fixed_vertices[v1.vertex_id] = True


def orthogonal_corrections_where_anchor_angle_is_less_than_90_or_270(free_point: Vertex, anchor_point: Vertex,
                                                                     fixed_point: Vertex, reflex_info: dict,
                                                                     adj_vertices: dict, id_to_vertex: dict,
                                                                     rotation_angle_dic: dict, fixed_vertices: dict,
                                                                     rotation_angle: float, t_angle: float,
                                                                     case_index: int) -> bool:
    """
    Helper method which implements orthogonal corrections at the anchor vertex for the following scenarios:
    1. angle subtended at vertex is between 0 and 90
    2. angle subtended at vertex is between 180 and 270
    This module changes the vertex in-place.

    :param Vertex free_point: free vertex connected to anchor_vertex
    :param Vertex anchor_point: anchor_vertex
    :param Vertex fixed_point: fixed vertex connected to anchor point (anchor_point)
    :param dict reflex_info: dict which says whether at given vertex we need to take reflex angle or not
    :param dict adj_vertices: dict which maps each vertex id to its adjacent vertices'
    :param dict id_to_vertex: dict which maps vertex id to vertex object
    :param dict rotation_angle_dic: dict which contains the rotation angle for each vertex to make perimeter orthogonal
    :param dict fixed_vertices: dict which contains whether given vertex is fixed or not
    :param float rotation_angle: the rotation angle for anchor vertex to make perimeter orthogonal (signed based on
                                direction)
    :param float t_angle: angle used in [To Avoid Facet disappearance] condition,
                 90 if (angle subtended at vertex is between 0 and 90) else 270
    :param int case_index: index used to indicate which case scenario in the logs
    :return bool: True if orthogonal correction at anchor vertex is skipped else False
    """
    orig_anchor_point = copy.deepcopy(anchor_point)
    angle_subtended_at_anchor_vertex = angle_subtended_by_vertex_to_the_roof(rotation_angle,
                                                                             reflex_info[anchor_point.vertex_id])
    angle_subtended_at_free_vertex = angle_subtended_by_vertex_to_the_roof(
        rotation_angle_dic[free_point.vertex_id], reflex_info[free_point.vertex_id])

    # Scenario need to implement [To Avoid Facet disappearance]
    if (t_angle - angle_subtended_at_anchor_vertex) >= angle_subtended_at_free_vertex:
        # logger.info(f"Orthogonal corrections: Case{case_index} - vertex skipped")
        return True
    else:
        foot_of_perpendicular_to_line_joining_anchor_and_fixed_vertices(free_point, anchor_point,
                                                                        fixed_point,
                                                                        mutate_anchor_point=True)
    if check_for_restoration(fixed_point, anchor_point, orig_anchor_point, free_point):
        # logger.info(f"Orthogonal corrections: Case{case_index + 1} - vertex skipped")
        return True
    else:
        # logger.info(f"Orthogonal corrections: Case{case_index + 1} - vertex modified")
        update_modified_angles(free_point, adj_vertices, id_to_vertex, rotation_angle_dic,
                               fixed_vertices)
    return False


def orthogonal_corrections_where_anchor_angle_is_greater_than_90_or_270(free_point: Vertex, anchor_point: Vertex,
                                                                        fixed_point: Vertex, support_point: Vertex,
                                                                        adj_vertices: dict, id_to_vertex: dict,
                                                                        rotation_angle_dic: dict, fixed_vertices: dict,
                                                                        rotation_angle: float, case_index: int) -> bool:
    """
    Helper method which implements orthogonal corrections at the anchor vertex for the following scenarios:
    1. angle subtended at vertex is between 90 and 180
    2. angle subtended at vertex is between 270 and 360
    This module changes the vertex in-place.

    :param Vertex free_point: free vertex connected to anchor_vertex
    :param Vertex anchor_point: anchor_vertex
    :param Vertex fixed_point: fixed vertex connected to anchor point (anchor_point)
    :param Vertex support_point: the other vertex connected to the free vertex (free_point)
    :param dict adj_vertices: dict which maps each vertex id to its adjacent vertices'
    :param dict id_to_vertex: dict which maps vertex id to vertex object
    :param dict rotation_angle_dic: dict which contains the rotation angle for each vertex to make perimeter orthogonal
    :param dict fixed_vertices: dict which contains whether given vertex is fixed or not
    :param float rotation_angle: the rotation angle for anchor vertex to make perimeter orthogonal (signed based on
                                direction)
    :param int case_index: index used to indicate which case scenario in the logs
    :return bool: True if orthogonal correction at anchor vertex is skipped else False
    """
    orig_free_point = copy.deepcopy(free_point)
    move_free_point_anchor_point_gets_fixed(free_point, anchor_point, fixed_point, support_point,
                                            mutate_free_point=True)
    if check_for_restoration(support_point, free_point, orig_free_point, anchor_point) or \
            (free_point.x == orig_free_point.x and free_point.y == orig_free_point.y):
        if fixed_vertices[support_point.vertex_id]:
            # logger.info(f"Orthogonal corrections: Case{case_index} and Case{case_index + 1} Skipped - vertex skipped")
            return True
        rotate_vertex(free_point, anchor_point, rotation_angle, mutate_free_point=True)
        if check_for_restoration(support_point, free_point, orig_free_point, anchor_point) or \
                (free_point.x == orig_free_point.x and free_point.y == orig_free_point.y):
            # logger.info(f"Orthogonal corrections: Case{case_index} and Case{case_index + 1} Skipped - vertex skipped")
            return True
        else:
            # logger.info(f"Orthogonal corrections: Case{case_index} - vertex modified")
            pass
    else:
        # logger.info(f"Orthogonal corrections: Case{case_index + 1} - vertex modified")
        pass
    update_modified_angles(free_point, adj_vertices, id_to_vertex, rotation_angle_dic,
                           fixed_vertices)
    update_modified_angles(support_point, adj_vertices, id_to_vertex, rotation_angle_dic,
                           fixed_vertices)
    return False


def orthogonal_corrections(graph: Graph, angle_threshold_for_fixed_point: int = 2,
                           angle_threshold_for_correction: int = 8) -> (int, int, int):
    """
    Find all the outer contour vertices which are fixed based on the angle_threshold_for_fixed_point.
    For remaining outer contour vertices, traverse one by one in anti-clock wise direction and based on the angle
    subtended at that vertex (different scenarios), correct the vertices so that perimeter orthogonality is achieved.
    This module changes the vertex in-place.

    :param Graph graph: The roof graph object
    :param int angle_threshold_for_fixed_point: The threshold for considering fixed point
    :param int angle_threshold_for_correction: The threshold for considering equality
    :return int targeted_vertices_count: number of vertices targeted to be modified
    :return int modified_vertices_count: number of vertices modified directly
    :return int skipped_vertices_count: number of vertices skipped which are in given range
    """
    logger.info("Orthogonal corrections started")
    # Get list of all the outer contour vertices
    outer_contour = graph.outer_contour
    outer_contour_ordered_vertices = outer_contour.ordered_vertices

    if len(outer_contour_ordered_vertices) <= 2:
        return 0, 0, 0

    # Fixing the start point for orthogonal corrections
    longest_edge = get_longest_edge(outer_contour)
    longest_edge_start_vertex_id = longest_edge.start.vertex_id  # we move in anti-clock-wise direction
    longest_edge_start_vertex_idx = -1
    for i, t_vertex in enumerate(outer_contour_ordered_vertices):
        if t_vertex.vertex_id == longest_edge_start_vertex_id:
            longest_edge_start_vertex_idx = i
            break
    outer_contour_ordered_vertices = outer_contour_ordered_vertices[longest_edge_start_vertex_idx:] + \
                                        outer_contour_ordered_vertices[:longest_edge_start_vertex_idx]

    vertices_set = {}  # dict which contains the vertices to be modified
    fixed_vertices = {}  # dict which contains whether given vertex is fixed or not
    rotation_angle_dic = {}  # dict which contains the rotation angle for each vertex to make perimeter orthogonal
    adj_vertices = {}  # dict which maps each vertex id to its adjacent vertices
    id_to_vertex = {}  # dict which maps vertex id to vertex object
    reflex_info = {}  # dict which says whether at given vertex we need to take reflex angle or not
    len_outer_contour_vertices = len(outer_contour_ordered_vertices)
    for i, t_vertex in enumerate(outer_contour_ordered_vertices):
        if i == 0:
            adj_vertices[t_vertex.vertex_id] = [
                outer_contour_ordered_vertices[len_outer_contour_vertices - 1].vertex_id,
                outer_contour_ordered_vertices[i + 1].vertex_id]
        elif i == len_outer_contour_vertices - 1:
            adj_vertices[t_vertex.vertex_id] = [outer_contour_ordered_vertices[i - 1].vertex_id,
                                                outer_contour_ordered_vertices[0].vertex_id]
        else:
            adj_vertices[t_vertex.vertex_id] = [outer_contour_ordered_vertices[i - 1].vertex_id,
                                                outer_contour_ordered_vertices[i + 1].vertex_id]
        vertices_set[i] = t_vertex.vertex_id
        id_to_vertex[t_vertex.vertex_id] = t_vertex
        fixed_vertices[t_vertex.vertex_id] = False
        rotation_angle_dic[t_vertex.vertex_id] = None
        reflex_info[t_vertex.vertex_id] = False

    for i, t_vertex in enumerate(outer_contour_ordered_vertices):
        adj_vertices_list = adj_vertices[t_vertex.vertex_id]
        reflex_info[t_vertex.vertex_id] = is_reflex(id_to_vertex[adj_vertices_list[0]], t_vertex,
                                                    id_to_vertex[adj_vertices_list[1]])
        angle_subtended = find_angle_btw_vertices(id_to_vertex[adj_vertices_list[0]], t_vertex,
                                                  id_to_vertex[adj_vertices_list[1]])

        # The angle of rotation
        rotation_angle_dic[t_vertex.vertex_id] = angle_subtended - 90

        # finding the fixed points
        if abs(180 - angle_subtended) < angle_threshold_for_fixed_point:
            fixed_vertices[t_vertex.vertex_id] = True
            vertices_set.pop(i)
            continue

        if abs(90 - angle_subtended) > angle_threshold_for_correction:
            vertices_set.pop(i)
            continue

        if abs(90 - angle_subtended) < angle_threshold_for_fixed_point:
            fixed_vertices[t_vertex.vertex_id] = True
            vertices_set.pop(i)

    targeted_vertices_count = len(vertices_set)
    # logger.info(f"Orthogonal corrections: vertices targeted to be modified {targeted_vertices_count}")
    modified_vertices_count = 0
    skipped_vertices_count = 0
    while len(vertices_set):
        list_check_set = list(vertices_set.keys())
        for t_index in list_check_set:
            t_vertex_id = outer_contour_ordered_vertices[t_index].vertex_id
            rotation_angle = rotation_angle_dic[t_vertex_id]

            if fixed_vertices[t_vertex_id]:
                # logger.info("Orthogonal corrections: Case where vertex modified from other vertex changes")
                vertices_set.pop(t_index)
                modified_vertices_count += 1
                continue

            # fixing free point and anchor point
            # check for fixed points
            t_adj_vertices = adj_vertices[t_vertex_id]
            left_vertex_fixed = fixed_vertices[t_adj_vertices[0]]
            right_vertex_fixed = fixed_vertices[t_adj_vertices[1]]

            # if both adjacent vertices are fixed, do nothing
            if left_vertex_fixed and right_vertex_fixed:
                # logger.info("Orthogonal corrections: Case1 - vertex skipped")
                fixed_vertices[t_vertex_id] = True
                vertices_set.pop(t_index)
                skipped_vertices_count += 1
                continue
            elif right_vertex_fixed:
                free_point = id_to_vertex[t_adj_vertices[0]]
                fixed_point = id_to_vertex[t_adj_vertices[1]]
                support_point = id_to_vertex[adj_vertices[free_point.vertex_id][0]]
            else:
                free_point = id_to_vertex[t_adj_vertices[1]]
                fixed_point = id_to_vertex[t_adj_vertices[0]]
                support_point = id_to_vertex[adj_vertices[free_point.vertex_id][1]]
            anchor_point = id_to_vertex[t_vertex_id]

            # Always try to make change in the direction towards the roof centroid when ever possible
            # Angle subtended at vertex is reflex
            if reflex_info[t_vertex_id]:

                # angle subtended at vertex is between 180 and 270
                if rotation_angle > 0:
                    if orthogonal_corrections_where_anchor_angle_is_less_than_90_or_270(free_point, anchor_point,
                                                                                        fixed_point, reflex_info,
                                                                                        adj_vertices, id_to_vertex,
                                                                                        rotation_angle_dic,
                                                                                        fixed_vertices, rotation_angle,
                                                                                        t_angle=270, case_index=2):
                        vertices_set.pop(t_index)
                        skipped_vertices_count += 1
                        continue

                # angle subtended at vertex is between 270 and 360
                else:
                    if not right_vertex_fixed:
                        rotation_angle = -1 * rotation_angle
                    if orthogonal_corrections_where_anchor_angle_is_greater_than_90_or_270(free_point, anchor_point,
                                                                                           fixed_point, support_point,
                                                                                           adj_vertices, id_to_vertex,
                                                                                           rotation_angle_dic,
                                                                                           fixed_vertices,
                                                                                           rotation_angle,
                                                                                           case_index=4):
                        vertices_set.pop(t_index)
                        skipped_vertices_count += 1
                        continue

            # Angle subtended at vertex is not reflex
            else:

                # angle subtended at vertex is between 90 and 180
                if rotation_angle > 0:
                    if right_vertex_fixed:
                        rotation_angle = -1 * rotation_angle
                    if orthogonal_corrections_where_anchor_angle_is_greater_than_90_or_270(free_point, anchor_point,
                                                                                           fixed_point, support_point,
                                                                                           adj_vertices, id_to_vertex,
                                                                                           rotation_angle_dic,
                                                                                           fixed_vertices,
                                                                                           rotation_angle,
                                                                                           case_index=6):
                        vertices_set.pop(t_index)
                        skipped_vertices_count += 1
                        continue

                # angle subtended at vertex is between 0 and 90
                else:
                    if orthogonal_corrections_where_anchor_angle_is_less_than_90_or_270(free_point, anchor_point,
                                                                                        fixed_point, reflex_info,
                                                                                        adj_vertices, id_to_vertex,
                                                                                        rotation_angle_dic,
                                                                                        fixed_vertices, rotation_angle,
                                                                                        t_angle=90, case_index=8):
                        vertices_set.pop(t_index)
                        skipped_vertices_count += 1
                        continue

            modified_vertices_count += 1
            rotation_angle_dic[t_vertex_id] = 0
            fixed_vertices[t_vertex_id] = True
            vertices_set.pop(t_index)

    if targeted_vertices_count != modified_vertices_count + skipped_vertices_count:
        logger.warning(f"Orthogonal corrections: All targeted vertices are not verified properly")

    logger.info(f"Orthogonal corrections: targeted vertices -- {targeted_vertices_count} directly modified vertices -- "
                f"{modified_vertices_count} skipped vertices -- {skipped_vertices_count}")
    logger.info("Orthogonal corrections end")
    return targeted_vertices_count, modified_vertices_count, skipped_vertices_count
