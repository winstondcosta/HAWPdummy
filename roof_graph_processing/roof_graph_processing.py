import logging
from roof_graph_processing.core.process_roof_graph import remove_intersecting_lines, fuse_junctions_and_create_graph, \
    fuse_junctions_with_edges_and_create_graph, remove_overlapping_lines, identify_disjoint_graphs, find_roof_surfaces, \
    remove_repeated_edges, identify_disjoint_sub_graphs, get_central_roof_graph

from roof_graph_processing.core.geometry.graph_helper import create_adjacency_list_from_undirected_edges
from roof_graph_processing.post_processing.outer_contour_post_processing import \
    orthogonal_corrections
from roof_graph_processing.post_processing.geometry_check import EstimateRoofSymmetry
from roof_graph_processing.post_processing.helper import get_undirected_edges_from_sub_graphs

logger = logging.getLogger("Post-processing")


def threshold(edges, edge_scores, edge_score_thresh=0.5):
    new_edges = []
    new_edge_scores = []

    for edge, edge_score in zip(edges, edge_scores):
        if edge_score < edge_score_thresh:
            continue

        new_edges.append(edge)
        new_edge_scores.append(edge_score)

    return new_edges, new_edge_scores


def build_roof_graph(img_shape: list, vertices: dict, undirected_edges: dict,
                     enable_orthogonal_corrections: bool = True):
    """
    builds roof graph from vertices and edges
    :param list img_shape: img dimensions
    :param dict vertices: list of vertices
    :param dict undirected_edges: list of undirected edges
    :param bool enable_orthogonal_corrections: tells whether orthogonal corrections need to be enabled or not
    :return tuple containing surface information including 'sub_graphs' and 'undirected edges'
    """
    logger.info("Build Roof Graph started")
    img_w, img_h, _ = img_shape
    min_dim = min(img_w, img_h)

    logger.info("Calling fuse junction")
    vertices, undirected_edges = fuse_junctions_and_create_graph(vertices, undirected_edges, epsilon=min_dim * 4 / 800)

    logger.info("Calling fuse junction with edges")
    # Required, make strict threshold
    vertices, undirected_edges = fuse_junctions_with_edges_and_create_graph(vertices, undirected_edges,
                                                                            epsilon=min_dim * 5 / 800)
    logger.info("Remove intersecting lines by adding new vertex at intersection")
    vertices, undirected_edges = remove_intersecting_lines(vertices, undirected_edges)

    logger.info("Calling remove overlapping lines")
    # Required, Modification needed based on confidence score
    vertices, undirected_edges = remove_overlapping_lines(vertices, undirected_edges, angle_tolerance_in_degrees=5.0)

    logger.info("Calling create adjacency list")
    super_adjacency_list, edge_meta_list = create_adjacency_list_from_undirected_edges(undirected_edges,
                                                                                       preserve_meta=True)
    logger.info("Calling identify disjoint graphs")
    sub_graphs = identify_disjoint_graphs(super_adjacency_list, vertices)

    for graph in sub_graphs:
        find_roof_surfaces(graph)

    logger.info("Calling identify disjoint sub graphs")
    sub_graphs = identify_disjoint_sub_graphs(sub_graphs)

    sub_graphs = [get_central_roof_graph(sub_graphs, (img_w, img_h))]

    logger.info("Calling remove repeated edges")
    for graph in sub_graphs:
        remove_repeated_edges(graph)
        # identify_azimuth_line(graph)

    logger.info("Calling get undirected edges from sub graphs")
    undirected_edges = get_undirected_edges_from_sub_graphs(sub_graphs, edge_meta_list)

    if enable_orthogonal_corrections:
        logger.info("Calling orthogonal corrections")
        for graph in sub_graphs:
            _, _, _ = orthogonal_corrections(graph)
            roof_symmetry = EstimateRoofSymmetry(graph)
            graph.set_undirected_edges_grouped_by_geometry(roof_symmetry.get_undirected_edges_grouped_by_geometry())

    logger.info("Build Roof Graph end")
    return sub_graphs, undirected_edges
