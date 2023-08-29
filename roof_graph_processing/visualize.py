import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import numpy as np

from roof_graph_processing.core.geometry.common_helper import generate_random_rgb
from roof_graph_processing.core.roof import UndirectedEdge

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def recursive_mkdir(path):
    sub_dirs = path.split('/')
    cur_path = ''
    for direc in sub_dirs:
        if direc == '.' or direc == '':
            continue

        cur_path = cur_path + direc + '/'

        if not os.path.exists(cur_path):
            os.mkdir(cur_path)


def draw_roof_lines(img: np.array,
                    img_name: str,
                    edges: dict,
                    save_dir: str) -> None:
    """
    Draws roof-lines with color coding based on confidence score of the segment as follows
    [0.9, 1.0] - Red    - (255, 0, 0)
    [0.8, 0.9) - Green  - (0, 255, 0)
    [0.7, 0.8) - Blue   - (0, 0, 255)
    [0.6, 0.7) - Black  - (0, 0, 0)
    [0.5, 0.6) - White  - (255, 255, 255)
    [0.0, 0.5) - Yellow - (255, 255, 0)

    :param img: Image on which roof-lines need to plotted
    :param img_name: Image name using which output image after plotting is saved
    :param edges: Dictionary of edges that need to be plotted
    :param save_dir : Directory for saving the files
    """
    recursive_mkdir(save_dir)

    for ue_uuid in edges:
        ue = edges[ue_uuid]
        edge_score = ue.meta["pred_score"] if "pred_score" in ue.meta else 1.0

        if edge_score >= 0.9:
            r, g, b = (255, 0, 0)  # Red
        elif edge_score >= 0.8:
            r, g, b = (0, 255, 0)  # Green
        elif edge_score >= 0.7:
            r, g, b = (0, 0, 255)  # Blue
        elif edge_score >= 0.6:
            r, g, b = (0, 0, 0)  # Black
        elif edge_score >= 0.5:
            r, g, b = (255, 255, 255)  # White
        else:
            r, g, b = (255, 255, 0)  # Yellow

        v1, v2 = ue.v1, ue.v2

        img = cv2.circle(img.copy(),
                         center=(int(v1.x), int(v1.y)),
                         radius=1,
                         color=(255, 0, 255))

        img = cv2.circle(img.copy(),
                         center=(int(v2.x), int(v2.y)),
                         radius=1,
                         color=(255, 0, 255))

        img = cv2.line(img.copy(),
                       (int(v1.x), int(v1.y)),
                       (int(v2.x), int(v2.y)),
                       (r, g, b),
                       thickness=1,
                       lineType=cv2.LINE_AA)

    plt.imsave(save_dir + img_name, img)


def draw_roof_surfaces(img,
                       img_name,
                       sub_graphs,
                       save_dir='roof_graph_post_processed/'):

    save_img_path = save_dir + img_name + '/'

    if os.path.exists(save_img_path):
        shutil.rmtree(save_img_path)

    for graph in sub_graphs:
        roof_surface_mask = np.zeros(img.shape)

        save_path = save_dir + img_name + '/'
        recursive_mkdir(save_path)

        img_cpy = img.copy()

        for roof_surface in graph.surface_list:
            roof_surface_polygon = []
            for p_idx in range(len(roof_surface.ordered_vertices)):
                start_vertex = roof_surface.ordered_vertices[p_idx]
                end_vertex = roof_surface.ordered_vertices[(p_idx + 1) % len(roof_surface.ordered_vertices)]

                roof_surface_polygon.append([start_vertex.x, start_vertex.y])

                img_cpy = cv2.line(img_cpy.copy(), (int(start_vertex.x), int(start_vertex.y)),
                                   (int(end_vertex.x), int(end_vertex.y)), (0, 0, 255), 1)

            roof_surface_polygon_np = np.int32(np.array(roof_surface_polygon).reshape(-1, 1, 2))

            roof_surface_mask = cv2.fillPoly(roof_surface_mask,
                                             pts=[roof_surface_polygon_np],
                                             color=generate_random_rgb())

        outer_contour_pts = []

        for p_idx in range(len(graph.outer_contour.ordered_vertices)):  # Marking the eaves
            start_vertex = graph.outer_contour.ordered_vertices[p_idx]
            end_vertex = graph.outer_contour.ordered_vertices[(p_idx + 1) %
                                                              len(graph.outer_contour.ordered_vertices)]

            outer_contour_pts.append([start_vertex.x, start_vertex.y])

            img_cpy = cv2.line(img_cpy.copy(), (int(start_vertex.x), int(start_vertex.y)),
                               (int(end_vertex.x), int(end_vertex.y)), (0, 255, 255), 1)

        for ue in graph.geometry_reinforced_edges:
            img_cpy = cv2.line(img_cpy.copy(), (int(ue.v1.x), int(ue.v1.y)), (int(ue.v2.x), int(ue.v2.y)),
                               (255, 0, 255), 2, lineType=cv2.LINE_AA)

        for ue_id in graph.dangling_undirected_edges:
            ue = graph.dangling_undirected_edges[ue_id]
            img_cpy = cv2.line(img_cpy.copy(), (int(ue.v1.x), int(ue.v1.y)), (int(ue.v2.x), int(ue.v2.y)), (0, 255, 0),
                               2, lineType=cv2.LINE_AA)

        img_after_rgp = np.uint8(0.6 * img_cpy + 0.4 * roof_surface_mask)
        plt.imsave(save_path + graph.graph_id + '.png', img_after_rgp)


def draw_edges_grouped_by_geometry(img,
                                   img_name,
                                   sub_graphs,
                                   save_dir='grouped_by_geometry/'):
    """
    Will mark all edges that are grouped by some geometric constraint in green
    Those edges that are ungrouped, will be drawn in red
    Args:
        img: The RGB image
        img_name: The filename of the RGB image
        sub_graphs: List of all the graph objects depicting roof graphs
        save_dir: The directory to save the visualization results
    Returns: None
    """

    save_img_path = save_dir + img_name + '/'

    if os.path.exists(save_img_path):
        shutil.rmtree(save_img_path)

    for graph in sub_graphs:
        save_path = save_dir + img_name + '/'
        recursive_mkdir(save_path)
        img_cpy = img.copy()

        undirected_edges_grouped_by_geometry = graph.get_undirected_edges_grouped_by_geometry()

        for roof_surface in graph.surface_list:
            for p_idx in range(len(roof_surface.ordered_vertices)):
                start_vertex = roof_surface.ordered_vertices[p_idx]
                end_vertex = roof_surface.ordered_vertices[(p_idx + 1) % len(roof_surface.ordered_vertices)]

                ue = UndirectedEdge(start_vertex, end_vertex)

                if ue.edge_id in undirected_edges_grouped_by_geometry:
                    # This edge has been grouped under geometry
                    img_cpy = cv2.line(img_cpy.copy(),
                                       (int(start_vertex.x), int(start_vertex.y)),
                                       (int(end_vertex.x), int(end_vertex.y)),
                                       (0, 255, 0),
                                       2,
                                       lineType=cv2.LINE_AA)

                else:
                    # This edge is not grouped under any geometric constraints
                    img_cpy = cv2.line(img_cpy.copy(),
                                       (int(start_vertex.x), int(start_vertex.y)),
                                       (int(end_vertex.x), int(end_vertex.y)),
                                       (255, 0, 0),
                                       2,
                                       lineType=cv2.LINE_AA)

        plt.imsave(save_path + graph.graph_id + '.png', img_cpy)
