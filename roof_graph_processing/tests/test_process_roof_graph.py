"""
import json
import numpy as np

from roof_graph_processing.core.process_roof_graph import remove_intersecting_lines, \
    identify_disjoint_sub_graphs, \
    remove_overlapping_lines, \
    fuse_junctions_and_create_graph, \
    fuse_junctions_with_edges_and_create_graph

from roof_graph_processing.roof_graph_processing import build_roof_graph


from roof_graph_processing.site_dependent_processing import create_edges_and_vertices_from_region_dict
from roof_graph_processing.visualize import draw_roof_lines, draw_roof_surfaces

def test_remove_intersecting_lines():
    with open('roof_graph_processing/fixtures/intersecting_lines/test_cases.json', 'r') as jsonfile:
        test_cases = json.load(jsonfile)
        jsonfile.close()

    for key in test_cases:
        dic = test_cases[key]
        img = np.uint8(np.zeros((512, 512, 3)) * 255)
        regions = dic['regions']

        vertices, undirected_edges = create_edges_and_vertices_from_region_dict(regions)

        # Before removal of intersecting lines

        draw_roof_lines(img.copy(),
                        dic['filename'],
                        undirected_edges,
                        save_dir='test_results/core/process_roof_graph/remove_intersecting_lines/before/')


        vertices, undirected_edges = remove_intersecting_lines(vertices, undirected_edges)

        # After removal of intersecting lines

        draw_roof_lines(img.copy(),
                        dic['filename'],
                        undirected_edges,
                        save_dir='test_results/core/process_roof_graph/remove_intersecting_lines/after/')


def test_remove_overlapping_lines():
    # Read fixtures and check for all the test cases
    with open('roof_graph_processing/fixtures/remove_overlapping_lines/test_cases.json', 'r') as jsonfile:
        test_cases = json.load(jsonfile)
        jsonfile.close()

    for key in test_cases:
        dic = test_cases[key]
        img = np.uint8(np.zeros((512, 512, 3)) * 255)
        regions = dic['regions']

        vertices, undirected_edges = create_edges_and_vertices_from_region_dict(regions)

        vertices, undirected_edges = remove_intersecting_lines(vertices, undirected_edges)

        vertices, undirected_edges = fuse_junctions_and_create_graph(vertices,
                                                                     undirected_edges,
                                                                     epsilon=512 * 12 / 800)

        vertices, undirected_edges = fuse_junctions_with_edges_and_create_graph(vertices,
                                                                                undirected_edges,
                                                                                epsilon=512 * 12 / 800)

        # Before removal of overlapping lines

        draw_roof_lines(img.copy(),
                        dic['filename'],
                        undirected_edges,
                        save_dir='test_results/core/process_roof_graph/remove_overlapping_lines/before/')

        vertices, undirected_edges = remove_overlapping_lines(vertices,
                                                              undirected_edges,
                                                              angle_tolerance_in_degrees=10.0)

        # After removal of overlapping lines

        draw_roof_lines(img.copy(),
                        dic['filename'],
                        undirected_edges,
                        save_dir='test_results/core/process_roof_graph/remove_overlapping_lines/after/')


def test_identify_disjoint_sub_graphs():
    with open('roof_graph_processing/fixtures/identify_disjoint_sub_graphs/test_identify_disjoint_sub_graphs.json',
              'r') as jsonfile:
        test_cases = json.load(jsonfile)

        for key in test_cases:
            dic = test_cases[key]
            img = np.uint8(np.zeros((512, 512, 3)) * 255)
            regions = dic['regions']

            vertices, undirected_edges = create_edges_and_vertices_from_region_dict(regions)

            sub_graphs, _ = build_roof_graph((512, 512, 3),
                                             vertices,
                                             undirected_edges)


            # Before merge, draw all roof graphs

            # print(len(sub_graphs))

            draw_roof_surfaces(img.copy(),
                               dic['filename'],
                               sub_graphs,
                               save_dir='test_results/disjoint_sub_graphs/before/')

            # After merge, draw all roof graphs

            sub_graphs_after = identify_disjoint_sub_graphs(sub_graphs)

            # print(len(sub_graphs_after))

            draw_roof_surfaces(img.copy(),
                               dic['filename'],
                               sub_graphs_after,
                               save_dir='test_results/disjoint_sub_graphs/after/')


if __name__=="__main__":
    test_remove_intersecting_lines()

"""
