"""
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

from roof_graph_processing.core.geometry.surface_helper import inflate_polygon
from roof_graph_processing.core.geometry.common_helper import euclidean_distance
from roof_graph_processing.site_dependent_processing import create_edges_and_vertices_from_region_dict
from roof_graph_processing.roof_graph_processing import build_roof_graph

from roof_graph_processing.core.roof import Vertex, Surface

def check_if_polygons_are_same(surface1: Surface, surface2: Surface):
    ordered_vertices1 = sorted([(np.round(v.x), np.round(v.y)) for v in surface1.ordered_vertices])
    ordered_vertices2 = sorted([(np.round(v.x), np.round(v.y)) for v in surface2.ordered_vertices])

    for v_tuple1, v_tuple2 in zip(ordered_vertices1, ordered_vertices2):
        v1 = Vertex(v_tuple1[0], v_tuple1[1])
        v2 = Vertex(v_tuple2[0], v_tuple2[1])

        if euclidean_distance(v1, v2) > 1e-8:
            print(str(v_tuple1) + " and " + str(v_tuple2))
            return False

    return True

def test_inflate_polygon():

    # Test 1 : Deflate the Polygon
    poly1 = Surface([Vertex(0, 0), Vertex(8, 0), Vertex(8, 10), Vertex(0, 10)])

    expected_poly = Surface([Vertex(3, 3), Vertex(3, 7), Vertex(5, 7), Vertex(5, 3)])
    obtained_poly = inflate_polygon(poly1,
                                    inflation_distance=-3)

    assert check_if_polygons_are_same(expected_poly, obtained_poly), "Test Case 1 Failed"


    # Test 2 : Inflate the Polygon
    expected_poly = Surface([Vertex(-3, -3), Vertex(-3, 13, Vertex(11, 13), Vertex(11, -3))])
    obtained_poly = inflate_polygon(poly1,
                                    inflation_distance=3)

    assert check_if_polygons_are_same(expected_poly, obtained_poly), "Test Case 2 Failed"

    # Too much deflation should cause the surface to converge to the centroid

    # Test 3
    poly2 = Surface([Vertex(0, 0), Vertex(1, 0), Vertex(1, 1), Vertex(0, 1)])
    obtained_poly = inflate_polygon(poly2,
                                    inflation_distance=-2)

    expected_poly = Surface([Vertex(0.5, 0.5), Vertex(0.5, 0.5), Vertex(0.5, 0.5), Vertex(0.5, 0.5)])

    assert check_if_polygons_are_same(expected_poly, obtained_poly), "Test Case 3 Failed"

    # Test 4
    poly3 = Surface([Vertex(0, 0), Vertex(8, 0), Vertex(8, 10), Vertex(0, 10)])
    obtained_poly = inflate_polygon(poly3,
                                    inflation_distance=-10)

    expected_poly = Surface([Vertex(4, 5), Vertex(4, 5), Vertex(4, 5), Vertex(4, 5)])

    assert check_if_polygons_are_same(expected_poly, obtained_poly), "Test Case 4 Failed"




    # For visual verifications
    with open('roof_graph_processing/fixtures/core/geometry/surface_helper/test_inflate_polygon.json', 'r') as jsonfile:
        test_cases = json.load(jsonfile)
        jsonfile.close()

    for key in test_cases:
        dic = test_cases[key]
        img = np.uint8(np.zeros((512, 512, 3)) * 255)
        regions = dic['regions']

        vertices, undirected_edges = create_edges_and_vertices_from_region_dict(regions)

        sub_graphs, _ = build_roof_graph(img.shape,
                                      vertices,
                                      undirected_edges)



        for graph in sub_graphs:
            for roof_surface in graph.surface_list:
                # Before
                for p_idx in range(len(roof_surface.ordered_vertices)):
                    start_vertex = roof_surface.ordered_vertices[p_idx]
                    end_vertex = roof_surface.ordered_vertices[(p_idx + 1) % len(roof_surface.ordered_vertices)]

                    img = cv2.line(img.copy(),
                                   (int(start_vertex.x), int(start_vertex.y)),
                                   (int(end_vertex.x), int(end_vertex.y)),
                                   (255, 0, 0),
                                   1)

                new_surface = inflate_polygon(roof_surface,
                                                inflation_distance=-5.0)

                # After
                for p_idx in range(len(new_surface.ordered_vertices)):
                    start_vertex = new_surface.ordered_vertices[p_idx]
                    end_vertex = new_surface.ordered_vertices[(p_idx + 1) % len(new_surface.ordered_vertices)]

                    img = cv2.line(img.copy(),
                                   (int(start_vertex.x), int(start_vertex.y)),
                                   (int(end_vertex.x), int(end_vertex.y)),
                                   (255, 255, 0),
                                   1)



                plt.imsave(dic['filename'], img)

"""
