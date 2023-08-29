import copy
import os
import json

import numpy as np
import pytest
from roof_graph_processing import roof_post_processing as rpp
from roof_graph_processing.roof_graph_processing import threshold, build_roof_graph
from roof_graph_processing.core.geometry.surface_helper import get_longest_edge

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

orthogonal_corrections_modified_vertices_data = {'1214946_36.150707_-115.072501': {2: [375.9999999896721, 501.0],
                                                                                   4: [400.0207299743645,
                                                                                       545.0671273329709],
                                                                                   29: [473.0400471924602,
                                                                                        195.68434258949128],
                                                                                   35: [387.9999992962964, 284.0],
                                                                                   42: [245.15624920325823,
                                                                                        260.0312499505948]}}


def test_orthogonal_corrections():
    json_path = "./roof_graph_processing/fixtures/orthogonal_corrections/test_orthogonal_corrections.json"
    json_data = json.load(open(json_path))

    for predict in json_data:
        img_name = predict['filename']
        vertices = predict['juncs_pred']
        vertex_scores = predict['juncs_score']
        edges_before_thresh_vals = predict['lines_pred']
        edge_scores_before_thresh = predict['lines_score']
        vertices_dict, edges_before_thresh = {}, []
        for i, (vx, vy) in enumerate(vertices):
            vertices_dict[(vx, vy)] = i
        for v1x, v1y, v2x, v2y in edges_before_thresh_vals:
            edges_before_thresh.append([vertices_dict[(v1x, v1y)], vertices_dict[(v2x, v2y)]])

        edges, edge_scores = threshold(edges_before_thresh, edge_scores_before_thresh, edge_score_thresh=0.75)

        edge_list = []
        vertex_meta_list = []
        edge_meta_list = []
        edge_vertex_id_list = []

        for edge, edge_score in zip(edges, edge_scores):
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            v1_score, v2_score = vertex_scores[edge[0]], vertex_scores[edge[1]]

            if (v1 + v2) not in edge_list and (v2 + v1) not in edge_list:
                v1_meta, v2_meta = {"pred_score": v1_score}, {"pred_score": v2_score}
                vertex_meta_list.append((v1_meta, v2_meta))
                edge_list.append(v1 + v2)
                edge_vertex_id_list.append(edge)
                edge_meta = {"pred_score": edge_score}
                edge_meta_list.append(edge_meta)

        vertices, undirected_edges = rpp.create_edges_and_vertices_from_edge_list(edge_list,
                                                                                  vertex_meta_list,
                                                                                  edge_meta_list)
        orig_vertices = copy.deepcopy(vertices)
        orig_undirected_edges = copy.deepcopy(undirected_edges)
        image_shape = [predict['width'], predict['height'], 3]

        try:
            sub_graphs, t_undirected_edges = build_roof_graph(image_shape, orig_vertices, orig_undirected_edges,
                                                              enable_orthogonal_corrections=False)
            orig_central_roof_graph = sub_graphs[0]  # graph without orthogonal corrections

        except Exception as e:
            pytest.fail(f"Exception at STEP1: build_roof_graph Image {img_name} failed: {e})")

        try:
            sub_graphs, t_undirected_edges = build_roof_graph(image_shape, vertices, undirected_edges)
            central_roof_graph = sub_graphs[0]  # graph with orthogonal corrections

        except Exception as e:
            pytest.fail(f"Exception at STEP2: build_roof_graph Image {img_name} failed: {e})")

        # verify the modified vertices after orthogonal corrections
        orig_outer_contour_vertices = orig_central_roof_graph.outer_contour.ordered_vertices
        modified_outer_contour_vertices = central_roof_graph.outer_contour.ordered_vertices
        longest_edge = get_longest_edge(orig_central_roof_graph.outer_contour)
        longest_edge_start_vertex_id = longest_edge.start.vertex_id  # we move in anti-clock-wise direction
        longest_edge_start_vertex_idx = -1
        for i, t_vertex in enumerate(orig_outer_contour_vertices):
            if t_vertex.vertex_id == longest_edge_start_vertex_id:
                longest_edge_start_vertex_idx = i
                break
        orig_outer_contour_vertices = orig_outer_contour_vertices[longest_edge_start_vertex_idx:] + \
                                        orig_outer_contour_vertices[:longest_edge_start_vertex_idx]
        modified_outer_contour_vertices = modified_outer_contour_vertices[longest_edge_start_vertex_idx:] + \
                                            modified_outer_contour_vertices[:longest_edge_start_vertex_idx]

        vertices_count_whose_pos_modified = 0
        for i, t_v1 in enumerate(orig_outer_contour_vertices):
            t_v2 = modified_outer_contour_vertices[i]
            if t_v1.x != t_v2.x or t_v1.y != t_v2.y:
                vertices_count_whose_pos_modified += 1
        assert len(orthogonal_corrections_modified_vertices_data[img_name[:-4]]) == vertices_count_whose_pos_modified

        for t_key, t_values in orthogonal_corrections_modified_vertices_data[img_name[:-4]].items():
            assert np.allclose([modified_outer_contour_vertices[t_key].x, modified_outer_contour_vertices[t_key].y],
                               t_values, rtol=0, atol=1e-12)
