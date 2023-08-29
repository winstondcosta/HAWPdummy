import json
import os
import shutil
import numpy as np

from roof_graph_processing.core.roof import Vertex, UndirectedEdge


def create_edges_and_vertices_from_region_dict(regions: list):
    # Create vertex list
    vertices = {}  # Vertex list
    undirected_edges = {}  # Primary edge list

    for region in regions:
        shape_attributes = region['shape_attributes']
        total_points = len(shape_attributes['all_points_x'])

        assert len(shape_attributes['all_points_x']) == len(shape_attributes['all_points_y']), \
            "ERROR: Number of x coordinates and y coordinates do not match"

        assert total_points >= 2, "ERROR: Cannot form an edge with less two points"

        i = 0
        x1, y1 = shape_attributes['all_points_x'][i], shape_attributes['all_points_y'][i]
        v1 = Vertex(x1, y1)  # Starting vertex
        start_vertex = v1

        if shape_attributes["name"] == "polyline":
            total_points = total_points - 1

        while i < total_points:
            # Complete the loop by keeping last vertex as initial vertex
            if i + 1 != total_points:
                x2 = shape_attributes['all_points_x'][(i + 1) % total_points]
                y2 = shape_attributes['all_points_y'][(i + 1) % total_points]
                v2 = Vertex(x2, y2)
            else:
                v2 = start_vertex

            vertices[v1.vertex_id] = v1
            vertices[v2.vertex_id] = v2

            ue = UndirectedEdge(v1, v2)

            undirected_edges[ue.edge_id] = ue

            i += 1
            v1 = v2

    return vertices, undirected_edges


def create_edges_and_vertices_from_edge_list(edge_list, vertex_meta_list=[], edge_meta_list=[]):
    # Create vertex list
    vertices = {}  # Vertex list
    undirected_edges = {}  # Primary edge list

    num_edges, num_vertices_meta, num_edge_meta = len(edge_list), len(vertex_meta_list), len(edge_meta_list)

    if num_edge_meta > num_edges:
        edge_meta_list = edge_meta_list[:num_edge_meta]

    else:
        for i in range(num_edges - num_edge_meta):
            edge_meta_list.append({})

    if num_vertices_meta > num_edges:
        vertex_meta_list = vertex_meta_list[:num_vertices_meta]

    else:
        for i in range(num_edges - num_vertices_meta):
            vertex_meta_list.append({})

    for edge, vertex_meta_tuple, edge_meta in zip(edge_list, vertex_meta_list, edge_meta_list):
        v1_x, v1_y = edge[0], edge[1]
        v2_x, v2_y = edge[2], edge[3]

        v1_meta, v2_meta = vertex_meta_tuple

        v1 = Vertex(v1_x, v1_y)
        v1.meta = v1_meta
        v2 = Vertex(v2_x, v2_y)
        v2.meta = v2_meta

        vertices[v1.vertex_id] = v1
        vertices[v2.vertex_id] = v2

        ue = UndirectedEdge(v1, v2)
        ue.meta = edge_meta

        undirected_edges[ue.edge_id] = ue

    return vertices, undirected_edges


def create_vertices_list(x: list, y: list, meta_list=[]):
    vertices = {}

    if len(meta_list) == 0:
        meta_list = [{} for _ in x]

    for x_coord, y_coord, meta in zip(x, y, meta_list):
        vertex = Vertex(int(x_coord * 4), int(y_coord * 4))
        vertex.meta = meta

        vertices[vertex.vertex_id] = vertex

    return vertices


def get_x_y_list_from_vertices_list(vertices):
    x = []
    y = []

    for vertex_id in vertices:
        vertex = vertices[vertex_id]

        x.append(np.float(vertex.x / 4))
        y.append(np.float(vertex.y / 4))

    x = np.array(x)
    y = np.array(y)

    return x, y
