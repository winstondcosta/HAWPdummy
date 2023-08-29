import uuid
import math
import numpy as np
from shapely.geometry import Polygon


class Vertex(object):
    def __init__(self, x, y, z=0, meta={}):
        self.x = x
        self.y = y
        self.z = z
        self.vertex_id = str(uuid.uuid4())
        self.meta = meta


class Edge(object):
    def __init__(self, start: Vertex, end: Vertex, meta={}):
        self.start = start  # Start vertex
        self.end = end  # End vertex
        self.edge_id = self.start.vertex_id + "-" + self.end.vertex_id
        self.edge_type = None
        self.meta = meta

    def edge_length(self):
        return math.sqrt((self.start.x - self.end.x)**2 + (self.start.y - self.end.y)**2)

    def edge_angle(self):
        return math.atan((self.end.y - self.start.y) / (self.end.x - self.start.x + 1e-8)) * 180 / np.pi


class Surface(object):
    def __init__(self, ordered_vertices: list, meta={}, order_anticlockwise=True):
        # List of objects of class Vertex ordered in anticlockwise direction to form a close loop
        self.ordered_vertices = ordered_vertices
        self.meta = meta
        self.order_anticlockwise = order_anticlockwise
        self.azimuth_in_degrees = 0
        self.pitch_in_degrees = 0
        self.azimuth_line = None
        self.obstructions = []
        self.parent_surface = None  # If this surface is fully engulfed by another surface

        if order_anticlockwise:
            self.ensure_vertex_ordering_is_counter_clockwise()

        self.surface_id = str(uuid.uuid4())
        self.edge_list = []

        self.shapely_polygon = None

        self.create_edges_from_ordered_vertices()

    def ensure_vertex_ordering_is_counter_clockwise(self):
        num_vertices = len(self.ordered_vertices)

        edge_sum = 0

        for v_id in range(num_vertices):
            start_vertex = self.ordered_vertices[v_id]
            end_vertex = self.ordered_vertices[(v_id + 1) % num_vertices]

            edge_sum += (end_vertex.x - start_vertex.x) * (end_vertex.y + start_vertex.y)

        if edge_sum < 0:
            # Clockwise, make it anticlockwise
            self.ordered_vertices = self.ordered_vertices[::-1]

    def create_edges_from_ordered_vertices(self):
        num_vertices = len(self.ordered_vertices)

        for v_id in range(num_vertices):
            start_vertex = self.ordered_vertices[v_id]
            end_vertex = self.ordered_vertices[(v_id + 1) % num_vertices]
            edge = Edge(start_vertex, end_vertex)
            self.edge_list.append(edge)

    def convert_surface_to_shapely_polygon(self):

        if self.shapely_polygon is not None:
            return self.shapely_polygon

        ordered_points = []
        for vertex in self.ordered_vertices:
            ordered_points.append((vertex.x, vertex.y))

        self.shapely_polygon = Polygon(ordered_points)

        return self.shapely_polygon

    def contains(self, surface):
        self_ = self.convert_surface_to_shapely_polygon()
        surface_ = surface.convert_surface_to_shapely_polygon()

        return self_.contains(surface_)


class UndirectedEdge(object):
    def __init__(self, v1: Vertex, v2: Vertex, meta={}):
        self.v1 = v1
        self.v2 = v2
        self.edge_id = self.combine_uuids()
        self.meta = meta

    def edge_length(self):
        return math.sqrt((self.v1.x - self.v2.x)**2 + (self.v1.y - self.v2.y)**2)

    def edge_angle(self):
        return math.atan((self.v2.y - self.v1.y) / (self.v2.x - self.v1.x + 1e-8)) * 180 / np.pi

    def combine_uuids(self):
        li = sorted([self.v1.vertex_id, self.v2.vertex_id])
        comb_uuid = ""
        for uuid_str in li:
            comb_uuid = comb_uuid + uuid_str + "-"

        return comb_uuid[:-1]


class Graph(object):
    def __init__(self, adjacency_list, vertices):
        self.adjacency_list = adjacency_list  # Dictionary of connections
        self.vertices = vertices  # Dictionary of vertex_id to Vertex object
        self.graph_id = str(uuid.uuid4())
        self.surface_list = []  # list of objects of class Surface

        self.outer_contour = Surface([])  # Surface, encompassing the entire rooftop

        self.roof_surface_undirected_edges = {}  # Stores all the edges that form a closed loop
        self.dangling_undirected_edges = {}  # Instead of removing the dangling edges, we are maintaining a list
        self.dangling_vertices = {}  # Will keep a list of singleton_vertices that form dangling edges
        self.remove_self_directing_edges()
        self.remove_dangling_edges()

        self.vertex_id_list = []  # List of vertex_id in the graph
        self.vertex_list = []  # List of Vertex objects that are part of the graph
        self.vertex_id_to_index = {}  # Map from vertex_id(uuid) to 0-based indexed vertex num
        self.simplified_adjacency_list = {}  # Simplified adjacency list with all vertices having 0-based indexing

        self.simplify_adjacency_list()

        self.num_vertices = len(self.vertex_id_list)
        self.connectivity_matrix = None  # Will be a list of lists depicting vertex connections

        self.create_connectivity_matrix_from_adjacency_list()

        self.geometry_reinforced_edges = []

        self.edges_grouped_by_geometry = []

    def set_undirected_edges_grouped_by_geometry(self, edges_grouped_by_geometry):
        self.edges_grouped_by_geometry = edges_grouped_by_geometry

    def get_undirected_edges_grouped_by_geometry(self):
        return self.edges_grouped_by_geometry

    def check_for_singleton_vertices(self, temp_adjacency_list):
        for vertex_id in temp_adjacency_list:
            if len(temp_adjacency_list[vertex_id]) == 1:
                return vertex_id

        return None

    def remove_self_directing_edges(self):
        new_adjacency_list = {}
        new_vertices = {}

        for vertex_id in self.adjacency_list:
            adj_vertex_id_list = []
            for adj_vertex_id in self.adjacency_list[vertex_id]:
                if adj_vertex_id == vertex_id:
                    continue
                if adj_vertex_id in adj_vertex_id_list:
                    # Duplicate connections are not acceptable
                    continue
                adj_vertex_id_list.append(adj_vertex_id)

            # Those vertices which only has self directing edges, delete
            if len(adj_vertex_id_list) > 0:
                new_adjacency_list[vertex_id] = adj_vertex_id_list

        self.adjacency_list = new_adjacency_list.copy()

        del new_adjacency_list

        for vertex_id in self.adjacency_list:
            new_vertices[vertex_id] = self.vertices[vertex_id]

        self.vertices = new_vertices.copy()

        del new_vertices

    def remove_dangling_edges(self):
        # Remove all edges that are not part of some loop
        new_adjacency_list = self.adjacency_list
        new_vertices = self.vertices

        while 1:
            singleton_vertex_id = self.check_for_singleton_vertices(new_adjacency_list)

            if singleton_vertex_id is None:
                break

            # Remove only connection to the singleton vertex
            only_adjacent_vertex_id = new_adjacency_list[singleton_vertex_id][0]

            singleton_vertex = self.vertices[singleton_vertex_id]
            only_adjacent_vertex = self.vertices[only_adjacent_vertex_id]

            if singleton_vertex.vertex_id not in self.dangling_vertices:
                self.dangling_vertices[singleton_vertex.vertex_id] = singleton_vertex

            if only_adjacent_vertex.vertex_id not in self.dangling_vertices:
                self.dangling_vertices[only_adjacent_vertex.vertex_id] = only_adjacent_vertex

            dangling_ue = UndirectedEdge(singleton_vertex, only_adjacent_vertex)
            self.dangling_undirected_edges[dangling_ue.edge_id] = dangling_ue

            new_adjacency_list[only_adjacent_vertex_id].remove(singleton_vertex_id)

            # Remove singleton vertex
            del new_adjacency_list[singleton_vertex_id]
            del new_vertices[singleton_vertex_id]

        self.adjacency_list = new_adjacency_list.copy()
        del new_adjacency_list

        self.vertices = new_vertices.copy()
        del new_vertices

    def simplify_adjacency_list(self):
        self.vertex_id_list = list(self.adjacency_list.keys())

        for v_id in range(len(self.vertex_id_list)):
            vertex_id = self.vertex_id_list[v_id]
            self.vertex_id_to_index[vertex_id] = v_id

        for vertex_id in self.adjacency_list:
            self.vertex_list.append(self.vertices[vertex_id])  # Add the Vertex object
            self.simplified_adjacency_list[self.vertex_id_to_index[vertex_id]] = []

            for adjacent_vertex_id in self.adjacency_list[vertex_id]:
                if vertex_id == adjacent_vertex_id:
                    # Self edges are not allowed
                    continue
                self.simplified_adjacency_list[self.vertex_id_to_index[vertex_id]].\
                    append(self.vertex_id_to_index[adjacent_vertex_id])

    def create_connectivity_matrix_from_adjacency_list(self):

        row = [0] * self.num_vertices
        self.connectivity_matrix = [row.copy() for _ in range(self.num_vertices)]

        for v_idx in self.simplified_adjacency_list:
            for adj_v_idx in self.simplified_adjacency_list[v_idx]:
                if v_idx == adj_v_idx:
                    # Self edges avoided
                    continue
                self.connectivity_matrix[v_idx][adj_v_idx] = 1
