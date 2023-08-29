"""
These set of modules will be used to check if the generated graph
has the symmetry of a rooftop
"""

from roof_graph_processing.core.roof import UndirectedEdge, Edge, Graph, Surface


class EstimateEdgeSymmetry(object):
    def __init__(self, all_edges, angle_tolerance_in_degrees=1.0, length_tolerance_in_pixels=0.0):
        self.__angle_tolerance_in_degrees = angle_tolerance_in_degrees
        self.__length_tolerance_in_pixels = length_tolerance_in_pixels
        self.__all_edges = all_edges

        # The following is a graph adjacency list where the "nodes" are the graph edges
        # And the connection between them is based on if the graph edges are parallel
        self.__edges_grouped_by_parallel = {}

        # The following is a graph adjacency list where the "nodes" are the graph edges
        # And the connection between them is based on if the graph edges are orthogonal
        self.__edges_grouped_by_orthogonal = {}

        # List of groups of parallel edges
        self.__parallel_edges = []

        # List of pairs of orthogonal edges
        self.__orthogonal_edges = []

        self.__edges_grouped_by_geometry = {}

        self.__symmetry_result = {}

    def __parallel_check(self):
        """
        Group lines that are parallel
        Returns: None
        """

        for edge1_id in self.__all_edges:
            edge1 = self.__all_edges[edge1_id]
            for edge2_id in self.__all_edges:
                edge2 = self.__all_edges[edge2_id]
                if edge1.edge_id == edge2.edge_id:
                    continue

                angle_subtended = abs(edge1.edge_angle() - edge2.edge_angle())

                while angle_subtended > 180:
                    angle_subtended -= 180

                if angle_subtended <= self.__angle_tolerance_in_degrees or \
                        (180 - angle_subtended) <= self.__angle_tolerance_in_degrees:

                    self.__edges_grouped_by_geometry[edge1.edge_id] = True
                    self.__edges_grouped_by_geometry[edge2.edge_id] = True

                    if edge1.edge_id not in self.__edges_grouped_by_parallel:
                        self.__edges_grouped_by_parallel[edge1.edge_id] = []

                    if edge2.edge_id not in self.__edges_grouped_by_parallel:
                        self.__edges_grouped_by_parallel[edge2.edge_id] = []

                    if edge2 not in self.__edges_grouped_by_parallel[edge1.edge_id]:
                        self.__edges_grouped_by_parallel[edge1.edge_id].append(edge2.edge_id)

                    if edge1 not in self.__edges_grouped_by_parallel[edge2.edge_id]:
                        self.__edges_grouped_by_parallel[edge2.edge_id].append(edge1.edge_id)

    def __orthogonal_check(self):
        """
        Group lines that are orthogonal
        Returns: None
        """
        for edge1_id in self.__all_edges:
            edge1 = self.__all_edges[edge1_id]
            for edge2_id in self.__all_edges:
                edge2 = self.__all_edges[edge2_id]
                if edge1.edge_id == edge2.edge_id:
                    continue

                angle_subtended = abs(edge1.edge_angle() - edge2.edge_angle())

                while angle_subtended > 180:
                    angle_subtended -= 180

                if abs(90 - angle_subtended) <= self.__angle_tolerance_in_degrees or \
                        abs(270 - angle_subtended) <= self.__angle_tolerance_in_degrees:

                    self.__edges_grouped_by_geometry[edge1.edge_id] = True
                    self.__edges_grouped_by_geometry[edge2.edge_id] = True

                    if edge1.edge_id not in self.__edges_grouped_by_orthogonal:
                        self.__edges_grouped_by_orthogonal[edge1.edge_id] = []

                    if edge2.edge_id not in self.__edges_grouped_by_orthogonal:
                        self.__edges_grouped_by_orthogonal[edge2.edge_id] = []

                    if edge2 not in self.__edges_grouped_by_orthogonal[edge1.edge_id]:
                        self.__edges_grouped_by_orthogonal[edge1.edge_id].append(edge2.edge_id)

                    if edge1 not in self.__edges_grouped_by_orthogonal[edge2.edge_id]:
                        self.__edges_grouped_by_orthogonal[edge2.edge_id].append(edge1.edge_id)

    def __cluster_parallel_edges(self):
        """
        Using the graph of parallel edges (adjacency list: self.__edges_grouped_by_parallel),
        create a list of groups of parallel edges using breadth first search
        Returns: None
        """

        done = {}

        while len(list(self.__edges_grouped_by_parallel.keys())) > len(list(done.keys())):
            cluster = []
            for edge_id in self.__edges_grouped_by_parallel:
                if edge_id in done:
                    continue

                cluster.append(edge_id)
                done[edge_id] = True

                cur_iter = [edge_id]

                while not cur_iter:  # BFS
                    next_iter = []
                    for parallel_edge_id in cur_iter:
                        if parallel_edge_id in done:
                            continue

                        cluster.append(parallel_edge_id)
                        done[parallel_edge_id] = True

                        next_iter.append(parallel_edge_id)

                    cur_iter = next_iter

            self.__parallel_edges.append(cluster)

    def __list_orthogonal_edge_pairs(self):
        """
        Using the graph of orthogonal edges (adjacency list: self.__edges_grouped_by_orthogonal),
        create a list of pairs of orthogonal edges
        Returns: None
        """

        done = {}

        # Store tuple of orthogonal edges
        for edge_id in self.__edges_grouped_by_orthogonal:
            for orthogonal_edge_id in self.__edges_grouped_by_orthogonal[edge_id]:
                if edge_id == orthogonal_edge_id:
                    print("ERROR: Edge orthogonal to itself!!")
                    continue

                tup = sorted([edge_id, orthogonal_edge_id])
                done[tuple(tup)] = True

        self.__orthogonal_edges = list(done.keys())

    def __run(self):
        self.__parallel_check()
        self.__orthogonal_check()

        self.__cluster_parallel_edges()
        self.__list_orthogonal_edge_pairs()

    def __serialize(self):
        self.__symmetry_result = {"edges_grouped_by_parallel": self.__edges_grouped_by_parallel,
                                  "edges_grouped_by_orthogonal": self.__edges_grouped_by_orthogonal,
                                  "edges_grouped_by_geometry": self.__edges_grouped_by_geometry}

    def get_symmetry_result(self):
        if len(list(self.__edges_grouped_by_geometry.keys())) == 0:
            self.__run()
            self.__serialize()

        return self.__symmetry_result

    def get_parallel_edges(self):
        if not self.__parallel_edges:
            self.__parallel_check()
            self.__cluster_parallel_edges()

        return self.__parallel_edges

    def get_orthogonal_edges(self):
        if not self.__orthogonal_edges:
            self.__orthogonal_check()
            self.__list_orthogonal_edge_pairs()

        return self.__orthogonal_edges


class EstimateSurfaceSymmetry(object):
    def __init__(self, surface: Surface, angle_tolerance_in_degrees=1.0, length_tolerance_in_pixels=0.0):
        self.__surface = surface
        self.__angle_tolerance_in_degrees = angle_tolerance_in_degrees
        self.__length_tolerance_in_pixels = length_tolerance_in_pixels

        self.__edges_grouped_by_geometry = {}

        self.__edges_grouped_by_parallel = {}
        self.__edges_grouped_by_orthogonal = {}
        self.__edges_grouped_by_surface_angles = {}
        self.__edges_grouped_by_surface_lengths = {}

        self.__all_edges = {}

        self.__preprocess()

        self.__symmetry_result = {}

    def __preprocess(self):
        for edge in self.__surface.edge_list:
            v1, v2 = edge.start, edge.end
            ue = UndirectedEdge(v1, v2)
            self.__all_edges[ue.edge_id] = ue

    def __surface_angle_check(self):
        """
        For each surface check edges that subtend equal angles with a common edge
        Returns: None
        """

        num_vertices = len(self.__surface.ordered_vertices)

        if num_vertices <= 2:
            return

        for v_idx in range(len(self.__surface.ordered_vertices)):
            left_v = self.__surface.ordered_vertices[v_idx]
            mid1_v = self.__surface.ordered_vertices[(v_idx + 1) % num_vertices]
            mid2_v = self.__surface.ordered_vertices[(v_idx + 2) % num_vertices]
            right_v = self.__surface.ordered_vertices[(v_idx + 3) % num_vertices]

            left_edge = Edge(left_v, mid1_v)
            mid_edge = Edge(mid1_v, mid2_v)
            right_edge = Edge(mid2_v, right_v)

            left_mid_angle = abs(left_edge.edge_angle() - mid_edge.edge_angle())
            mid_right_angle = abs(mid_edge.edge_angle() - right_edge.edge_angle())

            if abs(left_mid_angle - mid_right_angle) <= self.__angle_tolerance_in_degrees:
                # This means that the left and right angles are equal and hence polygon is symmetric
                # Adding left_edge and right_edge to list of edges

                # Convert left_edge and right_edge to undirected form to append
                left_edge = UndirectedEdge(left_edge.start, left_edge.end)
                right_edge = UndirectedEdge(right_edge.start, right_edge.end)

                self.__edges_grouped_by_geometry[left_edge.edge_id] = True
                self.__edges_grouped_by_geometry[right_edge.edge_id] = True

                if left_edge.edge_id not in self.__edges_grouped_by_surface_angles:
                    self.__edges_grouped_by_surface_angles[left_edge.edge_id] = []

                if right_edge in self.__edges_grouped_by_surface_angles[left_edge.edge_id]:
                    pass
                else:
                    self.__edges_grouped_by_surface_angles[left_edge.edge_id].append(right_edge.edge_id)

                if right_edge.edge_id not in self.__edges_grouped_by_surface_angles:
                    self.__edges_grouped_by_surface_angles[right_edge.edge_id] = []

                if left_edge in self.__edges_grouped_by_surface_angles[right_edge.edge_id]:
                    continue
                self.__edges_grouped_by_surface_angles[right_edge.edge_id].append(left_edge.edge_id)

    def __surface_length_check(self):
        """
        For each surface check if opposite edges are equal (based on a general trend of symmetry)
        Returns: None
        """

        for edge1_id in self.__all_edges:
            edge1 = self.__all_edges[edge1_id]
            for edge2_id in self.__all_edges:
                edge2 = self.__all_edges[edge2_id]
                if edge1.edge_id == edge2.edge_id:
                    continue

                if abs(edge1.edge_length() - edge2.edge_length()) <= self.__length_tolerance_in_pixels:
                    # These 2 edges have equal lengths as per tolerance factors
                    self.__edges_grouped_by_geometry[edge1.edge_id] = True
                    self.__edges_grouped_by_geometry[edge2.edge_id] = True

                    if edge1.edge_id not in self.__edges_grouped_by_surface_lengths:
                        self.__edges_grouped_by_surface_lengths[edge1.edge_id] = []

                    if edge2 not in self.__edges_grouped_by_surface_lengths[edge1.edge_id]:
                        self.__edges_grouped_by_surface_lengths[edge1.edge_id].append(edge2.edge_id)

                    if edge2.edge_id not in self.__edges_grouped_by_surface_lengths:
                        self.__edges_grouped_by_surface_lengths[edge2.edge_id] = []

                    if edge1 not in self.__edges_grouped_by_surface_lengths[edge2.edge_id]:
                        self.__edges_grouped_by_surface_lengths[edge2.edge_id].append(edge1.edge_id)

    def __serialize(self):
        """
        Serialises the obtained data into a dictionary format for later usage
        Returns: None
        """
        surface_symmetry = {"edges_grouped_by_parallel": self.__edges_grouped_by_parallel,
                            "edges_grouped_by_orthogonality": self.__edges_grouped_by_orthogonal,
                            "edges_grouped_by_surface_angles": self.__edges_grouped_by_surface_angles,
                            "edges_grouped_by_surface_lengths": self.__edges_grouped_by_surface_lengths}

        self.__symmetry_result = {"surface_id": self.__surface.surface_id,
                                  "surface_symmetry": surface_symmetry,
                                  "edges_grouped_by_geometry": self.__edges_grouped_by_geometry}

    def __run(self):
        self.__surface_angle_check()
        self.__surface_length_check()

        estimate_edge_symmetry = EstimateEdgeSymmetry(self.__all_edges,
                                                      angle_tolerance_in_degrees=self.__angle_tolerance_in_degrees,
                                                      length_tolerance_in_pixels=self.__length_tolerance_in_pixels)

        edge_symmetry = estimate_edge_symmetry.get_symmetry_result()

        self.__edges_grouped_by_parallel = edge_symmetry["edges_grouped_by_parallel"]
        self.__edges_grouped_by_orthogonal = edge_symmetry["edges_grouped_by_orthogonal"]

        self.__edges_grouped_by_geometry.update(edge_symmetry["edges_grouped_by_geometry"])

        self.__serialize()

    def get_symmetry_result(self):
        if not list(self.__symmetry_result.keys()):
            self.__run()

        return self.__symmetry_result


class EstimateRoofSymmetry(object):
    def __init__(self, graph: Graph, angle_tolerance_in_degrees=1.0, length_tolerance_in_pixels=0.0):
        self.__graph = graph
        self.__angle_tolerance_in_degrees = angle_tolerance_in_degrees
        self.__length_tolerance_in_pixels = length_tolerance_in_pixels

        self.__edges_grouped_by_parallel = {}
        self.__edges_grouped_by_orthogonal = {}

        self.__surface_symmetry = {}

        # To store all the edges present in the graph
        self.__all_edges = {}

        # Union of all groups of edges that are grouped by some geometric constraint
        self.__edges_grouped_by_geometry = {}

        self.__preprocess()

        self.__symmetry_ratio = 0.0

    def __preprocess(self):
        """
        Get all edges present in the graph under one dictionary for ensuing modules
        Returns: None
        """

        for surface in self.__graph.surface_list:
            for edge in surface.edge_list:
                v1, v2 = edge.start, edge.end
                ue = UndirectedEdge(v1, v2)
                self.__all_edges[ue.edge_id] = ue

    def __get_symmetry_ratio_of_roof(self):
        """
        A ratio of:
         number of edges grouped under geometry constraints (all above checks)
         divided by the total number of edges present in the roof graph

        Returns: None
        """

        num_edges_grouped_by_geometry = len(self.__edges_grouped_by_geometry.keys())
        num_edges = len(self.__all_edges.keys())

        if num_edges == 0.0:
            num_edges = 1e-8

        self.__symmetry_ratio = num_edges_grouped_by_geometry / num_edges

    def __run(self):

        estimate_edge_symmetry = EstimateEdgeSymmetry(self.__all_edges,
                                                      angle_tolerance_in_degrees=self.__angle_tolerance_in_degrees,
                                                      length_tolerance_in_pixels=self.__length_tolerance_in_pixels)

        edge_symmetry = estimate_edge_symmetry.get_symmetry_result()

        self.__edges_grouped_by_parallel = edge_symmetry["edges_grouped_by_parallel"]
        self.__edges_grouped_by_orthogonal = edge_symmetry["edges_grouped_by_orthogonal"]

        for surface in self.__graph.surface_list:
            estimate_surface_symmetry = \
                EstimateSurfaceSymmetry(surface,
                                        angle_tolerance_in_degrees=self.__angle_tolerance_in_degrees,
                                        length_tolerance_in_pixels=self.__length_tolerance_in_pixels)

            surface_symmetry = estimate_surface_symmetry.get_symmetry_result()

            self.__edges_grouped_by_geometry.update(surface_symmetry["edges_grouped_by_geometry"])

            self.__surface_symmetry[surface_symmetry["surface_id"]] = surface_symmetry["surface_symmetry"]

        self.__edges_grouped_by_geometry.update(edge_symmetry["edges_grouped_by_geometry"])

        self.__get_symmetry_ratio_of_roof()

    def get_symmetry_ratio(self):
        if len(list(self.__surface_symmetry.keys())) == 0:
            self.__run()

        return self.__symmetry_ratio

    def get_undirected_edges_grouped_by_geometry(self):
        """
        Will return all the edge ids that are part of some geometric grouping
        Returns: list of all edge ids that are part of some geometric grouping

        """
        return list(self.__edges_grouped_by_geometry.keys())
