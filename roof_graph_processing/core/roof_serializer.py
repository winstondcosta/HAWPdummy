from roof_graph_processing.core.roof import Vertex, Surface, Graph
import numpy as np
import math


def serializer_surface(surface: Surface):
    surface_dictionary = {"azimuthInDegrees": surface.azimuth_in_degrees, "pitchInDegrees": surface.pitch_in_degrees}

    ordered_vertices = []
    for p_idx in range(len(surface.ordered_vertices)):
        v = surface.ordered_vertices[p_idx]
        vertex_dictionary = {
            "x": v.x,
            "y": v.y,
            "z": 0.0
        }
        ordered_vertices.append(vertex_dictionary)
    surface_dictionary["orderedPoints"] = ordered_vertices

    return surface_dictionary


def serializer_surfaces(surface_list: Surface([])):
    surfaces_dictionary = {"surfaces": {}}
    for surface in surface_list:
        surface_dictionary = serializer_surface(surface)
        surfaces_dictionary["surfaces"][surface.surface_id] = surface_dictionary

    return surfaces_dictionary
