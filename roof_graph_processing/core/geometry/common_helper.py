import math
import numpy as np
from random import randint

from ...core.roof import Vertex


def euclidean_distance(v1: Vertex, v2: Vertex):
    return math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)


def negation_predicate(ele: bool):
    return not ele


def splice(arr, index, num_elements):
    return arr[:index] + arr[index + num_elements:]


def set_intersection(set1: set, set2: set):  # SG
    return set1.intersection(set2)


def are_sets_equal(set1: set, set2: set):  # SG
    return set1 == set2


def flatten_list(lst):
    flat_list = []
    for ele in lst:
        if not isinstance(ele, list):
            flat_list.append(ele)
        else:
            for ele_in_ele in ele:
                flat_list.append(ele_in_ele)

    return flat_list


def find_min_index(arr):  # SG
    # Find index of min value in an array
    min_value = min(arr)
    return arr.index(min_value)


def find_max_index(arr):  # SG
    # Find index of max value in an array
    max_value = max(arr)
    return arr.index(max_value)


def mod(x, n):  # SG
    # Calculate mod between two numbers
    return ((x % n) + n) % n


def first_index_of_satisfied_predicate(li: list, predicate):  # SG
    for idx in range(len(li)):
        if predicate(li[idx]):
            return idx

    return -1


def check_if_vertices_are_same(v1: Vertex, v2: Vertex):
    if v1.vertex_id == v2.vertex_id:
        return True
    return False


def apply_mask(image, mask, color, alpha=0.5):  # Borrowed from Matterport MRCNN
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
        image[image > 255] = 255

    return image


def generate_random_rgb():
    r = randint(128, 255)
    g = randint(64, 192)
    b = randint(0, 128)
    return r, g, b


def find_angle_btw_vertices(v1: Vertex, anchor_v: Vertex, v2: Vertex) -> float:
    """
    This method finds the angle subtended at anchor point

    :param Vertex v1: vertex connected to anchor point
    :param Vertex anchor_v:  anchor vertex
    :param Vertex v2: other vertex connected to anchor point
    :return float: angle subtended at anchor point
    """
    vec_1 = [(v1.x - anchor_v.x), (v1.y - anchor_v.y)]
    vec_2 = [(v2.x - anchor_v.x), (v2.y - anchor_v.y)]
    mod_vec1 = np.linalg.norm(vec_1) if np.linalg.norm(vec_1) != 0 else 1e-8
    mod_vec2 = np.linalg.norm(vec_2) if np.linalg.norm(vec_2) != 0 else 1e-8
    t_value = np.dot(vec_1, vec_2) / (mod_vec1 * mod_vec2)
    t_value = min(max(-1, t_value), 1)
    angle = math.acos(t_value)
    val = (angle * 180) / np.pi
    return round(val, 4)
