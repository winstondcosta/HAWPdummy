#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )

Examples:
    python dataset/wireframe.py /datadir/wireframe data/wireframe

Arguments:
    <src>                Original data directory of Huang's wireframe dataset
    <dst>                Directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import json
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.ndimage import zoom
from tqdm import tqdm
# try:
#     sys.path.append(".")
#     sys.path.append("..")
#     from lcnn.utils import parmap
# except Exception:
#     raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines, data_type = 'training'):
    global results
    row, col, _ = image.shape
    im_rescale = (row, col)
    heatmap_scale = (row, col)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    # jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)
    # joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)
    if(len(lines) == 0):
        return
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    # lines = lines[:, :, ::-1]

    junc = []
    jids = {}

    def jid(jun):
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun + (0,)))
        return len(junc) - 1

    lnid, done = [], set()
    lpos, lneg = [], []
    for v0, v1 in lines:
        val1, val2 = jid(v0), jid(v1)
        if (val1, val2) in done:
            continue
        done.add((val1, val2))
        lnid.append((val1, val2))
        lpos.append([junc[val1], junc[val2]])

        vint0, vint1 = to_int(v0), to_int(v1)
        # jmap[0][vint0] = 1
        # jmap[0][vint1] = 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0), *to_int(v1))
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    # for v in junc:
    #     vint = to_int(v[:2])
    #     joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

    llmap = zoom(lmap, [0.5, 0.5])
    lineset = set([frozenset(l) for l in lnid])
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])
    assert len(lneg) != 0, f"{lineset}, {junc}"
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=np.int)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=np.int32)
    lpos = np.array(lpos, dtype=np.float32)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)
    if(data_type == 'training'):
        values = {
            'junctions': [list(map(lambda x : x.item(), j[:2])) for j in junc],
            'height': row,
            'filename': prefix,
            'width': col,
            'edges_negative': [list(map(int, ln)) for ln in Lneg],
            'edges_positive': [list(map(int, lp)) for lp in Lpos],
        }
    else:
        lpos = lpos[:, :, :2].reshape(-1, 4)
        values = {
            'junc': [list(map(lambda x : x.item(), j[:2])) for j in junc],
            'height': row,
            'filename': prefix,
            'width': col,
            'lines': [list(map(lambda x : x.item(), lp[:4])) for lp in lpos],
        }
    results.append(values)

results = []


def main():
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]
    data_root = "data/wireframe/"
    data_output = "./check_format/"
    global results
    os.makedirs(data_output, exist_ok=True)
    for batch in ["training"]:#, 'valing', 'testing']:
        results = []
        folder = os.path.join(data_root, f"{batch}")
        dataset = {}
        # for file in os.listdir(os.path.join(folder, "annotations")):
        #     if(file.endswith("json")):
        #         dataset.update(json.load(open(os.path.join(folder, "annotations", file), "r")))
        dataset = json.load(open(os.path.join(folder, "annotations", "annotations.json"), "r"))
        def handle(data):
            data = dataset[data]
            im = cv2.imread(os.path.join(data_root, "images", data["filename"]))
            if im is None:
                print(os.path.join(folder, "rgb", data["filename"]))
                exit()
            prefix = ".".join(data["filename"].split(".")[:-1])
            lines, done = [], set()
            for reg in data['regions']:
                xs, ys = reg['shape_attributes']['all_points_x'], reg['shape_attributes']['all_points_y']
                length = len(xs)
                for i in range(length):
                    j = (i + 1) % length
                    x1, y1, x2, y2 = xs[i], ys[i], xs[j], ys[j]
                    if(f'{x1} {y1} {x2} {y2}' in done):
                        continue
                    lines.append([[xs[i], ys[i]], [xs[j], ys[j]]])
                    done.add(f'{x1} {y1} {x2} {y2}')
                    done.add(f'{x2} {y2} {x1} {y1}')

            lines = np.array(lines)
            # lines = np.array(data["lines"]).reshape(-1, 2, 2)
            # os.makedirs(os.path.join(data_output, batch), exist_ok=True)

            lines0 = lines.copy()
            # lines1 = lines.copy()
            # lines1[:, :, 0] = im.shape[1] - lines1[:, :, 0]
            # lines2 = lines.copy()
            # lines2[:, :, 1] = im.shape[0] - lines2[:, :, 1]
            # lines3 = lines.copy()
            # lines3[:, :, 0] = im.shape[1] - lines3[:, :, 0]
            # lines3[:, :, 1] = im.shape[0] - lines3[:, :, 1]

            path = os.path.join(data_output, batch, prefix)
            path = data["filename"]
            save_heatmap(f"{path}", im[::, ::], lines0, data_type = batch)
            # if batch != "valid":
            #     save_heatmap(f"{path}_1", im[::, ::-1], lines1)
            #     save_heatmap(f"{path}_2", im[::-1, ::], lines2)
            #     save_heatmap(f"{path}_3", im[::-1, ::-1], lines3)
            # print("Finishing", os.path.join(data_output, batch, prefix))

        # parmap(handle, dataset, 0)
        keys = list(dataset.keys())
        for key in tqdm(keys):
            handle(key)
        print(len(results), f"for {batch}")
        json.dump(results, open(f'{batch}.json', "w"))


if __name__ == "__main__":
    main()
