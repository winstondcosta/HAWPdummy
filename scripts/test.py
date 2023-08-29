import torch
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_test_dataset
from parsing.detector import get_hawp_model
from parsing.utils.logger import setup_logger
from parsing.utils.checkpoint import DetectronCheckpointer
from parsing.config.paths_catalog import DatasetCatalog
from parsing.utils.metric_evaluation import TPFP, AP
import os
import os.path as osp
import argparse
import logging
import json
import numpy as np
import cv2
import sys
#from roof_graph_processing import site_dependent_processing as sdp
from roof_graph_processing import roof_post_processing as rpp
from roof_graph_processing.roof_graph_processing import threshold, build_roof_graph
from roof_graph_processing.post_processing.outer_contour_post_processing import \
    orthogonal_corrections
from roof_graph_processing.post_processing.geometry_check import EstimateRoofSymmetry
from tqdm import tqdm

AVAILABLE_DATASETS = ('wireframe_test', 'york_test')

parser = argparse.ArgumentParser(description='HAWP Testing')

parser.add_argument("--config-file",
                    metavar="FILE",
                    help="path to config file",
                    type=str,
                    default=None,
                    )
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--display",
                    default=False,
                    action='store_true')
parser.add_argument('-t', '--threshold', dest='threshold', type=float, default=10.0,
                    help="the threshold for sAP evaluation")
parser.add_argument('--output', type=str, default="./results/", help="Folder name for saving images")
parser.add_argument('--rgp', default=False, action='store_true')
parser.add_argument("opts",
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER
                    )
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)


def test(cfg):
    logger = logging.getLogger("hawp.testing")
    device = cfg.MODEL.DEVICE
    model = get_hawp_model(pretrained=False, cfg_det=None, network_type="hawp")
    model = model.to(device)
    test_datasets = build_test_dataset(cfg)

    output_dir = cfg.OUTPUT_DIR
    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(cfg,
                                             model,
                                             save_dir=cfg.OUTPUT_DIR,
                                             save_to_disk=True,
                                             logger=logger)
        _ = checkpointer.load()
        model = model.eval()
    else:
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict['model'])
        model = model.eval()

    for name, dataset in test_datasets:
        results = []
        logger.info('Testing on {} dataset'.format(name))
        for i, (images, annotations) in tqdm(enumerate(dataset)):
            try:
                with torch.no_grad():
                    mask_network, output, extra_info = model(images.to(device), annotations)
                    output = to_device(output, 'cpu')
            except:
                print("Error at image " + str(i) + " " + annotations[0]['filename'])
                continue
            path = "data/wireframe/images/" + annotations[0]['filename']
            lines = output['lines_pred'].numpy()
            scores = output['lines_score'].numpy()
            vertices = output['juncs_pred'].numpy()
            img = cv2.imread(path)
            if ('scale' in annotations[0]):
                (sx, sy), scale = annotations[0]['start'], annotations[0]['scale']
                lines[:, ::2] -= sx
                lines[:, 1::2] -= sy
                lines /= scale
                vertices[:, 0] -= sx
                vertices[:, 1] -= sy
                vertices /= scale
                output['width'], output['height'] = len(img[0]), len(img)

            if args.display:
                def drawit(low, high, design, img):
                    curr = lines[(scores <= high) & (scores > low)]
                    for x1, y1, x2, y2 in curr:
                        x1, y1, x2, y2 = map(round, [x1, y1, x2, y2])
                        img = cv2.line(img, (x1, y1), (x2, y2), design, 1)
                    return img

                thres = list(reversed([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]))
                colors = list(reversed(
                    [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0), (255, 255, 255), (255, 0, 255), (0, 255, 255)]))

                for s, sym in zip(thres, colors):
                    if (s < 0.5): continue
                    img = drawit(s, s + 0.1, sym, img)
                imgname = ".".join(annotations[0]['filename'].split(".")[:-1])
                cv2.imwrite(os.path.join(args.output, imgname + ".png"), img)

            for k in output.keys():
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()
            results.append(output)

        outpath_dataset = osp.join(output_dir, '{}.json'.format(name))
        logger.info('Writing the results of the {} dataset with length {} into {}'.format(name, len(results),
                                                                                          outpath_dataset))
        with open(outpath_dataset, 'w') as _out:
            json.dump(results, _out)

        # Starting post processing on results
        if args.rgp:
            json_data = json.load(open(outpath_dataset))
            for predict in json_data:
                imgname = predict['filename']
                vertices = predict['juncs_pred']
                vertex_scores = predict['juncs_score']
                edges_before_thresh_vals = predict['lines_pred']
                edge_scores_before_thresh = predict['lines_score']
                vertices_dict, edges_before_thresh = {}, []

                for i, (vx, vy) in enumerate(vertices):
                    vertices_dict[(vx, vy)] = i

                for v1x, v1y, v2x, v2y in edges_before_thresh_vals:
                    edges_before_thresh.append([vertices_dict[(v1x, v1y)], vertices_dict[(v2x, v2y)]])

                edges, edge_scores = threshold(edges_before_thresh,
                                               edge_scores_before_thresh)

                edge_list, vertex_meta_list, edge_meta_list = [], [], []
                for edge, edge_score in zip(edges, edge_scores):
                    v1, v2 = vertices[edge[0]], vertices[edge[1]]
                    v1_score, v2_score = vertex_scores[edge[0]], vertex_scores[edge[1]]

                    if (v1 + v2) not in edge_list and (v2 + v1) not in edge_list:
                        v1_meta, v2_meta = {"pred_score": v1_score}, {"pred_score": v2_score}
                        vertex_meta_list.append((v1_meta, v2_meta))
                        edge_list.append(v1 + v2)
                        edge_meta = {"pred_score": edge_score}
                        edge_meta_list.append(edge_meta)

                vertices, undirected_edges = rpp.create_edges_and_vertices_from_edge_list(edge_list,
                                                                                          vertex_meta_list,
                                                                                          edge_meta_list)

                try:
                    sub_graphs, undirected_edges = build_roof_graph(img.shape,
                                                                    vertices,
                                                                    undirected_edges)
                except Exception as e:
                    print("Image " + str(imgname) + " " + str(e))
                    continue

                for graph in sub_graphs:
                    orthogonal_corrections(graph)
                    roof_symmetry = EstimateRoofSymmetry(graph)
                    graph.set_undirected_edges_grouped_by_geometry(
                        roof_symmetry.get_undirected_edges_grouped_by_geometry())
                graph = sub_graphs[0]  # Central house is only returned currently
                vertices, vertices_dict, edges = [], {}, []

                for key, value in graph.vertices.items():
                    vertices_dict[value.vertex_id] = [value.x, value.y]
                    vertices.append([value.x, value.y])
                vertices_done = set()
                for key, value in graph.adjacency_list.items():
                    vertices_done.add(key)
                    for v in value:
                        if v in vertices_done:
                            continue
                        edges.append(vertices_dict[key] + vertices_dict[v])
                predict['juncs_pred'] = vertices
                predict['juncs_score'] = [1.0] * len(vertices)
                predict['lines_pred'] = edges
                predict['lines_score'] = [1.0] * len(edges)
            outpath_dataset = osp.join(output_dir, '{}_rgp.json'.format(name))
            with open(outpath_dataset, 'w') as f:
                json.dump(json_data, f)
            print("Result file path changed to " + outpath_dataset)
        # Ending post processing

        if name not in AVAILABLE_DATASETS:
            continue
        logger.info('evaluating the results on the {} dataset'.format(name))
        ann_file = DatasetCatalog.get(name)['args']['ann_file']
        with open(ann_file, 'r') as _ann:
            annotations_list = json.load(_ann)
        annotations_dict = {
            ann['filename']: ann for ann in annotations_list
        }
        with open(outpath_dataset, 'r') as _res:
            result_list = json.load(_res)

        tp_list, fp_list, scores_list = [], [], []
        n_gt, precision, f1_score = 0, [], []
        f1_score_dict = {}
        for res in result_list:
            filename = res['filename']
            gt = annotations_dict[filename]
            lines_pred = np.array(res['lines_pred'], dtype=np.float32)
            scores = np.array(res['lines_score'], dtype=np.float32)
            lines_pred = lines_pred[scores >= 0.75]
            scores = scores[scores >= 0.75]
            sort_idx = np.argsort(-scores)

            lines_pred = lines_pred[sort_idx]
            scores = scores[sort_idx]
            lines_gt = np.array(gt['lines'], dtype=np.float32)

            # import pdb; pdb.set_trace()
            try:
                lines_pred[:, 0] *= 128 / float(res['width'])
                lines_pred[:, 1] *= 128 / float(res['height'])
                lines_pred[:, 2] *= 128 / float(res['width'])
                lines_pred[:, 3] *= 128 / float(res['height'])

                lines_gt[:, 0] *= 128 / float(gt['width'])
                lines_gt[:, 1] *= 128 / float(gt['height'])
                lines_gt[:, 2] *= 128 / float(gt['width'])
                lines_gt[:, 3] *= 128 / float(gt['height'])
            except:
                print("Issue with file " + filename)
                pass
            tp, fp = TPFP(lines_pred, lines_gt, args.threshold)
            n_gt += lines_gt.shape[0]
            tp_list.append(tp)
            fp_list.append(fp)
            scores_list.append(scores)

            idx = np.argsort(scores)[::-1]
            tp = np.cumsum(np.array(tp)[idx]) / lines_gt.shape[0]
            fp = np.cumsum(np.array(fp)[idx]) / lines_gt.shape[0]
            sAP_per_image = AP(tp, fp) * 100
            precision.append(sAP_per_image)
            p = (tp[-1]) / np.maximum(tp[-1] + fp[-1], 1e-9) if len(tp) > 0 else 0
            r = tp[-1] if len(tp) > 0 else 0
            f1score = (2 * 100 * p * r) / (p + r + 1e-9)
            f1_score.append(f1score)
            f1_score_dict[filename] = f1score

        tp_list = np.concatenate(tp_list)
        fp_list = np.concatenate(fp_list)
        scores_list = np.concatenate(scores_list)
        idx = np.argsort(scores_list)[::-1]
        tp = np.cumsum(tp_list[idx]) / n_gt
        fp = np.cumsum(fp_list[idx]) / n_gt
        rcs = tp
        pcs = tp / np.maximum(tp + fp, 1e-9)
        sAP = AP(tp, fp) * 100
        sAP_new, f1 = np.mean(precision), np.mean(f1_score)
        metric_string = 'sAP{} : {:.1f} F1 : {:.1f} F1(sAP, F1) : {:.1f}'.format(args.threshold, sAP_new, f1,
                                                                        (2 * sAP_new * f1) / (f1 + sAP_new))
        print(metric_string)


if __name__ == "__main__":
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = 'outputs/default'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger('hawp', ".")
    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")
    args.config_file = None
    test(cfg)
