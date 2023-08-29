from parsing.config.paths_catalog import DatasetCatalog
from parsing.utils.metric_evaluation import TPFP, AP
import argparse
import os
import os.path as osp
from termcolor import colored
import numpy as np
import json
import matplotlib.pyplot as plt

from roof_graph_processing import site_dependent_processing as sdp
from roof_graph_processing.roof_graph_processing import threshold, build_roof_graph
from roof_graph_processing.post_processing.outer_contour_post_processing import \
    orthogonal_corrections
from roof_graph_processing.post_processing.geometry_check import EstimateRoofSymmetry
import cv2
from tqdm import tqdm

AVAILABLE_DATASETS = ('wireframe_test', 'york_test')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Structural AP Evaluation')
    argparser.add_argument('--path',dest='path',type=str,required=True)
    argparser.add_argument('-t','--threshold', dest='threshold', type=float, default=10.0)
    argparser.add_argument('--rgp', default=False, action='store_true')

    args = argparser.parse_args()

    result_path = args.path

    assert result_path.endswith('.json'), \
        'The result file has to end with .json'

    dataset_name = osp.basename(result_path).rstrip('.json')
    
    assert dataset_name in AVAILABLE_DATASETS, \
        'Currently, we only support  {} datasets for evaluation'.format(
            colored(str(AVAILABLE_DATASETS),'red')
        )
    ann_file = DatasetCatalog.get(dataset_name)['args']['ann_file']
    
    with open(ann_file,'r') as _ann:
        annotations_list = json.load(_ann)
    
    annotations_dict = {
        ann['filename']: ann for ann in annotations_list
    }
    file = open("image_result_analsyis_rgp.csv", "w")
    issues = 0
    # Starting post processing on results
    if args.rgp:
        json_data = json.load(open(result_path))
        for fun, predict in tqdm(enumerate(json_data)):
            if fun != 198:
                continue
            imgname = predict['filename']
            print(imgname)
            if imgname in ['1328440_37.761495_-122.190014.png']:
                continue

            img = cv2.imread("./data/wireframe/images/" + imgname)
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

            vertices, undirected_edges = sdp.create_edges_and_vertices_from_edge_list(edge_list,
                                                                                      vertex_meta_list,
                                                                                      edge_meta_list)

            try:
                sub_graphs, undirected_edges = build_roof_graph(img.shape,
                                                                vertices,
                                                                undirected_edges)
            except Exception as e:
                print("Image " + str(imgname) + " " + str(e))
                issues += 1
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
        result_path = "wireframe_test_rgp.json"
        with open(result_path, 'w') as f:
            json.dump(json_data, f)
        print("Result file path changed to " + result_path)
        print("Total processing issue files, " + str(issues))
    # Ending post processing

    with open(result_path,'r') as _res:
        result_list = json.load(_res)

    for threshold in range(5, 6):
        tp_list, fp_list, scores_list = [],[],[]
        n_gt, precision, f1_score = 0, [], []
        for res in result_list:
            filename = res['filename']
            gt = annotations_dict[filename]
            lines_pred = np.array(res['lines_pred'],dtype=np.float32)
            scores = np.array(res['lines_score'],dtype=np.float32)
            lines_pred = lines_pred[scores >= threshold / 10.0]
            scores = scores[scores >= threshold / 10.0]
            sort_idx = np.argsort(-scores)

            lines_pred = lines_pred[sort_idx]
            scores = scores[sort_idx]
            # import pdb; pdb.set_trace()
            lines_pred[:,0] *= 128/float(res['width'])
            lines_pred[:,1] *= 128/float(res['height'])
            lines_pred[:,2] *= 128/float(res['width'])
            lines_pred[:,3] *= 128/float(res['height'])

            lines_gt   = np.array(gt['lines'],dtype=np.float32)
            lines_gt[:,0]  *= 128/float(gt['width'])
            lines_gt[:,1]  *= 128/float(gt['height'])
            lines_gt[:,2]  *= 128/float(gt['width'])
            lines_gt[:,3]  *= 128/float(gt['height'])

            tp, fp = TPFP(lines_pred,lines_gt,args.threshold)

            n_gt += lines_gt.shape[0]
            tp_list.append(tp)
            fp_list.append(fp)
            scores_list.append(scores)

            idx = np.argsort(scores)[::-1]
            tp = np.cumsum(np.array(tp)[idx]) / lines_gt.shape[0]
            fp = np.cumsum(np.array(fp)[idx]) / lines_gt.shape[0]
            sAP_per_image = AP(tp, fp) * 100
            precision.append(sAP_per_image)
            p = (tp[-1])/np.maximum(tp[-1]+fp[-1],1e-9) if len(tp) > 0 else 0
            r = tp[-1] if len(tp) > 0 else 0
            f1score = (2 * 100 * p * r) / (p + r + 1e-9)
            f1_score.append(f1score)
            file.write(f"{filename}, {sAP_per_image}, {p * 100}, {r * 100}, {f1score}\n")

        precision = np.array(precision)
        file.write("\n\n\n")
        first, second = "", ""
        for val in range(0, 100, 10):
            first += f"{val} - {val + 10}, "
            second += f"{len(np.where((precision >= val) & (precision < val + 10))[0])}, "
        file.write(f"{first}\n{second}\n")

        tp_list = np.concatenate(tp_list)
        fp_list = np.concatenate(fp_list)
        scores_list = np.concatenate(scores_list)
        idx = np.argsort(scores_list)[::-1]
        tp = np.cumsum(tp_list[idx])/n_gt
        fp = np.cumsum(fp_list[idx])/n_gt
        rcs = tp
        pcs = tp/np.maximum(tp+fp,1e-9)
        sAP = AP(tp,fp)*100
        sAP_new, f1 = np.mean(precision), np.mean(f1_score)
        sAP_string = 'sAP{} = New : {:.1f} Old : {:.1f} F1 : {:.1f} F1(sAP, F1) : {:.1f}'.format(args.threshold, sAP_new, sAP, f1, (2 * sAP_new * f1) / (f1 + sAP_new))
        print(sAP_string)