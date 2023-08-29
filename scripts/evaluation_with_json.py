import argparse
import json
import numpy as np
import os
import pandas as pd
from parsing.utils.logger import setup_logger

from roof_graph_processing import roof_post_processing as rpp
from roof_graph_processing.roof_graph_processing import threshold, build_roof_graph


def TPFP(lines_dt, lines_gt, threshold):
    lines_dt = lines_dt.reshape(-1, 2, 2)[:, :, ::-1]
    lines_gt = lines_gt.reshape(-1, 2, 2)[:, :, ::-1]
    diff = ((lines_dt[:, None, :, None] - lines_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(lines_gt), bool)
    tp = np.zeros(len(lines_dt), float)
    fp = np.zeros(len(lines_dt), float)

    for i in range(lines_dt.shape[0]):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def AP(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    return ap


def get_metrics(result_list, gt_path, facet_info_file_path=None, dis_thres=10.0, rgp=False,
                output_csv_path='image_result_analysis.csv'):
    """
            Get metrics for images present in GT json file with list of model outputs and ground truth json

            :param list result_list: List of dicts for each image file in hawp inference or test format
            :param str gt_path: Ground-truth json file with data in hawp format
            :param str facet_info_file_path: Csv file with lines and facet data for each csv file
            :param float dis_thres: Threshold value to considered TP for gt lines
            :param bool rgp: If post-processing need to be applied before getting metrics
            :param str output_csv_path: Csv file to dump site-wise analysis data
    """
    ann_file = gt_path
    output_file = open(output_csv_path, "w")

    # JSON file with GT in hawp test format
    with open(ann_file, 'r') as _ann:
        annotations_list = json.load(_ann)

    annotations_dict = {
        ann['filename']: ann for ann in annotations_list
    }

    if len(result_list) > 0 and 'vertices' in result_list[0]:
        # Converting to test format for uniformity
        for file_dict in result_list:
            file_dict['juncs_pred'] = file_dict.pop('vertices')
            file_dict['juncs_score'] = file_dict.pop('vertices-score')
            file_dict['lines_score'] = file_dict.pop('edges-weights')
            edges_indices, edges = file_dict.pop('edges'), []
            for v1, v2 in edges_indices:
                edges.append(file_dict['juncs_pred'][v1] + file_dict['juncs_pred'][v2])
            file_dict['lines_pred'] = edges
    if os.path.isfile(facet_info_file_path):
        facets = pd.read_csv(facet_info_file_path)
        facets_dict = {row['Filename']: [row['Regions'], row['Lines']] for i, row in facets.iterrows()}
    else:
        facets_dict = {}
    issues = 0
    confidence_value = 0.75
    # Starting post processing on results
    if rgp:
        for fun, predict in enumerate(result_list):
            # logger.info(f"Post processing started for image {predict['filename']}")
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

            edges, edge_scores = threshold(edges_before_thresh, edge_scores_before_thresh, confidence_value)

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

            vertices, undirected_edges = rpp.create_edges_and_vertices_from_edge_list(edge_list, vertex_meta_list,
                                                                                      edge_meta_list)

            try:
                sub_graphs, undirected_edges = build_roof_graph([predict['width'], predict['height'], 3], vertices,
                                                                   undirected_edges)
            except Exception as e:
                print("Image " + str(imgname) + " " + str(e))
                issues += 1
                continue

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
            # logger.info(f"Post processing end for image {predict['filename']}")
        result_path_rgp = "wireframe_test_rgp.json"
        with open(result_path_rgp, 'w') as f:
            json.dump(result_list, f)
        print("Post processed json(test_mode) saved to " + result_path_rgp)
        print("Total processing issue files, " + str(issues))
    # Ending post processing

    try:
        # Have the filtered files txt if only some files need to taken for evaluation
        with open("filtered_files.txt") as f:
            files = set(line.strip() for line in f.readlines())
    except FileNotFoundError as e:
        files = [file_dict['filename'] for file_dict in result_list]
    print("Total testing files", len(files))

    output_file.write(f"Filename,sAP,Precision,Recall,F1score,Facets,Lines\n")
    tp_list, fp_list, scores_list = [], [], []
    n_gt, precision, f1_score = 0, [], []
    f1_score_dict = {}
    for res in result_list:
        filename = res['filename']
        if filename not in files:
            continue
        gt = annotations_dict[filename]
        lines_pred = np.array(res['lines_pred'], dtype=np.float32)
        scores = np.array(res['lines_score'], dtype=np.float32)
        lines_pred = lines_pred[scores >= confidence_value]
        scores = scores[scores >= confidence_value]
        sort_idx = np.argsort(-scores)

        lines_pred = lines_pred[sort_idx]
        scores = scores[sort_idx]
        lines_pred[:, 0] *= 128 / float(res['width'])
        lines_pred[:, 1] *= 128 / float(res['height'])
        lines_pred[:, 2] *= 128 / float(res['width'])
        lines_pred[:, 3] *= 128 / float(res['height'])

        lines_gt = np.array(gt['lines'], dtype=np.float32)
        lines_gt[:, 0] *= 128 / float(gt['width'])
        lines_gt[:, 1] *= 128 / float(gt['height'])
        lines_gt[:, 2] *= 128 / float(gt['width'])
        lines_gt[:, 3] *= 128 / float(gt['height'])

        tp, fp = TPFP(lines_pred, lines_gt, dis_thres)

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
        output_file.write(f"{filename},{sAP_per_image},{p * 100},{r * 100},{f1score},")
        output_file.write(f"{','.join(map(str, facets_dict.get(filename, ['NA', 'NA'])))}\n")

    precision = np.array(precision)
    f1_score = np.array(f1_score)
    output_file.write("\n\n\n")
    if len(facets_dict) > 0:
        first, second = "F1score,", "Images,"
        ranges = [[1, 8], [9, 16], [17, 1000]]
        final_text = ["Low Complexity(1-8),", "Med Complexity(9-16),", "High Complexity(17-),"]
        for val in range(0, 100, 10):
            first += f"{val} -- {val + 10}, "
            val2 = val + 10 if val != 90 else val + 10.1
            files1 = set([each for each, value in f1_score_dict.items() if val <= value < val2])
            second += f"{len(files1)},"

            for idx in range(len(ranges)):
                low, high = ranges[idx]
                files2 = set(facets[(facets['Regions'] <= high) & (facets['Regions'] >= low)]['Filename'])
                final_text[idx] += f"{str(len(files1.intersection(files2)))},"

        output_file.write(f"{first}\n{second}\n")
        output_file.write("\n".join(final_text) + "\n")

        output_file.write(f"\n\n\n")
        first, second, third = "Facets,", "Images,", "Avg F1score,"
        for val in range(1, max(facets['Regions']) + 1, 6):
            first += f"{val} -- {val + 5}, "
            temp_facets = facets[(facets['Regions'] >= val) & (facets['Regions'] < val + 6)]
            second += f"{len(temp_facets)}, "
            selected = np.array([f1_score_dict[f] for f in temp_facets['Filename'] if f in files])
            third += f"{np.sum(selected) / len(selected)}, " if len(selected) > 0 else "NA,"
        output_file.write(f"{first}\n{second}\n{third}\n")

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
    metric_string = 'sAP{} : {:.1f} F1 : {:.1f} F1(sAP, F1) : {:.1f}'.format(dis_thres, sAP_new, f1,
                                                                             (2 * sAP_new * f1) / (f1 + sAP_new))
    print(metric_string)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Structural AP Evaluation')
    argparser.add_argument('--path', dest='path', type=str, required=True)
    argparser.add_argument('--gt_path', type=str, required=True)
    argparser.add_argument('--facet_info_file_path', type=str, required=True)
    argparser.add_argument('-t', '--threshold', dest='threshold', type=float, default=10.0)
    argparser.add_argument('--rgp', default=False, action='store_true')
    argparser.add_argument('--output_csv_path', default='image_result_analysis.csv', type=str)
    args = argparser.parse_args()
    logger = setup_logger('Post-processing', '.')
    results = json.load(open(args.path))
    get_metrics(results, args.gt_path, args.facet_info_file_path, dis_thres=args.threshold, rgp=args.rgp,
                output_csv_path=args.output_csv_path)
