import torch
import random
import numpy as np

from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_train_dataset
from parsing.detector_multi import WireframeDetector
from parsing.solver import make_lr_scheduler, make_optimizer
from parsing.utils.logger import setup_logger
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
from parsing.encoder.hafm import HAFMencoder
import torch.nn.functional as F
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
# from detectron2.utils.events import EventStorage
import os
import time
import datetime
import argparse
import logging
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()



class LossReducer(object):
    def __init__(self, cfg):
        # self.loss_keys = cfg.MODEL.LOSS_WEIGHTS.keys()
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k] * loss_dict[k]
                          for k in self.loss_weights.keys() if k in loss_dict.keys()])

        return total_loss

def proposal_lines(md_maps, dis_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0, height, device=device).float()
        _x = torch.arange(0, width, device=device).float()

        y0, x0 = torch.meshgrid(_y, _x)
        md_ = (md_maps[0] - 0.5) * np.pi * 2
        st_ = md_maps[1] * np.pi / 2
        ed_ = -md_maps[2] * np.pi / 2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        x_standard = torch.ones_like(cs_st)

        y_st = ss_st / cs_st
        y_ed = ss_ed / cs_ed

        x_st_rotated = (cs_md - ss_md * y_st) * dis_maps[0] * scale
        y_st_rotated = (ss_md + cs_md * y_st) * dis_maps[0] * scale

        x_ed_rotated = (cs_md - ss_md * y_ed) * dis_maps[0] * scale
        y_ed_rotated = (ss_md + cs_md * y_ed) * dis_maps[0] * scale

        x_st_final = (x_st_rotated + x0).clamp(min=0, max=width - 1)
        y_st_final = (y_st_rotated + y0).clamp(min=0, max=height - 1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0, max=width - 1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0, max=height - 1)

        lines = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final)).permute((1, 2, 0))

        return lines  # , normals

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk=300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)

    scores, index = torch.topk(jloc, k=topk)
    y = torch.div(index, width, rounding_mode='floor').float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores > th], scores[scores > th]

def pooling(features_per_image, lines_per_im, tspan, n_pts1, dim_loi, pool1d, fc2):
        h, w = features_per_image.size(1), features_per_image.size(2)
        U, V = lines_per_im[:, :2], lines_per_im[:, 2:]
        sampled_points = U[:, :, None] * tspan + V[:, :, None] * (1 - tspan) - 0.5
        sampled_points = sampled_points.permute((0, 2, 1)).reshape(-1, 2)
        px, py = sampled_points[:, 0], sampled_points[:, 1]
        px0 = px.floor().clamp(min=0, max=w - 1)
        py0 = py.floor().clamp(min=0, max=h - 1)
        px1 = (px0 + 1).clamp(min=0, max=w - 1)
        py1 = (py0 + 1).clamp(min=0, max=h - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((features_per_image[:, py0l, px0l] * (py1 - py) * (px1 - px) + features_per_image[:, py1l, px0l] * (
                    py - py0) * (px1 - px) + features_per_image[:, py0l, px1l] * (py1 - py) * (
                           px - px0) + features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(128, -1, 32)
              ).permute(1, 0, 2)

        # if self.pool1d is not None:
        xp = pool1d(xp)
        features_per_line = xp.view(-1, n_pts1 * dim_loi)
        logits = fc2(features_per_line).flatten()
        return logits

def train(cfg):
    logger = logging.getLogger("hawp.trainer")
    device = cfg.MODEL.DEVICE
    n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
    n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
    n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
    n_dyn_othr = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR
    n_dyn_othr2 = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR2
    n_pts1 = cfg.MODEL.PARSING_HEAD.N_PTS1
    dim_loi = cfg.MODEL.PARSING_HEAD.DIM_LOI
    loss = nn.BCEWithLogitsLoss(reduction='none')
    # cfg_det = get_cfg()
    # cfg_det_path = "config-files/train_config.yaml"
    # cfg_det.merge_from_filefile(cfg_det_path)
    #gpus = [int(i) for i in args.gpu.split(',')]
    gpu_ids = [0]
    model = WireframeDetector(cfg, None, network_type="hawp")
    hafm_encoder = HAFMencoder(cfg)
    ## multi-gpu
    if len(gpu_ids)>1:
         model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    #model = model.to(device)
    #model.train()
    # checkpoint = DetectionCheckpointer(model)
    # checkpoint.load(cfg_det.MODEL.WEIGHTS)
    train_dataset = build_train_dataset(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    loss_reducer = LossReducer(cfg)

    arguments = {}
    max_epoch = cfg.SOLVER.MAX_EPOCH
    arguments["max_epoch"] = max_epoch

    checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         optimizer,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)

    _ = checkpointer.load()
    start_training_time = time.time()
    end = time.time()

    ckpt_files = sorted([each for each in os.listdir("./outputs/hawp/") if 'model' in each])
    arguments['epoch'] = 0
    if len(ckpt_files):
        arguments['epoch'] = int(ckpt_files[-1].strip('.pth').strip('model_'))
    start_epoch = arguments['epoch']
    print(f"Starting from epoch {start_epoch}")
    epoch_size = len(train_dataset)

    global_iteration = epoch_size * start_epoch

    for epoch in range(start_epoch + 1, arguments['max_epoch'] + 1):
        meters = MetricLogger(" ")
        model.train()
        arguments['epoch'] = epoch
        for it, (images, annotations) in enumerate(train_dataset):
            #print(it)
            #continue
            data_time = time.time() - end
            images = images.to(device)
            annotations = to_device(annotations, device)
            # with EventStorage() as storage:
            targets, metas = hafm_encoder(annotations)
            loss_dict, loi_features, md_pred, dis_pred, res_pred, jloc_pred, joff_pred, tspan, pool1d, fc2 = model(images, targets, metas)

            batch_size = md_pred.size(0)
            for i, (md_pred_per_im, dis_pred_per_im, res_pred_per_im, meta) in enumerate(
                    zip(md_pred, dis_pred, res_pred, metas)):
                lines_pred = []
                if True:
                    for scale in [-1.0, 0.0, 1.0]:
                        _ = proposal_lines(md_pred_per_im, dis_pred_per_im + scale * res_pred_per_im).view(-1, 4)
                        lines_pred.append(_)
                else:
                    lines_pred.append(self.proposal_lines(md_pred_per_im, dis_pred_per_im).view(-1, 4))
                lines_pred = torch.cat(lines_pred)
                junction_gt = meta['junc'] #outside
                N = junction_gt.size(0)

                juncs_pred, _ = get_junctions(non_maximum_suppression(jloc_pred[i]), joff_pred[i],
                                            topk=min(N * 2 + 2, n_dyn_junc))
                dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1).min(
                    0)
                dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(
                    0)

                idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
                idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)
                iskeep = idx_junc_to_end_min < idx_junc_to_end_max
                idx_lines_for_junctions = torch.cat((idx_junc_to_end_min[iskeep, None], idx_junc_to_end_max[iskeep, None]),
                                                    dim=1).unique(dim=0)
                idx_lines_for_junctions_mirror = torch.cat(
                    (idx_lines_for_junctions[:, 1, None], idx_lines_for_junctions[:, 0, None]), dim=1)
                idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
                lines_adjusted = torch.cat(
                    (juncs_pred[idx_lines_for_junctions[:, 0]], juncs_pred[idx_lines_for_junctions[:, 1]]), dim=1)

                cost_, match_ = torch.sum((juncs_pred - junction_gt[:, None]) ** 2, dim=-1).min(0)
                match_[cost_ > 1.5 * 1.5] = N
                Lpos = meta['Lpos']
                Lneg = meta['Lneg']
                labels = Lpos[match_[idx_lines_for_junctions[:, 0]], match_[idx_lines_for_junctions[:, 1]]]

                iskeep = torch.zeros_like(labels, dtype=torch.bool)
                cdx = labels.nonzero().flatten()

                if len(cdx) > n_dyn_posl:
                    perm = torch.randperm(len(cdx), device=device)[:n_dyn_posl]
                    cdx = cdx[perm]

                iskeep[cdx] = 1

                if n_dyn_negl > 0:
                    cdx = Lneg[
                        match_[idx_lines_for_junctions[:, 0]], match_[idx_lines_for_junctions[:, 1]]].nonzero().flatten()

                    if len(cdx) > n_dyn_negl:
                        perm = torch.randperm(len(cdx), device=device)[:n_dyn_negl]
                        cdx = cdx[perm]

                    iskeep[cdx] = 1

                if n_dyn_othr > 0:
                    cdx = torch.randint(len(iskeep), (n_dyn_othr,), device=device)
                    iskeep[cdx] = 1

                if n_dyn_othr2 > 0:
                    cdx = (labels == 0).nonzero().flatten()
                    if len(cdx) > n_dyn_othr2:
                        perm = torch.randperm(len(cdx), device=device)[:n_dyn_othr2]
                        cdx = cdx[perm]
                    iskeep[cdx] = 1

                lines_selected = lines_adjusted[iskeep]
                labels_selected = labels[iskeep]

                lines_for_train = torch.cat((lines_selected, meta['lpre']))
                labels_for_train = torch.cat((labels_selected.float(), meta['lpre_label']))

                logits = pooling(loi_features[i], lines_for_train, tspan, n_pts1, dim_loi, pool1d, fc2)

                loss_ = loss(logits, labels_for_train)

                loss_positive = loss_[labels_for_train == 1].mean()
                loss_negative = loss_[labels_for_train == 0].mean()

                loss_dict['loss_pos'] += loss_positive / batch_size
                loss_dict['loss_neg'] += loss_negative / batch_size
        
            total_loss = loss_reducer(loss_dict)
            for key in loss_dict.keys():
                writer.add_scalar(key, loss_dict[key].item(), global_iteration)
            writer.add_scalar("total_loss", total_loss.item(), global_iteration)
            with torch.no_grad():
                loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
                loss_reduced = total_loss.item()
                meters.update(loss=loss_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_iteration += 1

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_batch = epoch_size * (max_epoch - epoch + 1) - it + 1
            eta_seconds = meters.time.global_avg * eta_batch
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if it % 200 == 0 or it + 1 == len(train_dataset):
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}\n",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=it,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

        checkpointer.save('model_{:05d}'.format(epoch))
        scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HAWP Training')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default = "./config-files/hawp.yaml",
                        required=False,
                        )
    parser.add_argument("--clean",
                        default=False,
                        action='store_true')

    parser.add_argument("--seed",
                        default=2,
                        type=int)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.LOSS_WEIGHTS.loss_mask = 1.0
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        if os.path.isdir(output_dir) and args.clean:
            import shutil

            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('hawp', output_dir, out_file='train.log')
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))

    save_config(cfg, output_config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    train(cfg)
