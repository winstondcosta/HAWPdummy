import os
import torch.nn as nn

from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from .multi_task_head import MultitaskHead
import torch
from detectron2.structures import ImageList, Instances



class HybridModel(GeneralizedRCNN):
    def __init__(self, cfg=None, backbone=None):
        assert cfg, "Config for detectron2 not provided"
        try:
            output_shape = backbone.output_shape()
        except:
            output_shape = (512, 512)
        super().__init__(
            backbone=None,
            proposal_generator=build_proposal_generator(cfg, output_shape),
            roi_heads=build_roi_heads(cfg, output_shape),
            input_format=cfg.INPUT.FORMAT,
            vis_period=cfg.VIS_PERIOD,
            pixel_mean=cfg.MODEL.PIXEL_MEAN,
            pixel_std=cfg.MODEL.PIXEL_STD,
        )
        self.fc = self._make_fc(256, 256)
        self.score = MultitaskHead(256, 9, head_size=[[3], [1], [1], [2], [2]])

    def forward(self, images, features, batched_inputs):
        if not self.training:
            return self.inference(images, features, batched_inputs)

        # images = self.preprocess_image(images)
        # features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        y = self.fc(features['p2'])
        score = self.score(y)
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses = {"loss_mask":sum([v for k, v in losses.items()])}
        return losses, [score], y

    def inference(
            self,
            images,
            features,
            batched_inputs,
            detected_instances=None,
            do_postprocess=True,
    ):
        assert not self.training

        # images = self.preprocess_image(images)
        # features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        y = self.fc(features['p2'])
        score = self.score(y)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes), [score], y
        else:
            return results, [score], y

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, self.relu)

    def preprocess_image(self, batched_inputs):
        images = [x for x in batched_inputs]
        images = ImageList.from_tensors(images, 32)
        return images


if __name__ == '__main__':
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg_path = "../../config-files/train_config.yaml"
    cfg.merge_from_file(cfg_path)
    model = HybridModel(cfg)
