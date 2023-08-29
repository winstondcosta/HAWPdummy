import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json
import copy
from PIL import Image
from skimage import io
import os
import os.path as osp
import numpy as np
import cv2

# from detectron2.structures import BoxMode
# from detectron2.data import detection_utils as utils
class TestDatasetWithAnnotations(Dataset):
    '''
    Format of the annotation file
    annotations[i] has the following dict items:
    - filename  # of the input image, str 
    - height    # of the input image, int
    - width     # of the input image, int
    - lines     # of the input image, list of list, N*4
    - junc      # of the input image, list of list, M*2
    '''

    def __init__(self, root, ann_file, transform = None):
        self.root = root
        with open(ann_file, 'r') as _:
            self.annotations = json.load(_)
        # with open(ann_file[:-9] + 'validation.json', 'r') as _:
        #     self.annotations += json.load(_)
        # mask_ann_path = "./data/wireframe/mask_data.json"
        # assert osp.isfile(mask_ann_path), "Mask annotations are missing"
        # with open(mask_ann_path, "r") as f:
        #     self.mask_annotations = json.load(f)
        self.mask_annotations = None
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def get_mask_anns(self, name, idx = 0):
        record = {}
        record["image_id"] = idx
        v = self.mask_annotations[name]
        annos = v["regions"]

        objs = []
        for anno in annos:
            region_attributes = anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        instances = utils.annotations_to_instances(
            objs, (512, 512), mask_format="polygon"
        )
        record['instances'] = instances
        return record

    def __getitem__(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        for key, _type in (['junc',np.float32],
                            ['lines',  np.float32]):
            ann[key] = np.array(ann[key],dtype=_type)
        mask = None
        if(self.mask_annotations):
            ann['mask_anns'] = self.get_mask_anns(osp.basename(ann['filename']), idx)
            mask = self.draw_mask(image, ann['mask_anns']['annotations'])

        if self.transform is not None:
            image, ann = self.transform(image,ann)
        
        if(mask is not None) and False:
            mask = cv2.imread("../pv-porch-code-data/mrcnn_pytorch_detectron2/masks/"+ann['filename']+".png", 0)
            mask = mask[np.newaxis, :, :]
            image = torch.cat([image, torch.Tensor(mask)], axis = 0)
        return image, ann

    def draw_mask(self, image, mask_ann):
        mask = np.zeros_like(image[:, :, 0])
        total = len(mask_ann)
        start, change = 255 // total, 255 // total
        for obj in mask_ann:
            points = obj['segmentation'][0]
            points = [[x, y] for x, y in zip(points[::2], points[1::2])]
            points.append(points[0])
            mask = cv2.fillPoly(mask, np.array([points]), color = start)
            start += change
        return mask

    def image(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        return image
    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])
    