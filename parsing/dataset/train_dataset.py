import torch
from torch.utils.data import Dataset
from skimage.transform import resize
import os.path as osp
import json
import cv2
from skimage import io
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import copy
import os
import math
from random import randint as ri
# from detectron2.structures import BoxMode
# from detectron2.data import detection_utils as utils
import albumentations as A
from scipy.ndimage import median_filter
from skimage.draw import line
# transform_1 = A.Compose([
#         A.RandomBrightness(p=0.2),
#         ])
transform_1 = A.Compose([
    A.RandomBrightnessContrast(p=0.8,brightness_limit=0.1,contrast_limit=0.2),
])

transform_2 = A.Compose([
        A.GaussNoise(p=0.8),
        ])

# transform_3 = A.Compose([
#         A.RandomContrast(p=0.8),
#         ])
transform_3= A.Compose([
    A.RandomBrightnessContrast(p=0.8,brightness_limit=0.4,contrast_limit=0.2),
])

transform_4 = A.Compose([
        A.Sharpen(p=0.8),
        ])

transform_5 = A.Compose([
        A.RandomSnow(p=0.8),
        ])

transform_6 = A.Compose([
        A.RandomFog(p=0.8),
        ])

transform_7 = A.Compose([
        A.RandomRain(p=0.8),
        ])

transform_8 = A.Compose([
        A.RandomShadow(p=0.8),
        ])

transform_9 = A.Compose([
        A.InvertImg(p=0.8),
        ])

transform_10 = A.Compose([
        A.FancyPCA(p=0.8),
        ])

transform_11 = A.Compose([
        A.RandomShadow(p=0.8),
        ])

transform_12 = A.Compose([
        A.ColorJitter(p=0.8),
        ])

transform_13 = A.Compose([
        A.Superpixels(p=0.8),
        ])

transform_14 = A.Compose([
        A.PixelDropout(p=0.8),
        ])

transform_15 = A.Compose([
        A.Emboss(p=0.8),
        ])

transform_16 = A.Compose([
        A.Downscale(p=0.8),
        ])

transform_17 = A.Compose([
        A.MultiplicativeNoise(p=0.8),
        ])

transform_18 = A.Compose([
        A.Posterize(p=0.8),
        ])

# transform_19 = A.Compose([
#         A.Solarize(p=0.8),
#         ])
transform_19 = A.Compose([
    A.Solarize(p=0.8,threshold=180),
])

transform_20 = A.Compose([
        A.Equalize(p=0.8),
        ])

transform_21 = A.Compose([
        A.HueSaturationValue(p=0.8),
        ])

# transform_22 = A.Compose([
#         A.RGBShift(p=0.8),
#         ])
transform_22 = A.Compose([
    A.RGBShift(p=0.8,r_shift_limit=20,
        g_shift_limit=20,
        b_shift_limit=20),

])

transform_23 = A.Compose([
        A.ChannelShuffle(p=0.8),
        ])

transform_24 = A.Compose([
        A.ToGray(p=0.8),
        ])

transform_25 = A.Compose([
        A.ToSepia(p=0.8),
        ])

transform_28 = A.Compose([
    A.CLAHE(p=0.8),
])

transform_30 = A.Compose([
    A.RandomToneCurve(p=0.8,scale=0.1),
])

def unsharp(image, sigma, strength):
    # Median filtering
    image_mf = median_filter(image, sigma)
    # Calculate the Laplacian
    image_mf = image_mf.astype(np.float64) 
    lap = cv2.Laplacian(image_mf, cv2.CV_64F)
    # Calculate the sharpened image
    sharp = image - strength * lap
    # Saturate the pixels in either direction
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0
    return sharp
class TrainDataset(Dataset):
    def __init__(self, root, ann_file, transform = None):
        self.root = root
        with open(ann_file,'r') as _:
            self.annotations = json.load(_)

        print(len(self.annotations))
        # mask_ann_path = "./data/wireframe/mask_data.json"
        # assert osp.isfile(mask_ann_path), "Mask annotations are missing"
        # with open(mask_ann_path, "r") as f:
        #     self.mask_annotations = json.load(f)
        self.mask_annotations = None
        self.transform = transform
        self.count = 0
        self.images = 0
        occ_folder = "./data/Stock images/"
        self.occ_files = [occ_folder+file for file in os.listdir(occ_folder) if "DS_" not in file]
        self.occ_length = len(self.occ_files)
        self.outer_contour = json.load(open("data/wireframe/train_outer_contour.json"))

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
    
    def __getitem__(self, idx_):
        idx = idx_%len(self.annotations)
        reminder = idx_//len(self.annotations)
        ann = copy.deepcopy(self.annotations[idx])
        ann['reminder'] = reminder
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        seg_img = io.imread(osp.join("/home/cirrusrays/Winston/ARD/seg_test_images/",ann['filename'])).astype(float)[:,:,:3]
        #image = io.imread(osp.join()).astype(float)[:,:,:3]

        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        # for key,_type in (['junctions',np.float32],
        #                   ['edges_positive',np.int32],
        #                   ['edges_negative',np.int32]):
        #     ann[key] = np.array(ann[key],dtype=_type)

        ann['junc'] = np.array(ann['junc'],dtype=np.float32)
        
        width = ann['width']
        height = ann['height']
        self.images += 1
        if False:
            if(random.randint(1, 10) <= 5):
                mask = cv2.imread(self.occ_files[ri(0, self.occ_length - 1)])
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                sx, sy, hm, wm = ri(165, 286), ri(165, 286), len(mask), len(mask[0])
                lines = self.outer_contour.get(ann['filename'], [])
                ratio = math.sqrt(random.choice([4900, 6400, 8100, 10000, 12100, 14400, 16900]) / (hm * wm))
                hm, wm = int(ratio * hm), int(wm * ratio)
                mask = cv2.resize(mask, (wm, hm))
                if lines:
                    idx = random.randint(0, len(lines) - 1)
                    [x1, y1], [x2, y2] = lines[idx], lines[(idx + 1)%len(lines)]
                    ratio = random.randint(1, 99) / 100
                    sx, sy = int(x1 + ratio * (x2 - x1)), int(y1 + ratio * (y2 - y1))            
                    sx, sy = max(0, sx - wm // 2), max(0, sy - hm // 2)

                mask_of_mask = (mask[:,:,0]<160) | (mask[:,:,1]<160) | (mask[:,:,2]<160)
                try:
                    mask = np.clip((mask * 0.4) + 30, 0, 255)
                    image[sy: sy+hm, sx: sx+wm][mask_of_mask] = mask[mask_of_mask]
                except:
                    pass
        # Try image blurring or sharpening
        if(random.randint(1, 10) <= 4):
            image = cv2.GaussianBlur(image, (2 * random.randint(1, 3) + 1, 2 * random.randint(1, 3) + 1), 0)

        # Try contrast or brightness augmentation
        if(random.randint(1, 10) <= 6):
            image = image.astype('uint8')
            transformed_1 = transform_1(image=image)
            image = transformed_1["image"]
            image = image.astype('float64')
            
        #brightness = random.randint(0, 50)
        #contrast = random.randint(7, 13) / 10
        #image = np.clip((image * contrast) + brightness, 0, 255)

        
        # if(self.mask_annotations) and False:
        #     # ann['mask_anns'] = self.get_mask_anns(osp.basename(ann['filename']), idx)
        #     # mask = self.draw_mask(image, ann['mask_anns']['annotations'])
        #     mask = cv2.imread("../pv-porch-code-data/mrcnn_pytorch_detectron2/masks/"+ann['filename']+".png", 0)
        # else:
        #     mask = np.zeros_like(image[:, :, 0])
        # if False:
        #     if self.transform is not None:
        #         image, ann = self.transform(image,ann)
        #         image = image.permute(2,1,0)
        #         image = np.array(image)

        # plt.imshow(image.astype('uint8'))
        # plt.show()

        reminder = random.randint(1, 22)
        #print(reminder)
        if reminder == 1:
            image = image[:,::-1,:]
            ann['junc'][:,0] = width-ann['junc'][:,0]
            #
            seg_img = seg_img[:,::-1,:]

        elif reminder == 2:
            image = image[::-1,:,:]
            ann['junc'][:,1] = height-ann['junc'][:,1]
            #
            seg_img = seg_img[::-1,:,:]
        elif reminder == 3:
            image = image[::-1,::-1,:]
            ann['junc'][:,0] = width-ann['junc'][:,0]
            ann['junc'][:,1] = height-ann['junc'][:,1]
            #
            seg_img = seg_img[::-1,::-1,:]
        
        elif reminder == 4:
            #print("4sharpen")
            transformed_4 = transform_4(image=image)
            image = transformed_4["image"]
                                                                        
        elif reminder == 5:
            #multiplicative noise
            image = image.astype('uint8')
            transformed_17 = transform_17(image=image)
            image = transformed_17["image"]
            image = image.astype('float64')

        elif reminder == 6:
            image = image.astype('uint8')
            transformed_25 = transform_25(image=image)
            image = transformed_25["image"]
            image = image.astype('float64')

        elif reminder == 7: #uint8 or float32
            transformed_23 = transform_23(image=image)
            image = transformed_23["image"]
        
        #new added
        elif reminder == 8: #uint8 #invert
            image = image.astype('uint8') #uint8 or float32
            transformed_9 = transform_9(image=image)
            image = transformed_9["image"]
            image = image.astype('float64')

        # elif reminder == 9:
        #     image = image.astype('uint8')
        #     transformed_1 = transform_1(image=image)
        #     image = transformed_1["image"]
        #     image = image.astype('float64')
            
        elif reminder == 9:
            image = image.astype('uint8')
            transformed_3 = transform_3(image=image)
            image = transformed_3["image"]
            image = image.astype('float64')

        # elif reminder == 9: #uint8
        #     image = image.astype('uint8') #not given --> try unint8
        #     transformed_12 = transform_12(image=image)
        #     image = transformed_12["image"]
        #     image = image.astype('float64')

        elif reminder == 10: #uint8 #inverted solarize
            image = image.astype('uint8')# any dtype
            transformed_9 = transform_9(image=image)
            image_inverted = transformed_9["image"]
            transformed_19 = transform_19(image=image_inverted)
            image = transformed_19["image"]
            image = image.astype('float64')

        # elif reminder == 11: #uint8
        #     image = image.astype('uint8')#only uint8
        #     transformed_20 = transform_20(image=image)
        #     image = transformed_20["image"]
        #     image = image.astype('float64')

        # elif reminder == 12: #uint8
        #     image = image.astype('uint8') #uint8 or float32
        #     transformed_21 = transform_21(image=image)
        #     image = transformed_21["image"]
        #     image = image.astype('float64')

        # elif reminder == 13: #uint8
        #     image = image.astype('uint8') #uint8 or float32
        #     transformed_22 = transform_22(image=image)
        #     image = transformed_22["image"]
        #     image = image.astype('float64')
        
        # elif reminder == 14: #uint8
        #     image = image.astype('uint8') #uint8 or float32
        #     transformed_24 = transform_24(image=image)
        #     image = transformed_24["image"]
        #     image = image.astype('float64')

        # elif reminder == 15: #uint8
        #     image = image.astype('uint8')#only uint8
        #     transformed_28 = transform_28(image=image)
        #     image = transformed_28["image"]
        #     image = image.astype('float64')
        
        # elif reminder == 16: #uint8
        #     image = image.astype('uint8')#only uint8
        #     transformed_30 = transform_30(image=image)
        #     image = transformed_30["image"]
        #     image = image.astype('float64')


        elif reminder >=11:
            angle = random.randint(1, 3)
            M = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle * 90, 1)
            image = cv2.warpAffine(image, M, (height, width))
            ann['junc'] -= 255.5 
            ann['junc'] = np.dot(ann['junc'], cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle * -90, 1))[:, :2]
            ann['junc'] += 255.5
            ann['junc'] = ann['junc'].astype('float32')
            #
            seg_img = cv2.warpAffine(seg_img, M, (height, width))
        else:
            pass
        
        # plt.imshow(image.astype('uint8'))
        # plt.show()

        if self.transform is not None:
            image, ann = self.transform(image,ann)
            seg_img, ann2 = self.transform(seg_img,ann)
            
            # seg_img = resize(seg_img, (512,512))/255.0
            # seg_img = F.to_tensor(seg_img)
            image = torch.cat([image, seg_img], dim=0)
            # imagep = image.permute(2,1,0)
            # plt.imshow(imagep)

        
        # if(mask is not None) and False:
        #     mask = mask[np.newaxis, :, :]
        #     mask = torch.Tensor(mask.copy())
        #     image = torch.cat([image, mask], axis = 0)

        #4c GT binary mask
        #512x512
        if False :
            blank_image = np.zeros((512, 512, 1))
            junctions = np.array(ann['junctions'])
            edges_positive = np.array(ann['edges_positive'])
            lines = np.concatenate((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),axis=-1)
            for i in range(len(lines)):
                rr, cc = line(int(lines[i,0]), int(lines[i,1]), int(lines[i,2]),int(lines[i,3]))
                blank_image[rr,cc] = 1
            
            mask = blank_image
            #print(mask.shape)
            if(mask is not None) and True:
                #mask = mask[np.newaxis, :, :]
                mask = torch.Tensor(mask.copy())
                mask = mask.permute(2,0,1)
                #image = image[np.newaxis, :, :]
                #print("masking")
                #print(image.shape)
                #print(mask.shape)
                image = torch.cat([image, mask], axis = 0)

                #image = image[np.newaxis, :, :]

        #LOG
        if False :
            for i in range(3):
                image[:, :, i] = unsharp(image[:, :, i], 5, 1)

        return image, ann

    def __len__(self):
        return len(self.annotations) * 8

    def draw_mask(self, image, mask_ann):
        mask = np.zeros_like(image[:, :, 0])
        total, kernel = len(mask_ann), np.ones((7, 7), dtype = np.uint8)
        start, change = 255 // total, 255 // total
        for obj in mask_ann:
            if(ri(1, 10) <= 3):
                start += change
                continue
            points = obj['segmentation'][0]
            points = [[x, y] for x, y in zip(points[::2], points[1::2])]
            points.append(points[0])
            tmask = np.zeros_like(mask)
            tmask = cv2.fillPoly(tmask, np.array([points]), color = start)
            tmask = cv2.erode(tmask, kernel, 1)
            mask += tmask
            start += change
        return mask


def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])