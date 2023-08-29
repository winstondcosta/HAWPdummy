import torch
import torchvision
from torchvision.transforms import functional as F
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, ann=None):
        if ann is None:
            for t in self.transforms:
                image = t(image)
            return image
        for t in self.transforms:
            image, ann = t(image, ann)
        return image, ann


class Resize(object):
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width = image_width
        self.ann_height = ann_height
        self.ann_width = ann_width

    def __call__(self, image, ann):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0

        sx = self.ann_width / ann['width']
        sy = self.ann_height / ann['height']
        ann['junctions'][:, 0] = np.clip(ann['junctions'][:, 0] * sx, 0, self.ann_width - 1e-4)
        ann['junctions'][:, 1] = np.clip(ann['junctions'][:, 1] * sy, 0, self.ann_height - 1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height

        return image, ann


class LetterBoxResize(object):
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width = image_width
        self.ann_height = ann_height
        self.ann_width = ann_width

    def __call__(self, image, ann):
        img_height, img_width = ann['height'], ann['width']
        scale = min(self.image_height / img_height, self.image_width / img_width)
        new_height, new_width = map(int, [img_height * scale, img_width * scale])
        new_image = np.zeros((self.image_height, self.image_width, 3), dtype='uint8')
        image = resize(image, (new_height, new_width))
        starth, startw = (self.image_height - new_height) // 2, (self.image_width - new_width) // 2
        new_image[starth:starth+new_height,startw:startw+new_width] = image

        s = scale / 4
        ann['junc'][:, 0] = np.clip(ann['junc'][:, 0] * s + startw // 4, 0, self.ann_width - 1e-4)
        ann['junc'][:, 1] = np.clip(ann['junc'][:, 1] * s + starth // 4, 0, self.ann_height - 1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height
        new_image = np.array(new_image, dtype=np.float32) / 255.0

        return new_image, ann


class ResizeImage(object):
    def __init__(self, image_height, image_width, letter_box = True):
        self.image_height = image_height
        self.image_width = image_width
        self.letter_box = letter_box

    def __call__(self, image, ann=None):
        if(self.letter_box):
            img_height, img_width, _ = image.shape
            scale = min(self.image_height / img_height, self.image_width / img_width)
            new_height, new_width = map(int, [img_height * scale, img_width * scale])
            new_image = np.zeros((self.image_height, self.image_width, 3), dtype='uint8')
            image = resize(image, (new_height, new_width))
            starth, startw = (self.image_height - new_height) // 2, (self.image_width - new_width) // 2
            new_image[starth:starth+new_height,startw:startw+new_width] = image
            if(ann is not None):
                ann['start'] = (startw, starth)
                ann['height'], ann['width'] = self.image_height, self.image_width
                ann['scale'] = scale
        else:
            new_image = resize(image, (self.image_height, self.image_width))
        new_image = np.array(new_image, dtype=np.float32) / 255.0
        if ann is None:
            return new_image
        return new_image, ann


class ToTensor(object):
    def __call__(self, image, anns=None):
        if anns is None:
            return F.to_tensor(image)

        for key, val in anns.items():
            if isinstance(val, np.ndarray):
                anns[key] = torch.from_numpy(val)
        return F.to_tensor(image), anns


class Normalize(object):
    def __init__(self, mean, std, to_255=False):
        self.mean = mean
        self.std = std
        self.to_255 = to_255

    def __call__(self, image, anns=None):
        if self.to_255:
            image *= 255.0

        # plt.imshow(np.array(image.permute(2,1,0)).astype('uint8'))
        # plt.show()

        image = F.normalize(image, mean=self.mean, std=self.std)

        # plt.imshow(np.array(image.permute(2,1,0)).astype('uint8'))
        # plt.show()
        if anns is None:
            return image
        return image, anns
