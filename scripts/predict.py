import torch
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset.build import build_transform
from parsing.detector import get_hawp_model
from parsing.utils.logger import setup_logger
from parsing.utils.mask_processing import mask_post_processing
from skimage import io
# from detectron2.config import get_cfg
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
import numpy as np
from random import randint as ri

parser = argparse.ArgumentParser(description='HAWP Testing')

parser.add_argument("--img",type=str,default = None,
                    help="image path")

parser.add_argument("--img_folder",type=str,default = None,
                    help="image folder path")                    

parser.add_argument("--threshold",
                    type=float,
                    default=0.9)
parser.add_argument("--model_path",type=str,default=None)

args = parser.parse_args()


def test(cfg, impath):
    device = cfg.MODEL.DEVICE
    # cfg_det = get_cfg()
    # cfg_det_path = "config-files/train_config.yaml"
    # cfg_det.merge_from_file(cfg_det_path)
    cfg_det = None

    model = get_hawp_model(pretrained=False, cfg_det = cfg_det, network_type = "hawp")
    if(args.model_path):
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()
    files = []
    if(args.img):
        files =  [args.img]
    else:
        files = [os.path.join(args.img_folder, file) for file in os.listdir(args.img_folder)]
    transform = build_transform(cfg)
    for impath in tqdm(files):
        image = io.imread(impath)
        image = cv2.resize(image, (512, 512))[:, :, :3]
        image_tensor = transform(image.astype(float))[None].to(device)
        meta = {
            'filename': impath,
            'height': image.shape[0],
            'width': image.shape[1],
            'mask_anns': {'height':image.shape[0], 'width':image.shape[1]},
        }
        
        with torch.no_grad():
            mask_network, output, _ = model(image_tensor,[meta])
            output = to_device(output,'cpu')

        def drawit(low, high, design, img):
            curr = lines[(scores<=high)&(scores>low)]
            for x1,y1,x2,y2 in curr:
                x1,y1,x2,y2=map(round,[x1,y1,x2,y2])
                img = cv2.line(img,(x1,y1),(x2,y2),design, 1)
            return img
        
        lines = output['lines_pred'].numpy()
        scores = output['lines_score'].numpy()
        idx = scores>args.threshold
        thres = list(reversed([0.9,0.8,0.7,0.6,0.5,0.4,0.3]))
        colorssym = list(reversed(["r-","g-","b-","k-","w-","m-","y-"]))
        colors = list(reversed([(0,0,255),(0,255,0),(255,0,0),(0,0,0),(255,255,255),(255,0,255),(0,255,255)]))
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_mask = img.copy()
        name = ".".join(os.path.basename(impath).split(".")[:-1])
        for s, sym in zip(thres, colors):
            if(s < 0.5):continue
            img = drawit(s,s+0.1,sym, img)
        
        if mask_network:
            instance_masks = mask_network[0]['instances'].pred_masks.cpu().numpy()
            masks = np.zeros_like(instance_masks, dtype = "uint8")
            masks[instance_masks] = 255
            instance_boxes = mask_network[0]['instances'].pred_boxes.tensor.cpu().numpy()
            instance_scores = mask_network[0]['instances'].scores.cpu().numpy()
            polygons = mask_post_processing(instance_boxes[instance_scores > 0.3], masks[instance_scores > 0.3], (512, 512))
            
            for mask, score in zip(instance_masks, instance_scores):
                if(score < 0.5):continue
                adder = img_mask.copy()
                adder[mask] = (ri(0, 255), ri(0, 255), ri(0, 255))
                img_mask = cv2.addWeighted(img_mask, 0.5, adder, 0.5, 0)
            # for p in polygons:
            #     img_mask = cv2.polylines(img_mask, np.array([p]), True, (255, 255, 255), 2)
            cv2.imwrite(os.path.join("./data/results/" + name + "_mask.png"), img_mask.astype('uint8'))

        # cv2.imshow("Image", img.astype('uint8'))
        # cv2.imshow("Image_mask", img_mask.astype('uint8'))
        # cv2.waitKey(0) ; exit()
        cv2.imwrite(os.path.join("./data/results/" + name + "_old.png"), img)

        # plt.figure(figsize=(6,6))    
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        #         hspace = 0, wspace = 0)
        # plt.imshow(image)
        # plt.plot([lines[idx,0],lines[idx,2]],
        #                     [lines[idx,1],lines[idx,3]], 'b-')
        # plt.plot(lines[idx,0],lines[idx,1],'c.')                        
        # plt.plot(lines[idx,2],lines[idx,3],'c.')                        
        # plt.axis('off')
        # plt.savefig(os.path.join("results", impath.split("/")[-1].split("\\")[-1]), bbox_inches='tight', pad_inches=0)
        # plt.clf()

        # plt.show()
    
if __name__ == "__main__":
    cfg.freeze()
    test(cfg,args.img)

