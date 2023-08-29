import cv2
import numpy as np

def mask_post_processing(prediction_boxes, prediction_masks, image_size):
        
    polygon_masks = {}
    polygon_masks["polygonObstructions"] = {}
    polygons = []

    for box, mask in zip(prediction_boxes, prediction_masks):
        x1, y1, x2, y2 = map(round, box)
        nx1, ny1, nx2, ny2 = max(0, x1 - 2), max(0, y1 - 2), (x2 + 2), (y2 + 2)
        masked_region = mask[ny1: ny2 + 1, nx1: nx2 + 1]
        pys, pxs = np.where(masked_region)
        if(len(pxs) < 512 * 512 * 0.002):
            # print("Found an intruder", len(pxs) / (512* 512))
            continue
        contours, _ = cv2.findContours(masked_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for object in contours:
            polygon_mask = {}
            polygon = []
            peri = cv2.arcLength(object, True)
            approx = cv2.approxPolyDP(object, 0.04 * peri, True)
            
            for point in approx:
                x = np.clip(int(point[0][0]) + nx1, a_min = 0, a_max = image_size[0]-1)
                y = np.clip(int(point[0][1]) + ny1, a_min = 0, a_max = image_size[1]-1)
                p = [x,y]
                polygon.append(p)
            polygons.append(polygon)
            break
    return polygons