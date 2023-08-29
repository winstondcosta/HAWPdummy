import os
from tqdm import tqdm
import json
import cv2

# sizes = {}
# for file in os.listdir("./data/wireframe/images/"):
# 	img = cv2.imread("./data/wireframe/images/"+file)
# 	sizes[img.shape] = sizes.get(img.shape, 0) + 1
# print(sizes)
# exit()
# data = {}
# for folder in ["training", "testing", "validation"]:
# 	ann_folder = "./data/wireframe/"+folder+"/annotations/"
# 	for file in os.listdir(ann_folder):
# 		corrected = {}
# 		curr = json.load(open(ann_folder + file, "r"))
# 		for k, v in curr.items():
# 			print(v['regions'][0])
# 			exit()
# 			corrected[v['filename']] = v
# 		data.update(corrected)

# 		print(os.path.join(ann_folder, file), len(curr), len(data), len(corrected))

# json.dump(data, open("./data/wireframe/mask_data.json", "w"))
# exit()
models_folder = "outputs/hawp/"
for file in tqdm(sorted(os.listdir(models_folder))):
	if("model" not in file):
		continue
	# if(file < "model_00026.pth"):continue
	os.system("python scripts/test.py --config-file "+models_folder+"config.yml --model_path "+models_folder+file)
