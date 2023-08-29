import cv2, os, sys, numpy as np

# Give two folder names you want to compare
folder1 = sys.argv[1] if len(sys.argv) > 1 else None
folder2 = sys.argv[2] if len(sys.argv) > 2 else None
folder3 = sys.argv[3] if len(sys.argv) > 3 else None
count = int(sys.argv[4]) if len(sys.argv) > 4 else 0
files = sorted(
    [file for file in set(os.listdir(folder1)).intersection(os.listdir(folder2)) if not file.startswith(".")])
print(len(files))

# Any csv data analysis if you need?
analysis1_data = {}
analysis2_data = {}
while count < len(files):
    file = files[count]
    sAP1, f11 = analysis1_data.get(file, [0, 0])[0], analysis1_data.get(file, [0, 0, 0, 0])[3]
    sAP2, f12 = analysis2_data.get(file, [0, 0])[0], analysis2_data.get(file, [0, 0, 0, 0])[3]

    img1 = cv2.imread(os.path.join(folder1, file))
    img2 = cv2.imread(os.path.join(folder2, file))
    img = np.concatenate((img1, img2), axis=1)
    # Add third image file if needed
    # img3 = cv2.imread(os.path.join(folder3, file))
    # img = np.concatenate((img1, img2, img3), axis = 1)
    cv2.imshow("Stacked", img)
    key = cv2.waitKey(0)
    # quit
    if key == ord('q'):
        print(count)
        break
    # go to previous image
    elif key == ord('b'):
        count -= 2
    # Save the stacked image
    elif key == ord('s'):
        cv2.imwrite(f"./dump/{file}", img)
    count += 1
