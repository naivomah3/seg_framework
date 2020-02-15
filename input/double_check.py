import os
import cv2
import random
import numpy as np
from tqdm import tqdm

#### Check missing files within images&mask
# for f in image_files:
#     if f not in mask_files:
#         print(f"{f} is missing")

#### Match dimension
# img_path = 'training/images/'
# mask_path = 'training/mask/'
# for f in mask_files:
#     img = cv2.imread(img_path + f, 1)
#     mask = cv2.imread(img_path + f)

    # if type(img) is not np.ndarray:
    #     os.remove(img_path + f)
    #     os.remove(mask_path + f)
    #     print(f"{f} get removed")

    # if img.shape[0] != 576:
    #     img_out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #     mask_out = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    #     cv2.imwrite(img_path + f, img_out)
    #     cv2.imwrite(mask_path + f, mask_out)
    #     print(f'shape: {img_out.shape}')
    #     print(f'corrected : {f}')
    # else:
    #     print(img.shape)


#### Random create set from source
# for i in tqdm(range(200)):
#     source = set(os.listdir('source/training_final_1/images/'))
#     dest = set(os.listdir('testing/images_in/'))
#     diff = source.difference(dest)
#     frames = list(diff)
#     random.shuffle(frames)
#     frame_id = frames[random.randint(0, len(source) - 1)]
#     # read&write images and masks
#     img = cv2.imread('source/training_final_1/images/' + frame_id, 1)
#     mask = cv2.imread('source/training_final_1/mask/' + frame_id)
#     cv2.imwrite('testing/images_in/' + frame_id, img)
#     cv2.imwrite('testing/mask_in/' + frame_id, mask)
#     # remove images from source
#     if os.path.exists('source/training_final_1/images/' + frame_id):
#         os.remove('source/training_final_1/images/' + frame_id)
#         os.remove('source/training_final_1/mask/' + frame_id)
#     else:
#         raise Exception("file doesn't exist")

with open("../labels.txt", 'r') as file:
    CLASSES = list(file)
    print(f"{len(CLASSES)}")
if not CLASSES:
    raise Exception(f"Unable to load label file {CLASSES}")

print("Everything is done :)")





