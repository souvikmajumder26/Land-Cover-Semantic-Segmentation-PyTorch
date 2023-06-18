import os
import cv2
import numpy as np
from tqdm import tqdm
from patchify import patchify

def patching(data_dir, patches_dir, file_type, patch_size):
    img_list = list(filter(lambda x:x.endswith((file_type)), os.listdir(data_dir)))
    for filename in tqdm(img_list):
        img = cv2.imread(os.path.join(data_dir, filename), 1)
        # cropping to have height and width perfectly divisible by patch_size
        max_height = (img.shape[0] // patch_size) * patch_size
        max_width = (img.shape[1] // patch_size) * patch_size
        img = img[0:max_height, 0:max_width]
        # patching
        # print(f"Patchifying {filename}...")
        patches = patchify(img, (patch_size, patch_size, 3), step = patch_size)  # non-overlapping
        # print("Patches shape:", patches.shape)
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, 0, :, :] # the 0 is an extra unncessary dimension added by patchify for multiple channels scenario
                cv2.imwrite(os.path.join(patches_dir, filename.replace(file_type, f"_patch_{i}_{j}" + file_type)), single_patch)

def discard_useless_patches(patches_img_dir, patches_mask_dir, discard_rate):
    for filename in tqdm(os.listdir(patches_mask_dir)):
        img_path = os.path.join(patches_img_dir, filename)
        mask_path = os.path.join(patches_mask_dir, filename)
        mask = cv2.imread(mask_path)
        classes, count = np.unique(mask, return_counts = True)
        # If background class occupies more than the discard rate of the image, discard the image and mask
        if (count[0] / count.sum()) > discard_rate:
            os.remove(img_path)
            os.remove(mask_path)