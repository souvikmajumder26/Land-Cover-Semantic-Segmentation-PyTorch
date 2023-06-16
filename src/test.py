import os
import cv2
import math
import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from patchify import patchify, unpatchify

from utils.constants import Constants
from utils.plot import visualize
from utils.logger import custom_logger
from utils.root_config import get_root_config


if __name__ == "__main__":

    ################################# Loading Variables and Paths from Config #################################

    ROOT, slice_config = get_root_config(__file__, Constants)

    # get the required variable values from config
    log_level = slice_config['vars']['log_level']
    file_type = slice_config['vars']['file_type']
    patch_size = slice_config['vars']['patch_size']  # size of each patch and window
    encoder = slice_config['vars']['encoder']        # the backbone/encoder of the model
    encoder_weights = slice_config['vars']['encoder_weights']
    classes = slice_config['vars']['test_classes']
    device = slice_config['vars']['device']

    # get the log file dir from config
    log_dir = ROOT / slice_config['dirs']['log_dir']
    # make the directory if it does not exist
    log_dir.mkdir(parents = True, exist_ok = True)
    # get the log file path
    log_path = log_dir / slice_config['vars']['test_log_name']
    # convert the path to string in a format compliant with the current OS
    log_path = log_path.as_posix()
    
    # initialize the logger
    logger = custom_logger("Land Cover Semantic Segmentation Test Logs", log_path, log_level)

    # get the dir of input images for inference from config
    img_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['test_dir'] / slice_config['dirs']['image_dir']
    img_dir = img_dir.as_posix()

    # get the dir of input masks for inference from config
    gt_mask_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['test_dir'] / slice_config['dirs']['mask_dir']
    gt_mask_dir = gt_mask_dir.as_posix()

    # get the model path from config
    model_name = slice_config['vars']['model_name']
    model_path = ROOT / slice_config['dirs']['model_dir'] / model_name
    model_path = model_path.as_posix()

    # get the predicted masks dir from config
    pred_mask_dir = ROOT / slice_config['dirs']['output_dir'] / slice_config['dirs']['pred_mask_dir']
    # make the directory if it does not exist
    pred_mask_dir.mkdir(parents = True, exist_ok = True)
    pred_mask_dir = pred_mask_dir.as_posix()

    # get the prediction plots dir from config
    pred_plot_dir = ROOT / slice_config['dirs']['output_dir'] / slice_config['dirs']['pred_plot_dir']
    # make the directory if it does not exist
    pred_plot_dir.mkdir(parents = True, exist_ok = True)
    pred_plot_dir = pred_plot_dir.as_posix()

    ###########################################################################################################

    ####################################### Functional Part of Program ########################################

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    model = torch.load(model_path, map_location=torch.device(device))

    class_values = [Constants.CLASSES.value.index(cls.lower()) for cls in classes]
    
    img_list = list(filter(lambda x:x.endswith((file_type)), os.listdir(img_dir)))

    print(f"\nTotal images found to test: {len(img_list)}")
    logger.info(f"Total images found to test: {len(img_list)}")

    try:
        for filename in img_list:

            print(f"\nPreparing image and ground truth mask file {filename}...")
            logger.info(f"Preparing image and ground truth mask file {filename}...")

            # reading image
            try:
                image = cv2.imread(os.path.join(img_dir, filename), 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Could not read image file {filename}!")
                raise e

            # reading ground truth mask
            try:
                gt_mask = cv2.imread(os.path.join(gt_mask_dir, filename), 0)
                # filter classes
                gt_masks = [(gt_mask == v) for v in class_values]
                gt_mask = np.stack(gt_masks, axis=-1).astype('float')
                gt_mask = gt_mask.argmax(2)
            except Exception as e:
                logger.error(f"Could not read ground truth mask file {filename}!")
                raise e

            # padding image to be perfectly divisible by patch_size
            try:
                pad_height = (math.ceil(image.shape[0] / patch_size) * patch_size) - image.shape[0]
                pad_width = (math.ceil(image.shape[1] / patch_size) * patch_size) - image.shape[1]
                padded_shape = ((0, pad_height), (0, pad_width), (0, 0))
                image_padded = np.pad(image, padded_shape, mode='reflect')
            except Exception as e:
                logger.error("Could not pad the image!")
                raise e

            # dividing image into patches according to patch_size in overlapping mode to have smooth reconstruction of predicted mask patches
            try:
                patches = patchify(image_padded, (patch_size, patch_size, 3), step=patch_size//2)[:, :, 0, :, :, :]
                mask_patches = np.empty(patches.shape[:-1], dtype=patches.dtype)
            except Exception as e:
                logger.error("Could not patchify the image!")
                raise e

            print("\nImage and ground truth mask preparation done successfully!")
            logger.info("Image and ground truth mask preparation done successfully!")

            # model prediction
            print(f"\nPredicting image file {filename}...")
            logger.info(f"Predicting image file {filename}...")
            try:
                for i in tqdm(range(0, patches.shape[0])):
                    for j in range(0, patches.shape[1]):
                        img_patch = preprocessing_fn(patches[i, j, :, :, :])
                        img_patch = img_patch.transpose(2, 0, 1).astype('float32')
                        x_tensor = torch.from_numpy(img_patch).to(device).unsqueeze(0)
                        pred_mask = model.predict(x_tensor)
                        pred_mask = pred_mask.squeeze().cpu().numpy().round()
                        pred_mask = pred_mask.transpose(1, 2, 0)
                        pred_mask = pred_mask.argmax(2)
                        mask_patches[i, j, :, :] = pred_mask
            except Exception as e:
                logger.error(f"Could not predict image file {filename}!")
                raise e

            # unpatch
            try:
                pred_mask = unpatchify(mask_patches, image_padded.shape[:-1])
            except Exception as e:
                logger.error("Could not unpatchify predicted mask patches!")
                raise e
            
            # unpad
            try:
                pred_mask = pred_mask[:image.shape[0], :image.shape[1]]
            except Exception as e:
                logger.error("Could not unpad reconstructed predicted mask!")
                raise e
            
            # classes found
            try:
                classes_found = []
                for cls in np.unique(pred_mask):
                    classes_found.append(Constants.CLASSES.value[cls])
                print(f"Total classes found in the predicted mask: {classes_found}")
                logger.info(f"Total classes found in the predicted mask: {classes_found}")
            except Exception as e:
                logger.error("Could not find classes in the predicted mask!")
                raise e
            
            # filter classes
            try:
                pred_masks = [(pred_mask == v) for v in class_values]
                pred_mask = np.stack(pred_masks, axis=-1).astype('float')
                pred_mask = pred_mask.argmax(2)
                print(f"Classes present in the predicted mask after filtering according to user input of 'test_classes': {classes}")
                logger.info(f"Classes present in the predicted mask after filtering according to user input of 'test_classes': {classes}")
            except Exception as e:
                logger.error("Could not filter user given classes from the predicted mask!")
                raise e
            
            try:
                cv2.imwrite(os.path.join(pred_mask_dir, filename), pred_mask)
                print("Predicted mask written successfully!")
                logger.info("Predicted mask written successfully!")
            except Exception as e:
                logger.error("Could not write the predicted mask!")
                raise e

            try:
                plot_fig = visualize(
                    image=image, 
                    ground_truth_mask=gt_mask, 
                    predicted_mask=pred_mask
                )
                plot_fig.savefig(os.path.join(pred_plot_dir, filename.split('.')[0] + '.png'))
                print("Prediction plot saved successfully!")
                logger.info("Prediction plot saved successfully!")
            except Exception as e:
                logger.error("Could not plot the image, ground truth mask, and predicted mask!")
                raise e

    except Exception as e:
        logger.error("No images found in 'data/test/images' folder!")
        raise e

    ###########################################################################################################