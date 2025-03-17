import os
import shutil
import torch
import splitfolders
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

from utils.constants import Constants
from utils.logger import custom_logger
from utils.root_config import get_root_config
from utils.patching import patching, discard_useless_patches
from utils.preprocess import get_training_augmentation, get_preprocessing
from utils.dataset import SegmentationDataset


if __name__ == "__main__":

    ################################# Loading Variables and Paths from Config #################################

    ROOT, slice_config = get_root_config(__file__, Constants)

    # get the required variable values from config
    log_level = slice_config['vars']['log_level']
    file_type = slice_config['vars']['file_type']
    patch_size = slice_config['vars']['patch_size']
    discard_rate = slice_config['vars']['discard_rate']
    batch_size = slice_config['vars']['batch_size']
    model_arch = slice_config['vars']['model_arch']
    encoder = slice_config['vars']['encoder']
    encoder_weights = slice_config['vars']['encoder_weights']
    activation = slice_config['vars']['activation']
    optimizer_choice = slice_config['vars']['optimizer_choice']
    init_lr = slice_config['vars']['init_lr']
    lr_reduce_factor = slice_config['vars']['reduce_lr_by_factor']
    lr_reduce_patience = slice_config['vars']['patience_epochs_before_reducing_lr']
    lr_reduce_threshold = slice_config['vars']['lr_reduce_threshold']
    minimum_lr = slice_config['vars']['minimum_lr']
    epochs = slice_config['vars']['epochs']
    all_classes = slice_config['vars']['all_classes']
    classes = slice_config['vars']['train_classes']
    device = slice_config['vars']['device']

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    # get the log file dir from config
    log_dir = ROOT / slice_config['dirs']['log_dir']
    # make the directory if it does not exist
    log_dir.mkdir(parents = True, exist_ok = True)
    # get the log file path
    log_path = log_dir / slice_config['vars']['train_log_name']
    # convert the path to string in a format compliant with the current OS
    log_path = log_path.as_posix()
    
    # initialize the logger
    logger = custom_logger("Land Cover Semantic Segmentation Train Logs", log_path, log_level)

    # get the train dir from config
    train_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir']
    train_dir = train_dir.as_posix()

    # get the dir of input images for training from config
    img_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir'] / slice_config['dirs']['image_dir']
    img_dir = img_dir.as_posix()

    # get the dir of input masks for training from config
    mask_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir'] / slice_config['dirs']['mask_dir']
    mask_dir = mask_dir.as_posix()

    # get the model dir from config
    model_dir = ROOT / slice_config['dirs']['model_dir']
    model_dir = model_dir.as_posix()

    ###########################################################################################################

    ####################################### Functional Part of Program ########################################

    # create the directory to save the patches of images and masks
    patches_dir = os.path.join(train_dir, f"patches_{patch_size}")
    patches_img_dir = os.path.join(patches_dir, "images")
    os.makedirs(patches_img_dir, exist_ok=True)
    patches_mask_dir = os.path.join(patches_dir, "masks")
    os.makedirs(patches_mask_dir, exist_ok=True)

    try:
        print("\nDividing images into patches...")
        patching(img_dir, patches_img_dir, file_type, patch_size)
        print("\nDivided images into patches successfully!")
        logger.info("Divided images into patches successfully!")
    except Exception as e:
        logger.error("Failed to divide images into patches!")
        raise e

    try:
        print("\nDividing masks into patches...")
        patching(mask_dir, patches_mask_dir, file_type, patch_size)
        print("\nDivided masks into patches successfully!")
        logger.info("Divided masks into patches successfully!")
    except Exception as e:
        logger.error("Failed to divide masks into patches!")
        raise e

    try:
        print("\nDiscarding useless patches where background covers more than 95% of the area...")
        discard_useless_patches(patches_img_dir, patches_mask_dir, discard_rate)
        print("\nDiscarded unused patches successfully!")
        logger.info("Discarded unused patches successfully!")
    except Exception as e:
        logger.error("Failed to discard unused patches!")
        raise e

    output_folder = os.path.join(patches_dir, "train_val_test")
    os.makedirs(output_folder, exist_ok=True)

    try:
        print("\nSplitting training and validation data...")
        # Split with a ratio.
        # To split into training, validation, and testing set, set a tuple to `ratio`, i.e, `(.8, .1, .1)`.
        splitfolders.ratio(patches_dir, output=output_folder, seed=42, ratio=(.8, .2), group_prefix=None, move=False) # splitting in training and validation only
        print("\nTraining and validation data split successfully!")
        logger.info("Training and validation data split successfully!")
    except Exception as e:
        logger.error("Failed to split training and validation data!")
        raise e

    train_dir = os.path.join(output_folder, "train")
    val_dir = os.path.join(output_folder, "val")
    # test_dir = os.path.join(output_folder, "test")
    x_train_dir = os.path.join(train_dir, "images")
    y_train_dir = os.path.join(train_dir, "masks")
    x_val_dir = os.path.join(val_dir, "images")
    y_val_dir = os.path.join(val_dir, "masks")

    try:
        # create segmentation model with pretrained encoder
        smp_model = getattr(smp, model_arch)
        model = smp_model(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
        )
        print("\nBuilt the model successfully!")
        logger.info("Built the model successfully!")
    except Exception as e:
        logger.error("Failed to build the model!")
        raise e

    try:
        train_dataset = SegmentationDataset(
            x_train_dir,
            y_train_dir,
            all_classes=all_classes,
            classes=classes,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        val_dataset = SegmentationDataset(
            x_val_dir,
            y_val_dir,
            all_classes=all_classes,
            classes=classes,
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        print("\nInitialized training and validation datasets and dataloaders!")
        logger.info("Initialized training and validation datasets and dataloaders!")
    except Exception as e:
        logger.error("Failed to initialize training and validation datasets and dataloaders!")
        raise e

    try:
        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5)
        ]
        print("\nInitialized the loss and evaluation metrics!")
        logger.info("Initialized the loss and evaluation metrics!")
    except Exception as e:
        logger.error("Failed to initialize loss and evaluation metrics")
        raise e

    try:
        torch_optimizer = getattr(torch.optim, optimizer_choice)
        optimizer = torch_optimizer([
            dict(params=model.parameters(), lr=init_lr),
        ])
        # Reduce LR on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = 'min', 
            factor = lr_reduce_factor, 
            patience = lr_reduce_patience, 
            threshold = lr_reduce_threshold, 
            min_lr = minimum_lr)
        
        print("\nInitialized the optimizer!")
        logger.info("Initialized the optimizer!")
    except Exception as e:
        logger.error("Failed to initialize the optimizer!")
        raise e

    try:
        # create epoch runners
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True,
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=True,
        )
        print("\nInitialized epoch runners!")
        logger.info("Initialized epoch runners!")
    except Exception as e:
        logger.error("Failed to initialize epoch runners!")
        raise e

    try:
        print("\nStarting model training...")
        max_score = 0
        for i in range(0, epochs):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # Do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, f'{model_dir}/landcover_{model_arch}_{encoder}_{optimizer_choice}_epochs{i}_patch{patch_size}_batch{batch_size}.pth')
                print('Current best model saved!')

            scheduler.step(valid_logs['dice_loss'])
        print("\nModel training finished!")
        logger.info("Model training finished!")
    except Exception as e:
        logger.error("Failed to train the model!")
        raise e

    shutil.rmtree(patches_dir)

    ###########################################################################################################