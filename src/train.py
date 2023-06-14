import os
import splitfolders
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

from utils.patching import patching, discard_useless_patches
from utils.preprocess import get_training_augmentation, get_preprocessing
from utils.dataset import SegmentationDataset

########## config ###########

BATCH_SIZE = 16
ENCODER = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['background', 'building', 'woodland', 'water']   # not training on 'road' class since it's instances in the data is too less
ACTIVATION = 'softmax2d'    # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cpu'
EPOCHS = 20

root_directory = "data/train"

img_dir = os.path.join(root_directory, "images")
mask_dir = os.path.join(root_directory, "masks")

patch_size = 512

patches_img_dir = os.path.join(f"patches_{patch_size}", "images")
patches_img_dir = os.path.join(root_directory, patches_img_dir)
os.makedirs(patches_img_dir, exist_ok=True)
patches_mask_dir = os.path.join(f"patches_{patch_size}", "masks")
patches_mask_dir = os.path.join(root_directory, patches_mask_dir)
os.makedirs(patches_mask_dir, exist_ok=True)

model_dir = "models"

#############################

print()
print("Dividing images into patches...")
patching(img_dir, patches_img_dir, patch_size)
print("Dividing images into patches completed successfull!")

print()
print("Dividing masks into patches...")
patching(mask_dir, patches_mask_dir, patch_size)
print("Dividing masks into patches completed successfull!")

discard_useless_patches(patches_img_dir, patches_mask_dir)



input_folder = patches_img_dir.strip("images")
print(input_folder)
output_folder = os.path.join(root_directory, "train_val_test")
print(output_folder)

os.makedirs(output_folder, exist_ok=True)

# Split with a ratio.
# To split into training, validation, and testing set, set a tuple to `ratio`, i.e, `(.8, .1, .1)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .2), group_prefix=None, move=False) # splitting in training and validation only

train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")
# test_dir = os.path.join(output_folder, "test")

x_train_dir = os.path.join(train_dir, "images")
y_train_dir = os.path.join(train_dir, "masks")

x_val_dir = os.path.join(val_dir, "images")
y_val_dir = os.path.join(val_dir, "masks")






# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)



train_dataset = SegmentationDataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

val_dataset = SegmentationDataset(
    x_val_dir,
    y_val_dir,
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5)
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0003),
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)


max_score = 0

for i in range(0, EPOCHS):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # Do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, f'{model_dir}/landcover_unet_{ENCODER}_epochs{i}_patch{patch_size}_batch{BATCH_SIZE}.pth')
        print('Model saved!')

    scheduler.step(valid_logs['dice_loss'])