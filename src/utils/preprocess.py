import albumentations as album

def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
        # album.ShiftScaleRotate(scale_limit=1.5, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),
        # album.GaussNoise(p=0.2),
        # album.Perspective(p=0.5),
        # album.OneOf(
        #     [
        #         album.CLAHE(p=1),
        #         album.RandomBrightnessContrast(p=1),
        #         album.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),
        # album.OneOf(
        #     [
        #         album.Sharpen(p=1),
        #         album.Blur(blur_limit=3, p=1),
        #         album.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return album.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albummentations.Compose

    """
    _transform = [
        album.Lambda(image=preprocessing_fn),
        album.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return album.Compose(_transform)