from .utils import MinMaxResize
from torchvision import transforms
from .randaug import RandAugment, Eq_Blur, RandAugment2


def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),                          # resize image
            transforms.ToTensor(),                                              # convert image to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),    # normalise image
        ]
    )


def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))                                 # additionally apply any two augmentations from RandAugment
    return trs

def pixelbert_transform_eq_blur(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    trs.transforms.insert(0, Eq_Blur())                                         # additionally apply histogram eq. and gaussian blur augmentations
    return trs

def pixelbert_transform_randaug_v2(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    trs.transforms.insert(0, RandAugment2(2, 9))                                # additionally apply any two augmentations from RandAugment2
    return trs

def resnet_transform(size=0):
    return transforms.Compose(
        [
            transforms.Resize(256),                                             # resize image
            transforms.CenterCrop(224),                                         # central crop
            transforms.ToTensor(),                                              # convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalise image
        ]
    )