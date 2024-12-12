from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
    pixelbert_transform_eq_blur,
    pixelbert_transform_randaug_v2,
    resnet_transform
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "pixelbert_eq_blur": pixelbert_transform_eq_blur,
    "pixelbert_randaug_v2": pixelbert_transform_randaug_v2,
    "resnet": resnet_transform,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
