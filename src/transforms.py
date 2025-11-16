import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform(img_size=224, grayscale_to_rgb=True, imagenet_norm=True):
    ops = [T.Resize((img_size, img_size))]
    if grayscale_to_rgb:
        ops.append(T.Grayscale(num_output_channels=3))
    ops.append(T.ToTensor())
    if imagenet_norm:
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(ops)

def get_eval_transform(img_size=224, grayscale_to_rgb=True, imagenet_norm=True):
    return get_train_transform(img_size, grayscale_to_rgb, imagenet_norm)
