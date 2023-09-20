"""
Dataset creation and preprocessing
prepares dataloaders for cityscapes dataset.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from labels_NYIT import labels

MEAN = np.array([72.55410438, 81.93415236, 71.4297832]) / 255
STD = np.array([51.04788791, 51.76003371, 50.94766331]) / 255

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CustomCityScapes(Dataset):
    "Cityscapes dataset"
    def __init__(self, root_dir, split="train", target_type="semantic",
                 image_transforms=None, mask_transforms=None):
        assert split in ("train", "val", "test"), "Undefined split"
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.target_type = target_type
        if not isinstance(self.target_type, list):
            self.target_type = list(self.target_type)

        # ./cityscapes/leftImg8bit/train/leftImg8bit/train <- mac/linux
        # .\\cityscapes\\leftImg8bit\\train\\leftImg8bit\\train <- windows
        self.images_split_path = os.path.join(root_dir, "leftImg8bit", split)
        self.annot_split_path = os.path.join(root_dir, "gtFine", split)

        self.image_paths_only = sorted([
            os.path.join(dir_path, file)
            for city in os.listdir(self.images_split_path)
            for dir_path, _, files in os.walk(
                os.path.join(self.images_split_path, city))
            for file in files])

        # "./cityscapes/leftImg8bit/train/zurich/\
        # zurich_000069_000019_leftImg8bit.png" -> "zurich_000069_000019"
        self.image_ids = [os.path.split(file)[1].replace("_leftImg8bit.png",
                                                         "")
                          for file in self.image_paths_only]
        # "zurich_000069_000019": "./cityscapes/leftImg8bit/train/zurich/\
        # zurich_000069_000019_leftImg8bit.png"
        self.image_paths = dict(zip(self.image_ids, self.image_paths_only))

        target_type_map = {'semantic': 'labelIds',
                           'color': 'color',
                           'polygons': 'polygon',
                           'instance': 'instanceIds'}
        self.target_type = [target_type_map[target]
                            for target in self.target_type]
        self.annot_paths = list(zip(*[self.get_annotation_paths(type_=type_)
                                      for type_ in self.target_type]))

    def __iter__(self):
        for index_ in range(self.__len__()):
            yield self.__getitem__(index_)

    def __getitem__(self, index):
        img = Image.open(self.image_paths_only[index])
        annots = [Image.open(target_item_path)
                  for target_item_path in self.annot_paths[index]]

        if self.image_transforms:
            img = self.image_transforms(img)

        if self.mask_transforms:
            annots = [self.mask_transforms(annot) for annot in annots]

        return img, annots

    def __repr__(self):
        return "Custom Cityscapes dataset"

    def __len__(self):
        return len(self.image_paths)

    def get_source_image_paths(self, id_=False):
        "Given the id, return the full path of an image"
        if not id_:
            return self.image_paths
        return self.image_paths[id_]

    def get_image(self, id_):
        "Given the id, return PIL Image"
        return Image.open(self.image_paths[id_])

    def get_annotation_paths(self, type_):
        "Given the file type, retrieve the annotation"
        self.annot_path_local = {'polygons': {},
                                 'color': {},
                                 'labelIds': {},
                                 'instanceIds': {}}
        for city in os.listdir(self.annot_split_path):
            for dir_path, _, files in os.walk(os.path.join(
               self.annot_split_path, city)):
                for file in files:
                    file_path = os.path.join(dir_path, file)
                    file_id, cat = os.path.splitext(
                        os.path.split(file_path)[1]
                        .replace('_gtFine_', '|'))[0].split('|')
                    self.annot_path_local[cat][file_id] = file_path

        return [self.annot_path_local[type_][id_] for id_ in self.image_ids]
# Dataset class ends here


def crop_eco_vehicle(image, bottom=-180):
    "Crop the POV vehicle"
    width, height = image.size
    return image.crop((0, 0, width, height+bottom))


def map_labels(image):
    "Map IDs to train ids for semantic masks"
    id2trainId = {label.id: label.trainId for label in labels}
    image_np = np.array(image)
    for id_, trainId_ in id2trainId.items():
        image_np[image_np == id_] = trainId_

    return image_np


# Transformations for training data
train_transforms = transforms.Compose([
    transforms.Lambda(crop_eco_vehicle),
    transforms.Resize((256, 512), interpolation=Image.NEAREST),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Transformations for evaluation (val/test) data
eval_transforms = transforms.Compose([
    transforms.Lambda(crop_eco_vehicle),
    transforms.Resize((256, 512), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Transformations for all annotation
annot_transforms = transforms.Compose([
    transforms.Lambda(crop_eco_vehicle),
    transforms.Resize((256, 512), interpolation=Image.NEAREST),
    transforms.Lambda(map_labels),
    transforms.ToTensor()
])

# Only resizing for debugging
resize = transforms.Compose([transforms.Resize((256, 512),
                                               interpolation=Image.NEAREST),
                             transforms.ToTensor()])

# Create training dataset
train_dataset = CustomCityScapes("./cityscapes", split="train",
                                 target_type=["color", "semantic"],
                                 image_transforms=train_transforms,
                                 mask_transforms=annot_transforms)
# Create validation dataset
val_dataset = CustomCityScapes("./cityscapes", split="val",
                               target_type=["color", "semantic"],
                               image_transforms=train_transforms,
                               mask_transforms=annot_transforms)
# Create testing dataset
test_dataset = CustomCityScapes("./cityscapes", split="test",
                                target_type=["color", "semantic"],
                                image_transforms=train_transforms,
                                mask_transforms=annot_transforms)
# Create debug dataset with just resizing
debug_dataset_resize_ = CustomCityScapes("./cityscapes", split="train",
                                         target_type=["color", "semantic"],
                                         image_transforms=resize,
                                         mask_transforms=resize)

# Create corresponding dataloaders
BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
debug_loader_resize_ = DataLoader(debug_dataset_resize_,
                                  batch_size=BATCH_SIZE, shuffle=False)

# Plotting - generating samples
# Iterating training dataloader
im_b0, (col_b0, sem_b0) = next(iter(train_loader))
debug_im_b0, (debug_col_b0, debug_sem_b0) = next(iter(debug_loader_resize_))

NUM_SAMPLES = 4
sample_images_plot = plt.figure(figsize=(15, 8))
subplot_idx = 1
for debug_im_, debug_col_, im_, col_ in zip(debug_im_b0[:NUM_SAMPLES],
                                            debug_col_b0[:NUM_SAMPLES],
                                            im_b0[:NUM_SAMPLES],
                                            col_b0[:NUM_SAMPLES]):
    # Debug Image
    plt.subplot(NUM_SAMPLES, 4, subplot_idx)
    subplot_idx += 1
    debug_im_ = np.array(debug_im_).transpose(1, 2, 0)
    plt.imshow(debug_im_)
    plt.axis("off")
    # Debug Color info
    plt.subplot(NUM_SAMPLES, 4, subplot_idx)
    subplot_idx += 1
    debug_col_ = np.array(debug_col_).transpose(1, 2, 0)
    plt.imshow(debug_col_)
    plt.axis("off")

    # Image
    plt.subplot(NUM_SAMPLES, 4, subplot_idx)
    subplot_idx += 1
    im_ = np.array(im_).transpose(1, 2, 0)
    plt.imshow(im_)
    plt.axis("off")
    # Color info
    plt.subplot(NUM_SAMPLES, 4, subplot_idx)
    subplot_idx += 1
    col_ = np.array(col_).transpose(1, 2, 0)
    plt.imshow(col_)
    plt.axis("off")

plt.tight_layout()
