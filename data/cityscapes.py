"""
dataset.py
Dataset creation and preprocessing
prepares dataloaders for cityscapes dataset.
"""

import os
import argparse

from PIL import Image
from torch.utils.data import Dataset


class Cityscapes(Dataset):
    "Cityscapes dataset"
    def __init__(self, root_dir, split="train", target_type="semantic",
                 image_transforms=None, mask_transforms=None):

        super().__init__()
        # Check if the split is defined.
        assert split in ("train", "val", "test"), "Undefined split"
        self.split = split
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.target_type = target_type
        if not isinstance(self.target_type, list):
            self.target_type = [self.target_type]

        # ./cityscapes/leftImg8bit/train/leftImg8bit/train <- mac/linux
        # .\\cityscapes\\leftImg8bit\\train\\leftImg8bit\\train <- windows
        self.images_split_base = os.path.join(root_dir, "leftImg8bit",
                                              self.split)
        self.annot_split_base = os.path.join(root_dir, "gtFine", self.split)

        # Recursively iterate over sub-directories to collect image paths
        self.image_paths = sorted([
            os.path.join(dir_path, file)
            for city in os.listdir(self.images_split_base)
            for dir_path, _, files in os.walk(
                os.path.join(self.images_split_base, city))
            for file in files])

        # zurich_000069_000019_leftImg8bit.png" -> "zurich_000069_000019"
        self.image_ids = [os.path.split(file)[1].replace("_leftImg8bit.png",
                                                         "")
                          for file in self.image_paths]

        # "zurich_000069_000019": "./cityscapes/leftImg8bit/train/zurich/\
        # zurich_000069_000019_leftImg8bit.png"
        self.id2image_path = dict(zip(self.image_ids, self.image_paths))

        # map user friendly names to actual notations
        target_type_map = {'semantic': 'labelIds',
                           'color': 'color',
                           'polygon': 'polygon',
                           'instance': 'instanceIds'}
        self.target_type = [target_type_map[target]
                            for target in self.target_type]
        # Retrieve annotation masks for each type.
        # result is a list of tuples of various masks.
        # [(semantic0, color0, ...), (semantic1, color1, ...), ...]
        self.annot_paths = list(zip(*[self.get_annotation_paths(type_=type_)
                                      for type_ in self.target_type]))

        print(f"Cityscapes ({self.split}):\t\
Loaded {len(self.image_paths)} images")

    def get_annotation_paths(self, type_):
        "Given the file type, retrieve the annotation"
        # Create buckets to store annotations masks
        self.annot_path_local = {'polygons': {},
                                 'color': {},
                                 'labelIds': {},
                                 'instanceIds': {}}
        # Iterate over sub dirs and retrieve all masks
        for city in os.listdir(self.annot_split_base):
            for dir_path, _, files in os.walk(os.path.join(
               self.annot_split_base, city)):
                for file in files:
                    file_path = os.path.join(dir_path, file)
                    file_id, cat = os.path.splitext(
                        os.path.split(file_path)[1]
                        .replace('_gtFine_', '|'))[0].split('|')
                    self.annot_path_local[cat][file_id] = file_path

        return [self.annot_path_local[type_][id_] for id_ in self.image_ids]

    def __repr__(self):
        return "Custom Cityscapes dataset"

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        for idx_ in range(self.__len__()):
            yield self.__getitem__(idx_)

    def get_source_image_paths(self, id_=False):
        "Given the id, return the full path of an image"
        if not id_:
            return self.image_paths
        return self.id2image_path[id_]

    def get_image(self, id_):
        "Given the id, return PIL Image"
        return Image.open(self.id2image_path[id_])

    def __getitem__(self, index):
        # Open the image from image paths at given index
        img = Image.open(self.image_paths[index])
        # Obtain the corresponding annotations, read them using PIL.
        annots = [Image.open(target_item_path)
                  for target_item_path in self.annot_paths[index]]

        # Pass through transformations if any.
        if self.image_transforms:
            img = self.image_transforms(img)

        if self.mask_transforms:
            annots = [self.mask_transforms(annot) for annot in annots]

        # If the target is only 1 type, take it out of the tuple.
        if len(annots) == 1:
            annots = annots[0]

        return img, annots


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="root_path")
    parser.add_argument(
        "--path",
        nargs="?",
        type=str,
        default="/Users/ajaydyavathi/My_Cityscapes/dataset/cityscapes/",
        help="Specify the root directory to cityscapes dataset"
    )
    parser.add_argument(
        "--split",
        nargs="?",
        type=str,
        default="train",
        help="Specify the split"
    )

    args = parser.parse_args()
    path = args.path
    train_dataset = Cityscapes(args.path, args.split,
                               target_type="color")

    N_PICS = 3
    plt.figure(figsize=(10, 8))
    for i in range(N_PICS):
        index_ = random.randint(0, len(train_dataset))
        image, color_mask = train_dataset[index_]
        plt.subplot(N_PICS, 2, 2*i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(N_PICS, 2, 2*i + 2)
        plt.imshow(color_mask)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
