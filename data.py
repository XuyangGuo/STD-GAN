import os
import random
import numpy as np
import torch
import torchvision.transforms as tform
from torchvision.transforms import functional as F
from PIL import Image


# ==============================================================================
# =                                   dataset                                  =
# ==============================================================================

# ======================================
# =               general              =
# ======================================


class AttributeDataset():
    """A general image-attributes dataset class."""

    def __init__(self, img_dir, att_file, split, size, use_atts=None, img_transform=None, att_transform=None,
                 pair_crop=False, test_start=182000):  # pair_crop=True, x0 and x1 will have the same ramdom crop

        def split_data(data):
            if split == 'train':
                return data[:182000]
            elif split == 'val' or split == 'test':
                return data[test_start:]

        self.use_atts = use_atts
        self.img_transform = img_transform
        self.att_transform = att_transform
        self.split = split
        self.size = size
        self.pair_crop = pair_crop

        self.att_names = list(np.genfromtxt(att_file, skip_header=1, max_rows=1, dtype=np.str))

        if img_dir is not None:
            img_names = np.genfromtxt(att_file, skip_header=2, usecols=0, dtype=np.str)
            self.img_paths = np.array([os.path.join(img_dir, img_name) for img_name in img_names])
        else:
            self.img_paths = []

        use_cols = [self.att_names.index(att) + 1 for att in use_atts] if use_atts else None
        self.att_labels = np.genfromtxt(att_file, skip_header=2, usecols=use_cols,
                                        dtype=np.float32).reshape(-1, len(use_cols))

        for check_att in ['No_Beard', 'Black_Hair', 'Male', 'Young']:
            if check_att in use_atts:
                self.att_labels[:, use_atts.index(check_att)] *= -1

        self.img_paths = split_data(self.img_paths)
        self.att_labels = split_data(self.att_labels)

        filter_idx = np.ones_like(self.att_labels, dtype=np.bool)  # used to filter some specific attributes
        # Mustache Goatee  gil!!!
        for check_att in ['Mustache', 'Goatee', 'No_Beard', 'Sideburns']:
            if check_att in use_atts:
                filter_idx[:, use_atts.index(check_att)] = split_data(
                    np.genfromtxt(att_file, skip_header=2, usecols=[self.att_names.index('Male') + 1],
                                  dtype=np.float32)) == 1

        # makeup gil!!!
        for check_att in ['Wearing_Lipstick', 'Heavy_Makeup', 'Wavy_Hair']:
            if check_att in use_atts:
                filter_idx[:, use_atts.index(check_att)] = split_data(
                    np.genfromtxt(att_file, skip_header=2, usecols=[self.att_names.index('Male') + 1],
                                  dtype=np.float32)) == -1

        for check_att in ['Brown_Hair', 'Blond_Hair', 'Gray_Hair']:
            if check_att in use_atts:
                black_hair_cols = split_data(np.genfromtxt(
                    att_file, skip_header=2, usecols=[self.att_names.index('Black_Hair') + 1]))
                filter_idx[:, use_atts.index(check_att)] = np.logical_or(
                    np.logical_and(self.att_labels[:, use_atts.index(check_att)] == -1, black_hair_cols == 1),
                    np.logical_and(self.att_labels[:, use_atts.index(check_att)] == 1, black_hair_cols == -1))

        if 'Black_Hair' in use_atts:
            brown = split_data(np.genfromtxt(
                att_file, skip_header=2, usecols=[self.att_names.index('Brown_Hair') + 1]))
            blond = split_data(np.genfromtxt(
                att_file, skip_header=2, usecols=[self.att_names.index('Blond_Hair') + 1]))
            gray = split_data(np.genfromtxt(
                att_file, skip_header=2, usecols=[self.att_names.index('Gray_Hair') + 1]))

            brown_or_blond = np.logical_or(np.logical_or(brown == 1, blond == 1), gray == 1)
            not_brown_and_not_blond = np.logical_and(np.logical_and(brown == -1, blond == -1), gray == -1)
            filter_idx[:, use_atts.index('Black_Hair')] = np.logical_or(
                np.logical_and(self.att_labels[:, use_atts.index('Black_Hair')] == -1, not_brown_and_not_blond),
                np.logical_and(self.att_labels[:, use_atts.index('Black_Hair')] == 1, brown_or_blond))

        att_labels, img_paths, self.positive_indexes, self.negative_indexes = [], [], [], []
        for i in range(len(use_atts)):
            label = self.att_labels[filter_idx[:, i] == 1, i]
            att_labels.append(self.att_labels[filter_idx[:, i] == 1, :])
            if len(self.img_paths) > 0:
                img_paths.append(self.img_paths[filter_idx[:, i] == 1])
            self.positive_indexes.append(np.where(label == 1)[0])
            self.negative_indexes.append(np.where(label == -1)[0])
        self.att_labels = att_labels
        self.img_paths = img_paths

        self.total_index = 0

    # ******
    # index=-1 means random sample negative data
    # ******
    def get_by_index(self, index, att_index):
        def get_by_index(index):
            att_label = torch.FloatTensor([self.att_labels[att_index][index]])
            if self.att_transform:
                att_label = self.att_transform(att_label)

            if len(self.img_paths) > 0:
                img = tform.ToTensor()(Image.open(self.img_paths[att_index][index]))  # [0, 1.0] tensor
                if self.img_transform:
                    img = self.img_transform(img)
                return img, att_label
            else:
                return att_label

        if self.pair_crop:
            x0, y0 = random.randint(14, 33), random.randint(0, 7)
            flip_p = random.uniform(0, 1)
            self.img_transform = tform.Compose([
                # crop face area 190 * 178 from 178x218
                # tform.Lambda(lambda x: x[:, 14:204, :]),  # origin
                # random crop, flip and resize on PLI image
                tform.Lambda(lambda x: x[:, x0:x0 + 170, y0:y0 + 170]),  # elegant crop
                tform.ToPILImage(),
                tform.Resize(self.size, Image.BICUBIC),  # elegant crop
                tform.Lambda(lambda x: F.hflip(x) if flip_p > 0.5 else x),
                # back to tensor
                tform.ToTensor(),
                tform.Lambda(lambda x: x * 2 - 1)  # to [-1, 1]
            ])

        if index == -1:
            index = random.randint(0, len(self.negative_indexes[att_index]) - 1)

        result0 = get_by_index(self.negative_indexes[att_index][index % len(self.negative_indexes[att_index])])
        if self.split == 'test' or self.split == 'val':
            result1 = get_by_index(self.positive_indexes[att_index][index % len(self.positive_indexes[att_index])])
        else:
            result1 = get_by_index(self.positive_indexes[att_index][random.randint(
                0, len(self.positive_indexes[att_index]) - 1)])
        return result0, result1

    def get_batch_randomly_with_att_index(self, batch_size, att_index):
        x0s, x1s, x0ls, x1ls = [], [], [], []
        for i in range(batch_size):
            (x0, x0l), (x1, x1l) = self.get_by_index(-1, att_index)
            x0s.append(x0.unsqueeze(dim=0))
            x1s.append(x1.unsqueeze(dim=0))
            x0ls.append(x0l)
            x1ls.append(x1l)
        return x0s, x0ls, x1s, x1ls

    def get_batch(self, batch_size):
        result = []
        for a in range(len(self.use_atts)):
            x0s, x1s = [], []
            for i in range(batch_size):
                ((x0, _), (x1, _)) = self.get_by_index(self.total_index + i, a)
                x0s.append(x0.unsqueeze(dim=0))
                x1s.append(x1.unsqueeze(dim=0))
            result.append([torch.cat(x0s, dim=0), torch.cat(x1s, dim=0)])
        self.total_index += batch_size
        return result

    def get_file(self, index, mask, att):
        if mask == 1:
            index = self.positive_indexes[att][index]
        else:
            index = self.negative_indexes[att][index]
        return self.img_paths[att][index]

    def get_image_num_with_attribute(self, mask, att):
        if mask == 1:
            return len(self.positive_indexes[att])
        else:
            return len(self.negative_indexes[att])


# ======================================
# =          CelebA  functions         =
# ======================================

def get_dataset_celeba(img_dir, att_file, use_atts, size, well_cropped=False, split='train', pair=False,
                       pair_crop=False, test_start=182000):
    # easy to be adapted to other dataset
    train_img_transform = tform.Compose([
        # crop face area 190 * 178 from 178x218
        tform.Lambda(lambda x: x[:, 14:204, :]),  # origin
        # random crop, flip and resize on PLI image
        tform.ToPILImage(),
        tform.RandomCrop(170),  # origin
        # tform.CenterCrop(170),
        # tform.CenterCrop(178),
        tform.Resize(size, Image.BICUBIC),
        tform.RandomHorizontalFlip(),
        # back to tensor
        tform.ToTensor(),
        tform.Lambda(lambda x: x * 2 - 1)  # to [-1, 1]
    ])

    val_test_img_transform = tform.Compose([
        # crop face area 190 * 178
        # tform.Lambda(lambda x: x[:, 14:204, :]),
        # center crop and resize on PLI image
        tform.ToPILImage(),
        tform.CenterCrop(170),  # origin # elegant crop
        # tform.CenterCrop(178),
        tform.Resize(size, Image.BICUBIC),  # elegant crop
        # back to tensor
        tform.ToTensor(),
        tform.Lambda(lambda x: x * 2 - 1)  # to [-1, 1]
    ])

    well_cropped_img_transform = tform.Compose([
        # resize on PLI image
        tform.ToPILImage(),
        tform.Resize(size, Image.BICUBIC),
        # back to tensor
        tform.ToTensor(),
        tform.Lambda(lambda x: x * 2 - 1)  # to [-1, 1]
    ])

    att_transform = tform.Lambda(lambda x: (x + 1) / 2)  # {0.0, 1.0}

    if well_cropped:
        img_transform = well_cropped_img_transform
    elif split == 'train':
        img_transform = train_img_transform
    else:
        img_transform = val_test_img_transform

    dataset = AttributeDataset(img_dir, att_file, split, size, use_atts, img_transform, att_transform,
                               pair_crop=pair_crop, test_start=test_start)
    return dataset


class Celeba_labels():
    def __init__(self, img_dir, att_file):
        self.att_names = list(np.genfromtxt(att_file, skip_header=1, max_rows=1, dtype=np.str))
        self.test_start = 190000
        img_names = np.genfromtxt(att_file, skip_header=2, usecols=0, dtype=np.str)[self.test_start:]
        self.img_paths = np.array([os.path.join(img_dir, img_name) for img_name in img_names])

        all_use_cols = [i for i in range(1, len(self.att_names) + 1)]
        self.all_labels = np.genfromtxt(att_file, skip_header=2, usecols=all_use_cols,
                                        dtype=np.float32).reshape(-1, len(all_use_cols))[self.test_start:]

    def filter(self, *filter_dicts):
        final_result = np.zeros(self.all_labels.shape[0], dtype='bool')
        for filter_dict in filter_dicts:
            filter_result = np.ones(self.all_labels.shape[0], dtype='bool')
            for att in filter_dict:
                att_idx_in_all = self.att_names.index(att)
                filter_result = np.logical_and(filter_result, self.all_labels[:, att_idx_in_all] == filter_dict[att])
            final_result = np.logical_or(final_result, filter_result)

        return self.img_paths[final_result]


if __name__ == '__main__':
    import config, sys

    cfg = config.get_config(sys.argv[1])
    train_dataset = get_dataset_celeba(img_dir=cfg.img_dir, att_file=cfg.att_file, use_atts=cfg.use_atts,
                                       well_cropped=cfg.well_cropped, size=cfg.img_size, split='train',
                                       pair_crop=True)

    r = train_dataset.get_batch(cfg.batch_size)
    pass
