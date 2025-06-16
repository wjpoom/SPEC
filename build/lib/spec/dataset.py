import os
import json
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Image2TextDataset(Dataset):
    def __init__(self, subset_root, image_preprocess=None):
        """
        Args:
            subset_root: the path to the root dir of a subset, (e.g. `absolute_size`)
        """
        self.subset_root = subset_root
        self.image_preprocess = image_preprocess

        ann = os.path.join(subset_root, 'image2text.json')
        with open(ann, 'r') as f:
            self.sample_list = json.load(f)
        f.close()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx]

        # query image
        image_path = os.path.join(self.subset_root, sample_info['query'])
        query_image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            query_image = self.image_preprocess(query_image)

        # candidate texts
        candidate_texts = sample_info['keys']

        # label
        label = sample_info['label']

        sample = {
            "query_image": query_image,
            "candidate_texts": candidate_texts,
            "label": label
        }

        return sample

    def collate_fn(self, batch):
        query_image = []
        candidate_texts = []
        label = []
        for sample in batch:
            query_image.append(sample['query_image'])
            candidate_texts.append(sample['candidate_texts'])
            label.append(sample['label'])
        if self.image_preprocess is not None:
            query_image = torch.stack(query_image, dim=0)
        batch = {
            'query_image': query_image,
            'candidate_texts': candidate_texts,
            'label': torch.tensor(label)
        }
        return batch


class Text2ImageDataset(Dataset):
    def __init__(self, subset_root, image_preprocess=None):
        """
        Args:
            subset_root: the path to the root dir of a subset, (e.g. `absolute_size`)
        """
        self.subset_root = subset_root
        self.image_preprocess = image_preprocess

        ann = os.path.join(subset_root, 'text2image.json')
        with open(ann, 'r') as f:
            self.sample_list = json.load(f)
        f.close()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx]

        # query text
        query_text = sample_info['query']

        # candidate images
        candidate_images = []
        for img in sample_info['keys']:
            img = Image.open(os.path.join(self.subset_root, img)).convert('RGB')
            if self.image_preprocess is not None:
                img = self.image_preprocess(img)
            candidate_images.append(img)
        if self.image_preprocess is not None:
            candidate_images = torch.stack(candidate_images, dim=0)

        # label
        label = sample_info['label']

        sample = {
            "query_text": query_text,
            "candidate_images": candidate_images,
            "label": label
        }

        return sample

    def collate_fn(self, batch):
        query_text = []
        candidate_images = []
        label = []
        for sample in batch:
            query_text.append(sample['query_text'])
            candidate_images.append(sample['candidate_images'])
            label.append(sample['label'])
        if self.image_preprocess is not None:
            candidate_images = torch.stack(candidate_images, dim=0)

        batch = {
            'query_text': query_text,
            'candidate_images': candidate_images,
            'label': torch.tensor(label)
        }
        return batch


def get_data(data_root, subset_names, image_preprocess, batch_size, num_workers):
    """
    Create SPEC datasets
    Args:
        data_root: the path to the dir contains all the subsets' data
        subset_names: selected subsets names that you wish to evaluate your model on
        image_preprocess: image_preprocess
        batch_size: batch_size for dataloader
        num_workers: num_workers for dataloader
    Return:
        A list contains selected sub-datasets
    """

    data = {}
    for subset_nm in subset_names:

        subset_root_path = os.path.join(data_root, subset_nm)
        i2t_dataset = Image2TextDataset(subset_root=subset_root_path,
                                        image_preprocess=image_preprocess)
        t2i_dataset = Text2ImageDataset(subset_root=subset_root_path,
                                        image_preprocess=image_preprocess)

        i2t_loader = DataLoader(dataset=i2t_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=i2t_dataset.collate_fn,
                                shuffle=False,
                                drop_last=False)

        t2i_loader = DataLoader(dataset=t2i_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=t2i_dataset.collate_fn,
                                shuffle=False,
                                drop_last=False)

        data[subset_nm] = {
            'i2t_dataloader': i2t_loader,
            't2i_dataloader': t2i_loader
        }

    return data
