import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
#import nltk
import json
from PIL import Image
#from build_clothing_vocab import Vocabulary
from preprocess import letterbox_image
import cv2
from torch.utils.data.sampler import SubsetRandomSampler
import sys

#defintion for fashion multi-attribute clothing data-set
class ClothingDataset(data.Dataset):
    """Clothing Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json_file, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: clothing annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        ids_temp = []
        json_data = open(json_file).read()
        
        self.root = root
        self.clothing = json.loads(json_data)

        for data_t in self.clothing:
        #    if data_t['id'] == 30001:
        #        break
            ids_temp.append(data_t['id'])
        print(len(ids_temp))
        self.ids = ids_temp
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        clothing = self.clothing
        #ann_id = self.ids[index+1]
        ann_id = index
        try:
            caption = clothing[ann_id]['caption']
        except Exception as ex:
            print('------------------------------')
            print(ann_id)
            print(index)
        img_id  = clothing[ann_id]['image']
        x1      = clothing[ann_id]['xmin']
        y1      = clothing[ann_id]['ymin']
        x2      = clothing[ann_id]['xmax']
        y2      = clothing[ann_id]['ymax']
        gt_label = clothing[ann_id]['gt_label']
        
        img_path = str(img_id)+'.jpg'
        #print(self.root + img_path)
        bbox = torch.FloatTensor([x1, y1, x2, y2])
        #print(str(index)+ ' ' + str(ann_id))

        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        crop_image = image.crop([x1,y1,x2,y2])
     #   crop_image.save(str(img_id)+ "_test.jpg")
      
        if self.transform is not None:
            img_ = self.transform(crop_image)
        target = list(map(int, gt_label.split(' ')))
        
        return img_, target, bbox

    def __len__(self):
        return len(self.ids)

def get_train_valid_loader(root, json_f, vocab, transform, batch_size, valid_size, random_seed, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for  clothing dataset."""
    # Clothing caption dataset

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
        ])

    train_dataset = ClothingDataset(root=root,
                       json_file=json_f,
                       vocab=vocab,
                       transform=transform_train)

    valid_dataset = ClothingDataset(root=root,
                       json_file=json_f,
                       vocab=vocab,
                       transform=transform_test)
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    # Data loader for Clothing dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                              batch_size=batch_size,
                                              sampler=train_sampler,
                                              num_workers=num_workers
                                              )

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                              batch_size=batch_size,
                                              sampler=valid_sampler,
                                              num_workers=num_workers
                                              )

    return train_loader, valid_loader

def get_test_loader(root, json_f, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for  clothing dataset."""
    # Clothing caption dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
        ])

    test_dataset = ClothingDataset(root=root,
                       json_file=json_f,
                       vocab=vocab,
                       transform=transform_test)

    # Data loader for Clothing dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
 

    return test_loader
