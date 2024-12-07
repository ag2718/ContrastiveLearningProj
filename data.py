'''
Installs needed:
pip install datasets
pip install torchvision
'''
import os
import torch
import datasets
from datasets import load_dataset
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from torchvision import transforms
from tree import Tree

dogs = [
    1,  # "Hunting Dog"
    [
        2,  # "Sporting Dog"
        [
            3,  # "Spaniel"
            [156],  # Blenheim spaniel
            [215],  # Brittany spaniel
            [216],  # Clumber, clumber spaniel
            [219],  # Cocker spaniel, English cocker spaniel, cocker
            [217],  # English springer, English springer spaniel
            [218],  # Welsh springer spaniel
            [220],  # Sussex spaniel
            [221],  # Irish water spaniel
        ],
        [
            4,  # "Retriever"
            [205],  # Flat-coated retriever
            [206],  # Curly-coated retriever
            [207],  # Golden retriever
            [208],  # Labrador retriever
            [209],  # Chesapeake Bay retriever
        ],
        [
            5,  # "Pointer"
            [210],  # German short-haired pointer
            [211],  # Vizsla, Hungarian pointer
        ],
        [
            6,  # "Setter"
            [212],  # English setter
            [213],  # Irish setter, red setter
            [214],  # Gordon setter
        ],
    ],
    [
        7,  # "Terrier"
        [
            8,  # "Wirehair"
            [189],  # Lakeland terrier
            [190],  # Sealyham terrier, Sealyham
        ],
        [
            9,  # "Bullterrier"
            [179],  # Staffordshire bullterrier, Staffordshire bull terrier
            [180],  # American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
        ],
        [
            10,  # "Fox Terrier"
            [188],  # Wire-haired fox terrier
        ],
        [
            11,  # "Schnauzer"
            [196],  # Miniature schnauzer
            [197],  # Giant schnauzer
            [198],  # Standard schnauzer
        ],
        [191],  # Airedale, Airedale terrier
        [193],  # Australian terrier
        [181],  # Bedlington terrier
        [182],  # Border terrier
        [192],  # Cairn, cairn terrier
        [194],  # Dandie Dinmont, Dandie Dinmont terrier
        [195],  # Boston bull, Boston terrier
        [184],  # Irish terrier
        [183],  # Kerry blue terrier
        [185],  # Norfolk terrier
        [186],  # Norwich terrier
        [199],  # Scotch terrier, Scottish terrier, Scottie
        [200],  # Tibetan terrier, chrysanthemum dog
        [201],  # Silky terrier, Sydney silky
        [202],  # Soft-coated wheaten terrier
        [203],  # West Highland white terrier
        [187],  # Yorkshire terrier
    ],
    [
        12,  # "Hound"
        [
            13,  # "Coonhound"
            [165],  # Black-and-tan coonhound
            [166],  # Walker hound, Walker foxhound
        ],
        [
            14,  # "Foxhound"
            [167],  # English foxhound
            [168],  # Redbone
        ],
        [
            15,  # "Greyhound"
            [171],  # Italian greyhound
            [172],  # Whippet
        ],
        [
            16,  # "Wolfhound"
            [169],  # Borzoi, Russian wolfhound
            [170],  # Irish wolfhound
        ],
        [
            17,  # "Other Hounds"
            [160],  # Afghan hound, Afghan
            [161],  # Basset, basset hound
            [162],  # Beagle
            [163],  # Bloodhound, sleuthhound
            [164],  # Bluetick
            [173],  # Ibizan hound, Ibizan Podenco
            [174],  # Norwegian elkhound, elkhound
            [175],  # Otterhound, otter hound
            [176],  # Saluki, gazelle hound
            [177],  # Scottish deerhound, deerhound
            [178],  # Weimaraner
            [159],  # Rhodesian ridgeback
        ],
    ],
]

def calculate_num_classes(array):
    if isinstance(array[0], int):
        return len(array)
    return sum(calculate_num_classes(subarray) for subarray in array)

def load_data():

    train_ds = load_dataset("imagenet-1k", split='train', streaming=True, trust_remote_code=True)
    val_ds = load_dataset("imagenet-1k", split='validation', streaming=True, trust_remote_code=True)

    # List of dog class indices in ImageNet
    num_total_classes = calculate_num_classes(dogs)
    class_tree =  Tree(dogs)
    dog_classes = np.array(class_tree.nodes_at_depth(num_total_classes))


    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Ensure 3 channels
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dog_dataset = train_ds.filter(lambda example: example['label'] in dog_classes)

    dog_dataset_val = val_ds.filter(lambda example: example['label'] in dog_classes)

    def collate_fn(batch):
        """
        Custom collate function to apply transformations on streaming data.
        """
        images, labels = [], []
        for example in batch:
            image = transform(example['image'])  # Apply transformations
            images.append(image)
            labels.append(example['label'])
        return torch.stack(images), torch.tensor(labels)

    # Create a DataLoader for the filtered dataset
    train_dataloader = DataLoader(dog_dataset, batch_size=100, num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(dog_dataset_val, batch_size=100, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, dogs, num_total_classes