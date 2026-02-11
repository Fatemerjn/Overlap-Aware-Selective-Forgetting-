import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset


_CIFAR10_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
]

_CIFAR10_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
]

_CIFAR100_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]

_CIFAR100_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]

_TINYIMAGENET_TRAIN_TRANSFORMS = [
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

_TINYIMAGENET_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

_CIFAR100_SUPERCLASSES = [
    ("aquatic_mammals", ["beaver", "dolphin", "otter", "seal", "whale"]),
    ("fish", ["aquarium_fish", "flatfish", "ray", "shark", "trout"]),
    ("flowers", ["orchid", "poppy", "rose", "sunflower", "tulip"]),
    ("food_containers", ["bottle", "bowl", "can", "cup", "plate"]),
    ("fruit_and_vegetables", ["apple", "mushroom", "orange", "pear", "sweet_pepper"]),
    ("household_electrical_devices", ["clock", "keyboard", "lamp", "telephone", "television"]),
    ("household_furniture", ["bed", "chair", "couch", "table", "wardrobe"]),
    ("insects", ["bee", "beetle", "butterfly", "caterpillar", "cockroach"]),
    ("large_carnivores", ["bear", "leopard", "lion", "tiger", "wolf"]),
    ("large_man_made_outdoor_things", ["bridge", "castle", "house", "road", "skyscraper"]),
    ("large_natural_outdoor_scenes", ["cloud", "forest", "mountain", "plain", "sea"]),
    ("large_omnivores_and_herbivores", ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"]),
    ("medium_mammals", ["fox", "porcupine", "possum", "raccoon", "skunk"]),
    ("non_insect_invertebrates", ["crab", "lobster", "snail", "spider", "worm"]),
    ("people", ["baby", "boy", "girl", "man", "woman"]),
    ("reptiles", ["crocodile", "dinosaur", "lizard", "snake", "turtle"]),
    ("small_mammals", ["hamster", "mouse", "rabbit", "shrew", "squirrel"]),
    ("trees", ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"]),
    ("vehicles_1", ["bicycle", "bus", "motorcycle", "pickup_truck", "train"]),
    ("vehicles_2", ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]),
]


def get_cifar100_superclass_tasks(dataset):
    label_to_index = {label: idx for idx, label in enumerate(dataset.classes)}
    labels_per_task = []
    for _, class_names in _CIFAR100_SUPERCLASSES:
        labels_per_task.append([label_to_index[name] for name in class_names])
    permutation = [label for task_labels in labels_per_task for label in task_labels]
    return labels_per_task, permutation


class SubDataset(Dataset):
    def __init__(self, original_dataset, sub_labels, permutation):
        super().__init__()
        self.dataset = original_dataset
        self.permutation = permutation
        self.sub_indices = []
        for index in range(len(self.dataset)):
            label = self.dataset.targets[index]
            if label in sub_labels:
                self.sub_indices.append(index)

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        return sample[0], self.permutation.index(sample[1])


def get_task_datasets(args):
    T = args.n_tasks
    CPT = args.class_per_task

    data = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'tinyimagenet': datasets.ImageFolder,
    }
    train_transform = {
        'cifar10': _CIFAR10_TRAIN_TRANSFORMS,
        'cifar100': _CIFAR100_TRAIN_TRANSFORMS,
        'tinyimagenet': _TINYIMAGENET_TRAIN_TRANSFORMS,
    }
    test_transform = {
        'cifar10': _CIFAR10_TEST_TRANSFORMS,
        'cifar100': _CIFAR100_TEST_TRANSFORMS,
        'tinyimagenet': _TINYIMAGENET_TEST_TRANSFORMS,
    }

    if args.dataset == 'tinyimagenet':
        train = data[args.dataset](os.path.join(args.data_dir, 'tinyimagenet', 'train'),
                                   transform=transforms.Compose(train_transform[args.dataset]))
        test = data[args.dataset](os.path.join(args.data_dir, 'tinyimagenet', 'val'),
                                  transform=transforms.Compose(test_transform[args.dataset]))
    else:
        train = data[args.dataset](args.data_dir, train=True, download=True,
                                   transform=transforms.Compose(train_transform[args.dataset]))
        test = data[args.dataset](args.data_dir, train=False, download=True,
                                  transform=transforms.Compose(test_transform[args.dataset]))

    if args.dataset == "cifar100":
        labels_per_task, permutation = get_cifar100_superclass_tasks(train)
        if CPT != 5:
            raise ValueError("CIFAR-100 superclass tasks require --class_per_task 5.")
        if T > len(labels_per_task):
            raise ValueError("CIFAR-100 superclass tasks require --n_tasks <= 20.")
        labels_per_task = labels_per_task[:T]
        permutation = [label for task_labels in labels_per_task for label in task_labels]
    else:
        # generate randomized labels-per-task
        permutation = np.random.permutation(np.arange(T * CPT))
        labels_per_task = [list(permutation[task_id * CPT:(task_id + 1) * CPT]) for task_id in range(T)]

    print("Labels per task: ", labels_per_task)

    train_datasets, test_datasets = [], []
    for task_id, labels in enumerate(labels_per_task):
        train_datasets.append(SubDataset(train, labels, list(permutation)))
        test_datasets.append(SubDataset(test, labels, list(permutation)))

    return train_datasets, test_datasets
