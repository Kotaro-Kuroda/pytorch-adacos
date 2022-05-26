import torch
import glob
from torchvision import transforms
import cv2
import os


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, train_dir, height, width, classes):
        super().__init__()
        self.train_dir = train_dir
        self.height = height
        self.width = width
        self.classes = classes
        self.list_data = self._get_list_data()

    def _get_list_data(self):
        list_dir = [directory for directory in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, directory))]
        list_data = []
        for directory in list_dir:
            label = self.classes.index(directory)
            directory = os.path.join(self.train_dir, directory)
            list_image = glob.glob(f'{directory}/*.jpeg')
            for image in list_image:
                img = cv2.imread(image)
                data = (img, label)
                list_data.append(data)
        return list_data

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.height, self.width)),
            transforms.Normalize((0.406, 0.456, 0.485), (0.225, 0.224, 0.229)),
        ])
        img, label = self.list_data[index]
        img = transform(img)
        return img, label

    def __len__(self):
        return len(self.list_data)


def dataloader(train_dir, dataset_class, batch_size, height, width):
    dataset = MyDataset(train_dir, height, width, dataset_class)
    torch.manual_seed(2020)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_dataloader
