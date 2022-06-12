from torch.utils.data import Dataset
from torchvision import transforms as T
from config import config
import cv2


class MaskDataset(Dataset):
    def __init__(self, images):
        super(MaskDataset, self).__init__()

        self.images = images

        self.no_aug_transforms = T.Compose([
            T.ToTensor(),
        ]) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        target_img = self.images[index]
        label = target_img.split("/")[-2]

        if label == "M":
            class_ = 0.0
        elif label == "N":
            class_ = 1.0
        else:
            raise ValueError("What the fuck!!!!")

        img = cv2.imread(target_img)
        img = self.no_aug_transforms(img)


        return img, class_