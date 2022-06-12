import sys
sys.path.append("/mnt/829A20D99A20CB8B/projects/github_projects/Face_Mask_Detection_2")
from config import config
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from Dataset import MaskDataset
from MaskClassifier import MaskClassifier
from glob import glob
import random
from tqdm import tqdm 
import torch



BATCH_SIZE = 128
EPOCHS = 100
INITIAL_LEARNING_RATE = 1e-3
device = 'cuda'

images_list = [file for file in sorted(glob(config["path_to_ds"]))]

for i in range(10):
    random.shuffle(images_list)


val_images = images_list[0: int(0.1 * len(images_list))]
train_images = images_list[int(0.1 * len(images_list)): ]


train_ds = MaskDataset(images=train_images)
val_ds = MaskDataset(images=val_images)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)


model = MaskClassifier(backbone_name="efficientnet_b4")
model.to(device)


criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
best_val_acc = 0

if __name__ == "__main__":
    for epoch in range(EPOCHS):

        model.train()
        running_loss = []
        running_acc = []
        for images, labels in tqdm(train_loader):
            # labels = labels.type(torch.LongTensor)
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)
            

            logits = model(images)
            loss = criterion(logits, labels) 

            pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            correctly_classified = 0
            for index, i in enumerate(labels):
                if i == pred_classes[index]:
                    correctly_classified += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            running_acc.append((correctly_classified / len(labels)) * 100)
            running_loss.append(loss.item())
        

        
        model.eval()
        val_running_acc = []
        val_running_loss = []

        for val_images, val_labels in tqdm(val_loader):
            val_labels = val_labels.long()
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_logits = model(val_images)

            val_loss = criterion(val_logits, val_labels) 

            val_pred_classes = torch.argmax(torch.softmax(val_logits, dim=1), dim=1)

            val_correctly_classified = 0
            for index, i in enumerate(val_labels):
                if i == val_pred_classes[index]:
                    val_correctly_classified += 1

            val_running_acc.append((val_correctly_classified / len(val_labels)) * 100)
            val_running_loss.append(val_loss.item())


        
        avg_train_loss = round(sum(running_loss) / len(running_loss), 4) 
        avg_train_acc = round(sum(running_acc) / len(running_acc), 4)
        avg_val_loss = round(sum(val_running_loss) / len(val_running_loss), 4) 
        avg_val_acc = round(sum(val_running_acc) / len(val_running_acc), 4) 

        if avg_val_acc > best_val_acc:
            torch.save(model.state_dict(), config["path_to_save_maske_classifier_ckpt"])
        print(f"Epoch: {epoch} | Train ACC: {avg_train_acc} | Train Loss: {avg_train_loss} | Val ACC: {avg_val_acc} |  Val Loss: {avg_val_loss}")


