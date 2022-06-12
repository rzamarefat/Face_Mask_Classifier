
from MaskClassifier import MaskClassifier
import torch
from torchvision import transforms as T
import cv2


path_to_img = "/mnt/829A20D99A20CB8B/projects/github_projects/Face_Mask_Detection_2/test_img/face_no_mask_00.png"
device = "cuda"
transforms = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224))
])

img = cv2.imread(path_to_img)
img = transforms(img)
img = torch.unsqueeze(img, 0)
img = img.to(device)


model = MaskClassifier()
model.to(device)
model.load_state_dict(torch.load("/mnt/829A20D99A20CB8B/projects/github_projects/Face_Mask_Detection_2/mask_classifier.pt"))
model.eval()

logits = model(img)
print(logits)
pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)

pred_class = pred_class.item()

if pred_class == 1:
    print("No Mask")
else: 
    print("Mask")

