import sys
sys.path.append("/mnt/829A20D99A20CB8B/projects/github_projects/Face_Mask_Detection_2")

from uuid import uuid1
import cv2
import argparse
from FaceBoxes.FaceBoxes import FaceBoxes
import numpy as np
from MaskClassifier.MaskClassifier import MaskClassifier
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from glob import glob


class FaceMaskDetector:
    def __init__(self):
        # self.__video_path = args.video_path
        self.__video_path = "/mnt/829A20D99A20CB8B/projects/github_projects/Face_Mask_Detection/test_videos/sample2_short.mp4"
        # self.__image_path = args.image_path 
        self.__face_margin = 0
        self.__face_crop_dim = (224, 224)
        self.__max_dim_for_img = 1400 

        self.mask_classifier = MaskClassifier()
        self.mask_classifier.load_state_dict(torch.load("/mnt/829A20D99A20CB8B/projects/github_projects/Face_Mask_Detection/mask_classifier.pt"))
        self.mask_classifier.eval()

        self.inference_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            
        ])

    def __apply_img_transformations(self, face):
        return self.inference_transforms(face)

    def __detect_faces(self, frame):    
        face_boxes = FaceBoxes(timer_flag=True)
        dets = face_boxes(frame)  # xmin, ymin, w, h
        return dets


    def __crop_faces(self, frame, dets):
        faces = []
        for d in dets:
            (x, y), (x2, y2) = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
            width, height = x2 - x, y2 - y
            face = frame[max(y-int(self.__face_margin*height), 0):min(y2+int(self.__face_margin*height), frame.shape[0]), max(x-int(self.__face_margin*width), 0):min(x2+int(self.__face_margin*width), frame.shape[1])]
            face = cv2.resize(face, self.__face_crop_dim)

            # path_to_save = f"/mnt/829A20D99A20CB8B/projects/github_projects/Face_Mask_Detection/DATASET/raw_crop/{uuid1()}.jpg"
            # cv2.imwrite(path_to_save, face)

            face = Image.fromarray(face)

            face = self.__apply_img_transformations(face)
            faces.append(face)

        # faces = np.array(faces)        
        # faces = faces.astype('float32')
        
        return faces
    
    def __draw_bounding_boxes(self, image, dets, classification_data):
        for index, d in enumerate(dets):
            (x, y), (x2, y2) = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
            width, height = x2 - x, y2 - y

            start_point = (max(x-int(self.__face_margin*width), 0), max(y-int(self.__face_margin*height), 0))

            end_point =  (min(x2+int(self.__face_margin*width), image.shape[1]), min(y2+int(self.__face_margin*height), image.shape[0]))
            
            color = (36,255,12) if classification_data[index] == "No Mask" else (255,36,12)
            cv2.rectangle(image, start_point, end_point, color, 2)
            cv2.putText(image, classification_data[index], (x-2, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image

    def __visualise_batch_of_faces(self, batch_of_faces):
        for b in batch_of_faces:
            cv2.imshow("face", b)
            cv2.waitKey(0)

    def __check_size_and_resize(self, img):
        if img.shape[0] > self.__max_dim_for_img or img.shape[1] > self.__max_dim_for_img:
            img = cv2.resize(img, (self.__max_dim_for_img, self.__max_dim_for_img))

        return img

    def detect_mask_on_video(self):
        vid = cv2.VideoCapture(self.__video_path)

        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
    
        while(True):
            ret, frame = vid.read()
            
            frame = self.__check_size_and_resize(frame)
            dets = self.__detect_faces(frame)
            faces = self.__crop_faces(frame=frame, dets=dets)
            
            if not(len(faces) == 0):
                classification_data = self.__classify_faces(faces)
                frame = self.__draw_bounding_boxes(frame, dets, classification_data)


            output.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()

    def __classify_faces(self, faces):
        faces = torch.stack(faces)
        classes = torch.argmax(torch.softmax(self.mask_classifier(faces), dim=1), dim=1)

        preds = []
        classes = classes.detach().to("cpu").numpy()
        for c in classes:
            if c == 1:
                tag = "No Mask"
            else:
                tag = "Mask"
            
            preds.append(tag)

        return preds

    def detect_mask_on_single_image(self, path_to_img):
        # if path_to_img is None:
        #     img = cv2.imread(self.__image_path)
        # else:
        #     img = cv2.imread(path_to_img)

        # img = cv2.imread(self.__image_path)
        img = cv2.imread(path_to_img)
        img = self.__check_size_and_resize(img)
        dets = self.__detect_faces(img)
        faces = self.__crop_faces(frame=img, dets=dets)
        
        classification_data = self.__classify_faces(faces)

        # self.__visualise_batch_of_faces(faces)
        img = self.__draw_bounding_boxes(img, dets, classification_data)

        
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--mode", type=str, default="image", help="image mode or video mode: image or video")
    # parser.add_argument("--image_path", type=str, default="image", help="abs path to the target image")
    # parser.add_argument("--video_path", type=str, default="video", help="abs path to the target image")
    # parser.add_argument("--classifier_model_path", type=str, default="video", help="abs path to the classifier model")
    # parser.add_argument("--device", type=str, default="cpu", help="cpu or gpu")
    
    # args = parser.parse_args()
    
    
    
    print("=================")
    face_mask_detector = FaceMaskDetector()#args)

    face_mask_detector.detect_mask_on_video()
    
    # img_file = "/mnt/829A20D99A20CB8B/projects/github_projects/Face_Mask_Detection/test_img/both.jpg"
    # face_mask_detector.detect_mask_on_single_image(img_file)