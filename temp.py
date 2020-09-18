import imutils
import cv2
import io
import numpy as np
import torch 
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image 
import torch.nn.functional as F



def start_cam():

    # parameters for loading data and images
    detection_model_path = 'haarcascade_frontalface.xml'
    face_model = torch.jit.load("face.pth")
    weapon_model = torch.jit.load("weapon.pth")

    face_transforms = transforms.Compose([
                transforms.Resize((50, 50)),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    weapon_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])




    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    WEAPONS = ['Yes','No']
    face_model.eval()
    weapon_model.eval()



    # starting video streaming
    cv2.namedWindow('your_face')
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        frame = camera.read()[1]
        #reading the frame
        frame = imutils.resize(frame,width=300)
        faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = frame[fY:fY + fH, fX:fX + fW]
            roi = Image.fromarray(roi)
            face = face_transforms(roi).unsqueeze(0)
            weapon = frame
            weapon = Image.fromarray(weapon)
            weapon = weapon_transforms(weapon).unsqueeze(0)
            
            
            preds_face = face_model(face)
            preds_face = preds_face.detach().numpy().squeeze(0)
            label_face = EMOTIONS[preds_face.argmax()]

            preds_weapon = weapon_model(weapon)
            preds_weapon = preds_weapon.detach().numpy().squeeze(0)
            print(preds_weapon)
            label_weapon = WEAPONS[preds_weapon.argmax()]

        else: continue

    
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds_face)):


                    text = "{}: {:.2f}%".format(emotion, prob * 100)




                
                    w = int(prob * 100)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label_face, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                (0, 0, 255), 2)

                    cv2.putText(frameClone,"Weapon: {}".format(label_weapon), (200, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))


        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()