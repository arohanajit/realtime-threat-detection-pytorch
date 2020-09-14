import imutils
import cv2
import io
import numpy as np
import torch 
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image 
import torch.nn.functional as F

# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface.xml'
emotion_model_path = 'model.pt'

# model definiton
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        
            
        self.fc1 = nn.Linear(32*4*4, 512)
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 7)
        
    def forward(self, x):
        out = self.block(x)
        out = self.block2(out)
        out = self.block2(out)
        out = out.view(out.size(0), -1)   # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

my_transforms = transforms.Compose([
            transforms.Resize((50, 50)),
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
model = torch.load(emotion_model_path, map_location=torch.device('cpu'))
model.eval()


feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
   feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

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
        roi = my_transforms(roi).unsqueeze(0)
        
        
        preds = model(roi)
        preds = preds.detach().numpy().squeeze(0)
        label = EMOTIONS[preds.argmax()]
    else: continue

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):


                text = "{}: {:.2f}%".format(emotion, prob * 100)



                
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)


    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()