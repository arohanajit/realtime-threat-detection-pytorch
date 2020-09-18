# realtime-threat-detection-pytorch
This model detects human emotions and weapons in real time capture and thus determines whether a person is a threat or not. A current state, this model shows emotions and status of weapon detection on webcam capture.

## Methods/Libraries used
- OpenCV
- PyTorch Deep Learning Model
- Image Transforms

## Files
- Haarcascade_FrontFace.xml: to detect face boundaries
- face.pth: Pytorch deep learning model saved for emotions (Accuracy: 89.2%)
- weapon.pth: Pytorch deep learning model to detect weapons (Accuracy: 96.3%)

### Model Definitions:
- Face detection model: https://www.kaggle.com/arohanajit232/cnn-classifier-pytorch-test-acc-89-2
- Weapon detection: https://www.kaggle.com/arohanajit232/vgg16-classifier-pytorch-test-acc-96-3

### Procedure:
- Webcam capture is turned on.
- Face is detected using `detectMultiScale`
- Face boundaries are stored in `roi` for emotion detection.
- Whole webcam screen is stored in `weapon` for weapon detection.
- The frames are transformed using transform function into a format suitable for model processing
- The predicted labels and values are store.

For Display:
- Probability of each emotion is detected is calculated through a loop
- All these emotions are displayed on `canvas` frame using rectanglel lengths according to probability of each emotion.
- `frameClone` is the clone of frame. The emotion having highest probability is shown on border of face detected.
- The label showing whether weapon is detected or not is shown at bottom.

### Possible Use Cases
- The emotion detected are: Anger, Contempt, Disgust, Fear, Happy, Sadness, Surprise
- Weapon detection works as: Yes, No
- A person can be a possible threat when negative emotions such as Fear, Anger and Sadness have a high positive values, and Weapon detection is `True`.
- Another variant of this project can be if weapon detection is `True` and mask detection also equates to `True`, since it's highly likely terrorists use masks to cover their face.
