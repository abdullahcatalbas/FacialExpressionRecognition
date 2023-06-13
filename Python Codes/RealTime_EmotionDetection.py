# pip install opencv2-python
# pip install numpy
# pip install keras
# pip install tensorflow
from keras.models import model_from_json
import numpy as np
import cv2


class FER_Model(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FER_Model.EMOTIONS_LIST[np.argmax(self.preds)]


face_detector = cv2.CascadeClassifier("Haarcascade/haarcascade_frontalface_default.xml")
model = FER_Model(
    "../Models\model_layer-10_Conv/Model_Conv10_layer.json",
    "../Models/model_layer-10_Conv/ModelWeight_Conv10_layer.h5",
)


class FER_Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        rej, fr = self.video.read()
        fr = cv2.resize(fr, (1200, 720))

        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        # gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_fr, 1.3, 5)

        for x, y, w, h in faces:
            fc = gray_fr[y : y + h, x : x + w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return fr


# Detect Emotion
def Emotion_Detector(camera):
    while True:
        frame = camera.get_frame()

        cv2.imshow("Facial Expression Recognization", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


Emotion_Detector(FER_Video())
