# pip install opencv2-python
# pip install numpy
# pip install keras
# pip install matplotlib
# pip install sklearn.metrics
# pip install numpy
# pip install keras,tensorflow

import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

emotion_dict = {
    "Angry": 0,
    "Disgusted": 1,
    "Fearful": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprised": 6,
}

# load json and create model
json_file = open("../Models\model_layer-10_Conv/Model_Conv10_layer.json", "r")
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("../Models/model_layer-10_Conv/ModelWeight_Conv10_layer.h5")
print("Model loaded from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator()

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
    "../Dataset/test",
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False,
)

# do prediction on test data
predictions = emotion_model.predict(test_generator)


# confusion matrix
con_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print(con_matrix)
confusionMatrix_display = ConfusionMatrixDisplay(
    confusion_matrix=con_matrix, display_labels=emotion_dict
)


confusionMatrix_display.plot(cmap=plt.cm.Blues)

fig = confusionMatrix_display.figure_
fig.set_figwidth(10)
fig.set_figheight(8)
fig.suptitle("Confusion matrix of 10 Conv layers and 2 FC layer network")


plt.show()
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))
