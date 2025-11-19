from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

IMG_SIZE = (512, 256)
LANE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'lane_detection_model.h5')
TEST_IMAGE_PATH = 'C:\\Users\\user\\Documents\\github\\self-driving-car-simulation\\images\\lane-detection-cnn\\demo\\lane3.jpg'
POS_WEIGHT = 67

class Cast(Layer):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self._dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'dtype': self._dtype})
        return config
    
def iou_metric(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou

def weighted_binary_crossentropy(y_true, y_pred, pos_weight=POS_WEIGHT):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    pos_loss = -y_true * tf.math.log(y_pred) * pos_weight
    neg_loss = -(1 - y_true) * tf.math.log(1 - y_pred)
    
    return tf.reduce_mean(pos_loss + neg_loss)

def weighted_dice_loss(y_true, y_pred, pos_weight=POS_WEIGHT):
    smooth = 1.0
    
    weighted_y_true = y_true * pos_weight
    
    y_true_f = tf.reshape(weighted_y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    weighted_sum = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    return 1 - (2. * intersection + smooth) / (weighted_sum + smooth)

def combined_loss(y_true, y_pred, pos_weight=POS_WEIGHT):
    increased_weight = pos_weight * 3.0
    bce = weighted_binary_crossentropy(y_true, y_pred, increased_weight)
    dice = weighted_dice_loss(y_true, y_pred, increased_weight)
    loss = 0.5 * bce + 0.5 * dice
    return tf.cast(loss, tf.float32)

def test_model():
    image = cv.imread(TEST_IMAGE_PATH)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    prediction = predict_lane(image)


    if len(prediction.shape) == 4:
        prediction_display = prediction[0, :, :, 0]
    else:
        prediction_display = prediction

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Prediction")

    plt.imshow(prediction_display, cmap='gray')

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return prediction


def img_preprocessing(image):
    img = image.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict_lane(frame):
    model = load_model(
        LANE_MODEL_PATH,
        custom_objects={
            'Cast': Cast,
            'iou_metric': iou_metric,
            'weighted_binary_crossentropy': weighted_binary_crossentropy,
            'weighted_dice_loss': weighted_dice_loss,
            'combined_loss': combined_loss,
        },
        compile=False
    )
    input_tensor = img_preprocessing(frame)
    prediction = model.predict(input_tensor)
    return prediction


test_model()
