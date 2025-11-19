import sys
if 'pandas' in sys.modules:
    del sys.modules['pandas']

import gc
import os
import math
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA_PATH = '/kaggle/input/tusimple-preprocessed-data/tusimple_preprocessed/training'
NUM_IMAGES = 7252
BATCH_SIZE = 64
IMG_HEIGHT = 256
IMG_WIDTH = 320
TARGET_SIZE = (IMG_HEIGHT, IMG_WIDTH)
SUBSET_SIZE = 2000
EPOCHS = 32
TRAIN_BATCH_SIZE = 8
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 100

def load_data(data_path=DATA_PATH, target_size=TARGET_SIZE, batch_size=BATCH_SIZE):    
    img_generator = keras.preprocessing.image.ImageDataGenerator()
    images_set = img_generator.flow_from_directory(
        data_path,
        shuffle=False,
        batch_size=batch_size,
        class_mode='binary',
        target_size=target_size
    )
    
    num_batches = NUM_IMAGES // batch_size + 1
    X, Y = [], []
    
    for i in range(num_batches):
        batch = next(images_set)
        batch_images = batch[0]
        batch_labels = batch[1]
        
        for ind, lb in enumerate(batch_labels):
            if lb == 0:  # Ground truth image
                X.append(batch_images[ind])
            else:  # Ground truth mask
                Y.append(np.mean(batch_images[ind], axis=2))
        
        if i % 10 == 0:
            print(f'Processed batch {i}/{num_batches}')
    
    # Convert to numpy arrays and shuffle
    X = np.array(X)
    Y = np.array(Y)
    X, Y = shuffle(X, Y, random_state=RANDOM_STATE)
    
    # Process masks
    Y = (Y >= 100).astype('int').reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    
    # Use subset for faster training
    X = X[:SUBSET_SIZE]
    Y = Y[:SUBSET_SIZE]
    
    print(f"Data loaded: X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Y range: [{Y.min()}, {Y.max()}]")
    
    # Clean up memory
    del images_set
    gc.collect()
    
    return X, Y

# U-Net model definition
def create_unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(input_size)
    rescale = keras.layers.Rescaling(1./255)(inputs)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(rescale)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_model(model, X_train, Y_train, X_val, Y_val, epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE):
    model.compile(
        optimizer='adam', 
        loss=keras.losses.BinaryFocalCrossentropy(), 
        metrics=['accuracy']
    )
    
    print("Model compiled successfully!")
    model.summary()
    
    tensorboard_log_dir = './logs'
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint("save_at_{epoch}.h5"),
        TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1),
    ]
    
    print(f"Starting training for {epochs} epochs...")
    
    # Train the model
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_val, Y_val),
        batch_size=batch_size
    )
    
    print("Training completed!")
    return history


# Evaluation functions
def evaluate_model(model, X_val, Y_val):
    # Make predictions
    preds = model.predict(X_val)
    preds_binary = (preds >= 0.5).astype('int')
    
    print(f"Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
    
    return preds, preds_binary

def calculate_metrics(Y_val, preds_binary):
    # Initialize metrics
    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])
    
    # Calculate metrics
    accuracy.update_state(Y_val, preds_binary)
    precision.update_state(Y_val, preds_binary)
    recall.update_state(Y_val, preds_binary)
    iou.update_state(Y_val, preds_binary)
    
    # Get metric values
    accuracy_value = accuracy.result().numpy()
    precision_value = precision.result().numpy()
    recall_value = recall.result().numpy()
    iou_value = iou.result().numpy()
    
    # Calculate F1 score
    f1_score = 2 * (precision_value * recall_value) / (precision_value + recall_value)
    
    # Calculate MSE and RMSE
    mse_value = mean_squared_error(Y_val.flatten(), preds_binary.flatten())
    rmse_value = math.sqrt(mse_value)
    
    metrics_dict = {
        'accuracy': accuracy_value,
        'precision': precision_value,
        'recall': recall_value,
        'f1_score': f1_score,
        'iou': iou_value,
        'mse': mse_value,
        'rmse': rmse_value
    }
    
    # Print metrics
    for metric_name, value in metrics_dict.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    
    return metrics_dict

def visualize_predictions(X_val, preds_binary, Y_val, start_idx=90, end_idx=98, save_images=True):
    plt.figure(figsize=(15, 45))
    index = 1
    
    if save_images:
        os.makedirs('./out', exist_ok=True)
    
    for i, (img, pred, ground_truth) in enumerate(zip(X_val[start_idx:end_idx], 
                                                      preds_binary[start_idx:end_idx], 
                                                      Y_val[start_idx:end_idx])):
        if save_images:
            cv2.imwrite(f'./out/img-{i+1}.jpg', img)
            cv2.imwrite(f'./out/pred-{i+1}.jpg', pred*255.)
            cv2.imwrite(f'./out/ground-{i+1}.jpg', ground_truth*255.)
        
        plt.subplot(len(range(start_idx, end_idx)), 2, index)
        plt.imshow(img/255.)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(len(range(start_idx, end_idx)), 2, index+1)
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        index += 2
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

def main():
    try:
        X, Y = load_data()
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
        )
        
        print(f"Training set: X_train={X_train.shape}, Y_train={Y_train.shape}")
        print(f"Validation set: X_val={X_val.shape}, Y_val={Y_val.shape}")
        
        del X, Y
        gc.collect()
        
        model = create_unet_model()
        
        history = train_model(model, X_train, Y_train, X_val, Y_val)
        
        preds, preds_binary = evaluate_model(model, X_val, Y_val)
        
        metrics = calculate_metrics(Y_val, preds_binary)
        
        visualize_predictions(X_val, preds_binary, Y_val)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='Training Loss', linestyle='-', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='-', color='orange')
        
        epochs_completed = len(history.history['loss'])
        start_loss_coord = (0, history.history['loss'][0])
        end_loss_coord = (epochs_completed-1, history.history['loss'][-1])
        plt.scatter(*start_loss_coord, color='green', s=50, label='Start')
        plt.scatter(*end_loss_coord, color='red', s=50, label='End')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.ylim(-0.1, 1.1)
        
        plt.text(*start_loss_coord, f'({start_loss_coord[0]}, {start_loss_coord[1]:.4f})', 
                 color='green', fontsize=10, ha='right', va='bottom', weight='bold')
        plt.text(*end_loss_coord, f'({end_loss_coord[0]}, {end_loss_coord[1]:.4f})', 
                 color='red', fontsize=10, ha='right', va='bottom', weight='bold')
        
        plt.subplot(2, 1, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linestyle='-', color='blue')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='-', color='orange')
        
        start_accuracy_coord = (0, history.history['accuracy'][0])
        end_accuracy_coord = (epochs_completed-1, history.history['accuracy'][-1])
        plt.scatter(*start_accuracy_coord, color='green', s=50, label='Start')
        plt.scatter(*end_accuracy_coord, color='red', s=50, label='End')
        
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.ylim(-0.1, 1.1)
        
        plt.text(*start_accuracy_coord, f'({start_accuracy_coord[0]}, {start_accuracy_coord[1]:.4f})', 
                 color='green', fontsize=10, ha='right', va='bottom', weight='bold')
        plt.text(*end_accuracy_coord, f'({end_accuracy_coord[0]}, {end_accuracy_coord[1]:.4f})', 
                 color='red', fontsize=10, ha='right', va='bottom', weight='bold')
        
        plt.tight_layout()
        plt.savefig('training_validation_plot.png')
        plt.show()
        
        model_save_path = "lane_detection_unet_trained.h5"
        model.save(model_save_path)

        return model, history, metrics
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    model, history, metrics = main()


