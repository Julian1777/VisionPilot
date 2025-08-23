import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
import cv2 as cv
import numpy as np
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import zipfile
import glob


IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    INPUT_DIR = '/kaggle/input'
    WORKING_DIR = '/kaggle/working'
    SCRIPT_DIR = WORKING_DIR
    ZIP_FILE = None
    CULANE_DIR = '/kaggle/input/culane-dataset/culane'
    IMAGES_DIR = os.path.join(CULANE_DIR, "images")
    ANNOTATIONS_DIR = os.path.join(CULANE_DIR, "annotations")
    MASKS_DIR = os.path.join(CULANE_DIR, "masks")
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ZIP_FILE = None
    local_zip_patterns = [
        os.path.join(SCRIPT_DIR, "dataset", "culane_dataset_split.zip.001"),
        os.path.join(SCRIPT_DIR, "dataset", "culane_dataset_split.zip"),
        os.path.join(SCRIPT_DIR, "culane_dataset_split.zip.001"),
        os.path.join(SCRIPT_DIR, "culane_dataset_split.zip")
    ]
    for zip_path in local_zip_patterns:
        if os.path.exists(zip_path):
            ZIP_FILE = zip_path
            break
    CULANE_DIR = os.path.join(SCRIPT_DIR, "dataset", "culane")
    IMAGES_DIR = os.path.join(CULANE_DIR, "images")
    ANNOTATIONS_DIR = os.path.join(CULANE_DIR, "annotations")
    MASKS_DIR = os.path.join(CULANE_DIR, "masks")
if IS_KAGGLE:
    OUTPUT_DIR = '/kaggle/working'
    MODEL_PATH = os.path.join(OUTPUT_DIR, "lane_detection_model")
else:
    OUTPUT_DIR = SCRIPT_DIR
    MODEL_PATH = "lane_detection_model"

if not IS_KAGGLE:
    os.makedirs(CULANE_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
 

MAX_SAMPLES = 20000
IMG_SIZE = (512, 256)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 1000
POS_WEIGHT = 67
SEED = 123
EPOCHS = 30

tf.keras.mixed_precision.set_global_policy('mixed_float16')

print(f"Running environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
if IS_KAGGLE:
    print(f"Kaggle input directory: {INPUT_DIR}")
    print(f"Kaggle working directory: {WORKING_DIR}")
print(f"Script directory: {SCRIPT_DIR}")
print(f"Dataset zip file: {ZIP_FILE}")
print(f"Output directories will be created at: {CULANE_DIR}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU found: {gpus[0]}")
else:
    print("No GPU found, using CPU")

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
    bce = weighted_binary_crossentropy(y_true, y_pred, pos_weight)
    
    dice = weighted_dice_loss(y_true, y_pred, pos_weight)
    
    loss = 0.5 * bce + 0.5 * dice
    
    return tf.cast(loss, tf.float32)


def visualize_predictions(model, dataset, num_images=3):
    if isinstance(dataset, tf.data.Dataset):
        for i, (images, masks) in enumerate(dataset.take(1)):
            if i >= num_images:
                break
                
            display_count = min(num_images, images.shape[0])
            pred_masks = model.predict(images[:display_count])
            pred_masks = (pred_masks > 0.5).astype("float32")
            
            plt.figure(figsize=(15, 5*display_count))
            for j in range(display_count):
                plt.subplot(display_count, 3, j*3+1)
                plt.imshow(images[j])
                plt.title("Image")
                plt.axis('off')
                
                plt.subplot(display_count, 3, j*3+2)
                plt.imshow(masks[j], cmap='gray')
                plt.title("True Mask")
                plt.axis('off')
                
                plt.subplot(display_count, 3, j*3+3)
                plt.imshow(pred_masks[j], cmap='gray')
                plt.title("Predicted Mask")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    else:
        if isinstance(dataset, tuple) and len(dataset) == 2:
            images, masks = dataset
        else:
            print("Invalid input to visualize_predictions")
            return
            
        if len(images) > num_images:
            images = images[:num_images]
            masks = masks[:num_images]
            
        pred_masks = model.predict(images)
        pred_masks = (pred_masks > 0.5).astype("float32")
        
        plt.figure(figsize=(15, 5*num_images))
        for j in range(num_images):
            plt.subplot(num_images, 3, j*3+1)
            plt.imshow(images[j])
            plt.title("Image")
            plt.axis('off')
            
            plt.subplot(num_images, 3, j*3+2)
            plt.imshow(masks[j], cmap='gray')
            plt.title("True Mask")
            plt.axis('off')
            
            plt.subplot(num_images, 3, j*3+3)
            plt.imshow(pred_masks[j], cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def load_image_mask_pair(image_path, mask_path):
    image_path = tf.cast(image_path, tf.string)
    mask_path = tf.cast(mask_path, tf.string)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.cast(mask > 127, tf.float32)

    return image, mask

def augment_data(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    image = tf.image.random_brightness(image, 0.2)
    
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def get_aligned_image_mask_pairs(images_dir, masks_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
    
    def extract_number(filename):
        match = re.search(r'img_(\d+)', filename)
        return int(match.group(1)) if match else None
    
    image_map = {}
    mask_map = {}
    
    for img_file in image_files:
        num = extract_number(img_file)
        if num is not None:
            image_map[num] = os.path.join(images_dir, img_file)
    
    for mask_file in mask_files:
        num = extract_number(mask_file)
        if num is not None:
            mask_map[num] = os.path.join(masks_dir, mask_file)
    
    paired_images = []
    paired_masks = []
    
    for num in sorted(image_map.keys()):
        if num in mask_map:
            paired_images.append(image_map[num])
            paired_masks.append(mask_map[num])
    
    print(f"Found {len(paired_images)} aligned image-mask pairs")
    return paired_images, paired_masks

def get_dataset_paths(images_dir, masks_dir, max_samples=None):
    image_paths, mask_paths = get_aligned_image_mask_pairs(images_dir, masks_dir)
    
    if max_samples and len(image_paths) > max_samples:
        print(f"Limiting dataset to {max_samples} samples (from {len(image_paths)} available)")
        image_paths = image_paths[:max_samples]
        mask_paths = mask_paths[:max_samples]
    
    print(f"Dataset size: {len(image_paths)} samples")
    return image_paths, mask_paths

def create_dataset_from_paths(image_paths, mask_paths, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def extract_dataset_zip():
    # Bypass extraction if folders already exist and are non-empty
    if all([
        os.path.exists(IMAGES_DIR) and len(os.listdir(IMAGES_DIR)) > 0,
        os.path.exists(MASKS_DIR) and len(os.listdir(MASKS_DIR)) > 0,
        os.path.exists(ANNOTATIONS_DIR) and len(os.listdir(ANNOTATIONS_DIR)) > 0
    ]):
        print("Dataset folders already exist and are non-empty. Skipping extraction.")
        return True

    if not ZIP_FILE:
        print("No zip file found, assuming dataset is already extracted")
        return False

    print(f"Found zip file: {ZIP_FILE}")

    print("Extracting dataset...")

    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(SCRIPT_DIR)
        print("Dataset extraction completed")
        return True
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def draw_lane_mask(anno_path, image_shape, thickness=18):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)  # H x W
        with open(anno_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                coords = list(map(float, line.strip().split()))
                points = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]
                for i in range(1, len(points)):
                    cv.line(mask, points[i-1], points[i], color=255, thickness=thickness)
        return mask

def process_dataset():
    extract_dataset_zip()
    
    possible_dataset_paths = [
        os.path.join(SCRIPT_DIR, "dataset"),
        os.path.join(SCRIPT_DIR, "culane"),
        SCRIPT_DIR
    ]
    
    dataset_base = None
    for path in possible_dataset_paths:
        if os.path.exists(path):
            subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            if any(folder.startswith('driver_') for folder in subfolders):
                dataset_base = path
                print(f"Found dataset at: {dataset_base}")
                break
    
    if not dataset_base:
        print("Could not find extracted dataset. Looking for known folder patterns...")
        for root, dirs, files in os.walk(SCRIPT_DIR):
            if any(d.startswith('driver_') for d in dirs):
                dataset_base = root
                print(f"Found dataset at: {dataset_base}")
                break
    
    if not dataset_base:
        print("ERROR: Could not find dataset after extraction. Please check the zip file contents.")
        try:
            print("Top-level entries under SCRIPT_DIR:", os.listdir(SCRIPT_DIR))
        except Exception:
            pass
        return {
            "base_dir": CULANE_DIR,
            "images_dir": IMAGES_DIR,
            "annotations_dir": ANNOTATIONS_DIR,
            "masks_dir": MASKS_DIR,
            "count": 0
        }

    dataset_folders = []
    for item in os.listdir(dataset_base):
        item_path = os.path.join(dataset_base, item)
        if os.path.isdir(item_path) and item.startswith('driver_'):
            dataset_folders.append(item)
    
    if not dataset_folders:
        # Fallback to known folder names
        dataset_folders = [
            "driver_161_90frame",
            "driver_23_30frame",
            "driver_182_30frame"
        ]
    
    print(f"Found dataset folders: {dataset_folders}")
    
    output_dir = CULANE_DIR
    images_dir = IMAGES_DIR
    annotations_dir = ANNOTATIONS_DIR
    
    lane_class_dir = os.path.join(images_dir)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(lane_class_dir, exist_ok=True)

    img_count = 0
    total_subfolder_count = 0
    
    for folder_name in dataset_folders:
        base_dir = os.path.join(dataset_base, folder_name)
        
        if not os.path.exists(base_dir):
            print(f"Warning: Dataset directory not found at {base_dir}, skipping...")
            continue
            
        print(f"\nProcessing dataset folder: {folder_name}")
        
        subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
        total_subfolder_count += len(subfolders)
        print(f"Found {len(subfolders)} subfolders in {folder_name}")
        
        for subfolder in tqdm(subfolders, desc=f"Processing {folder_name} subfolders"):
            files = os.listdir(subfolder)
            
            img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            img_files = [f for f in files if any(f.endswith(ext) for ext in img_extensions)]
            
            for img_file in img_files:
                img_path = os.path.join(subfolder, img_file)
                img_basename, img_ext = os.path.splitext(img_file)
                
                anno_file = f"{img_basename}.lines.txt"
                anno_path = os.path.join(subfolder, anno_file)
                
                if not os.path.exists(anno_path):
                    continue
                
                img_count += 1
                new_img_name = f"img_{img_count}{img_ext}"
                new_anno_name = f"img_{img_count}_anno.txt"
                
                shutil.copy(img_path, os.path.join(lane_class_dir, new_img_name))
                shutil.copy(anno_path, os.path.join(annotations_dir, new_anno_name))
                
                img = cv.imread(img_path)
                mask = draw_lane_mask(anno_path, img.shape)
                mask = cv.resize(mask, IMG_SIZE, interpolation=cv.INTER_NEAREST)
                mask_output_path = os.path.join(MASKS_DIR, f"img_{img_count}.png")
                cv.imwrite(mask_output_path, mask)
                
                if img_count % 100 == 0:
                    print(f"Processed {img_count} images so far")
                
                # Limit to MAX_SAMPLES during processing
                if img_count >= MAX_SAMPLES:
                    print(f"Reached maximum sample limit of {MAX_SAMPLES}")
                    break
            
            if img_count >= MAX_SAMPLES:
                break
        
        if img_count >= MAX_SAMPLES:
            break
    
    print(f"\nProcessing complete. Organized {img_count} image-annotation pairs from {total_subfolder_count} subfolders across {len(dataset_folders)} datasets.")
    
    copied_images = len(os.listdir(images_dir))
    copied_annos = len(os.listdir(annotations_dir))
    masks_count = len(os.listdir(MASKS_DIR))
    print(f"Files in output directories: {copied_images} images, {copied_annos} annotations, {masks_count} masks")
    
    return {
        "base_dir": output_dir,
        "images_dir": images_dir,
        "annotations_dir": annotations_dir,
        "masks_dir": MASKS_DIR,
        "count": img_count
    }

def generate_masks_only():
    print("Generating masks from existing images and annotations...")
    
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
    
    total_images = len(image_files)
    print(f"Found {total_images} images. Generating masks...")
    
    for i, img_file in enumerate(tqdm(image_files, desc="Generating masks")):
        img_path = os.path.join(IMAGES_DIR, img_file)
        
        img_basename, img_ext = os.path.splitext(img_file)
        img_number = img_basename.split('_')[-1]
        
        anno_file = f"img_{img_number}_anno.txt"
        anno_path = os.path.join(ANNOTATIONS_DIR, anno_file)
        
        if not os.path.exists(anno_path):
            print(f"Warning: Annotation not found for {img_file}, skipping...")
            continue
        
        img = cv.imread(img_path)
        
        mask = draw_lane_mask(anno_path, img.shape)
        mask = cv.resize(mask, IMG_SIZE, interpolation=cv.INTER_NEAREST)
        
        mask_output_path = os.path.join(MASKS_DIR, f"img_{img_number}.png")
        cv.imwrite(mask_output_path, mask)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_images} masks")
    
    masks_count = len(os.listdir(MASKS_DIR))
    print(f"Mask generation complete. Created {masks_count} masks.")

def save_model_multiple_formats(model, base_path):
    print(f"Saving model in multiple formats with base name: {base_path}")
    
    h5_path = f"{base_path}.h5"
    model.save(h5_path)
    print(f"Saved model as: {h5_path}")
    
    savedmodel_path = f"{base_path}_savedmodel.keras"
    model.save(savedmodel_path)
    print(f"Saved model as: {savedmodel_path}")
    
    weights_path = f"{base_path}.weights.h5"
    model.save_weights(weights_path)
    print(f"Saved weights as: {weights_path}")
    
    json_path = f"{base_path}_architecture.json"
    with open(json_path, 'w') as f:
        f.write(model.to_json())
    print(f"Saved architecture as: {json_path}")
    
    return h5_path, savedmodel_path, weights_path, json_path

def create_lane_segmenation_model(input_shape=INPUT_SHAPE):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    base_model = tf.keras.applications.EfficientNetB0(
        input_tensor=inputs,
        include_top=False, 
        weights='imagenet'
    )
    
    backbone = tf.keras.Model(inputs=base_model.input, outputs=base_model.output, name='efficientnet_backbone')
    backbone.trainable = False
    
    # Get the output of the backbone
    x = backbone.output
    
    # Decoder path - Simple upsampling
    # First upsampling: ~7×7 -> ~14×14 
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Second upsampling: ~14×14 -> ~28×28
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Third upsampling: ~28×28 -> ~56×56
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Fourth upsampling: ~56×56 -> ~112×112
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Final upsampling: ~112×112 -> ~224×224
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Output layer
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, backbone

masks_exist = os.path.exists(MASKS_DIR) and len(os.listdir(MASKS_DIR)) > 0
masks_complete = masks_exist and os.path.exists(IMAGES_DIR) and len(os.listdir(MASKS_DIR)) == len([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])

image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
annotation_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.txt')]

masks_exist = os.path.exists(MASKS_DIR) and len(os.listdir(MASKS_DIR)) > 0
masks_complete = masks_exist and len(os.listdir(MASKS_DIR)) == len(image_files)

print(f"Images folder: {IMAGES_DIR} contains {len(image_files)} images.")
print(f"Annotations folder: {ANNOTATIONS_DIR} contains {len(annotation_files)} annotations.")

if masks_exist:
    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]
    print(f"Masks folder: {MASKS_DIR} contains {len(mask_files)} masks.")

if IS_KAGGLE:
    processed_data = {
        "base_dir": CULANE_DIR,
        "images_dir": IMAGES_DIR,
        "annotations_dir": ANNOTATIONS_DIR,
        "masks_dir": MASKS_DIR,
        "count": len(image_files)
    }
else:
    if len(image_files) > 0 and len(annotation_files) > 0:
        if masks_complete:
            print(f"Dataset already processed. Found {len(image_files)} images and {len(mask_files)} masks.")
            processed_data = {
                "base_dir": CULANE_DIR,
                "images_dir": IMAGES_DIR,
                "annotations_dir": ANNOTATIONS_DIR,
                "masks_dir": MASKS_DIR,
                "count": len(image_files)
            }
        else:
            print("Images and annotations exist, but masks are missing or incomplete.")
            print("Generating only the masks...")
            generate_masks_only()
            mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]
            processed_data = {
                "base_dir": CULANE_DIR,
                "images_dir": IMAGES_DIR,
                "annotations_dir": ANNOTATIONS_DIR,
                "masks_dir": MASKS_DIR,
                "count": len(image_files)
            }
    else:
        print("Images or annotations missing. Running complete dataset processing...")
        processed_data = process_dataset()
    

DATASET_PATH = processed_data["images_dir"]
print(f"Updated DATASET_PATH to: {DATASET_PATH}")

print(f"Creating dataset with max samples: {MAX_SAMPLES}")

image_paths, mask_paths = get_aligned_image_mask_pairs(IMAGES_DIR, MASKS_DIR)
if len(image_paths) > MAX_SAMPLES:
    image_paths = image_paths[:MAX_SAMPLES]
    mask_paths = mask_paths[:MAX_SAMPLES]
dataset_size = len(image_paths)
print(f"Final dataset size: {dataset_size}")

train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")

np.random.seed(SEED)
indices = np.arange(dataset_size)
np.random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_image_paths = [image_paths[i] for i in train_indices]
train_mask_paths = [mask_paths[i] for i in train_indices]
val_image_paths = [image_paths[i] for i in val_indices]
val_mask_paths = [mask_paths[i] for i in val_indices]
test_image_paths = [image_paths[i] for i in test_indices]
test_mask_paths = [mask_paths[i] for i in test_indices]

train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
train_ds = train_ds.map(load_image_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)  # Augment training data
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))
val_ds = val_ds.map(load_image_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_mask_paths))
test_ds = test_ds.map(load_image_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Train dataset: {len(train_ds)} batches")
print(f"Validation dataset: {len(val_ds)} batches")
print(f"Test dataset: {len(test_ds)} batches")



print(f"Starting training...")
if os.path.exists(f"{MODEL_PATH}.h5"):
    print(f"Loading existing model weights from {MODEL_PATH}.h5")
    model, backbone = create_lane_segmenation_model(input_shape=INPUT_SHAPE)
    model.load_weights(f"{MODEL_PATH}.h5")
else:
    print(f"No existing model found. Initializing new model.")
    model, backbone = create_lane_segmenation_model(input_shape=INPUT_SHAPE)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "best_model.weights.h5"),
    monitor="val_iou_metric",
    mode="max",
    save_best_only=True,
    verbose=1,
    save_weights_only=True
)

early_stopping = EarlyStopping(
    monitor='val_iou_metric',
    mode='max',
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_iou_metric',
    mode='max',
    factor=0.5,
    patience=6,
    verbose=1,
    min_lr=1e-6
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer, 
    loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, pos_weight=30.0), 
    metrics=[iou_metric]
)

print("Phase 1: Training with frozen backbone")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

print("Phase 2: Fine-tuning backbone layers")
backbone = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name == 'efficientnet_backbone':
        backbone = layer
        print(f"Found backbone model: {backbone.name}")
        break

if backbone is None:
    print("Could not find backbone model by name, attempting to find by type...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            print(f"Found model layer: {backbone.name}")
            break

if backbone is None:
    print("WARNING: No backbone found. Skipping fine-tuning phase.")
    history_phase2 = None
else:
    backbone.trainable = True
    print(f"Backbone has {len(backbone.layers)} layers")
    if len(backbone.layers) >= 100:
        for layer in backbone.layers[:100]:
            layer.trainable = False
    else:
        freeze_count = int(len(backbone.layers) * 0.3)
        for layer in backbone.layers[:freeze_count]:
            layer.trainable = False
        print(f"Froze first {freeze_count} of {len(backbone.layers)} layers")

    optimizer_ft = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer_ft,
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, pos_weight=20.0),
        metrics=[iou_metric]
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[checkpoint, reduce_lr]
    )

if history_phase2 is not None:
    combined_history = {}
    for key in history_phase1.history:
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
else:
    combined_history = history_phase1.history

save_model_multiple_formats(model, MODEL_PATH)
visualize_predictions(model, val_ds)

plt.figure(figsize=(16, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(combined_history['loss'], label='Train Loss')
plt.plot(combined_history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# IoU
plt.subplot(1, 2, 2)
plt.plot(combined_history['iou_metric'], label='Train IoU')
plt.plot(combined_history['val_iou_metric'], label='Val IoU')
plt.title('IoU over Epochs')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()

# Save plot to OUTPUT_DIR
plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(plot_path)
print(f"Saved training curves plot to: {plot_path}")
plt.show()