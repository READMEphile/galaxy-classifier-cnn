import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create project directories
base_dir = "galaxy_classification"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

# Download and prepare Galaxy Zoo dataset (simplified version)
def download_galaxy_zoo_sample():
    print("Downloading Galaxy Zoo sample dataset...")
    
    # For demonstration purposes, we'll create synthetic data
    # In a real project, you would download the actual dataset
    
    # Create directory for images
    image_dir = os.path.join(base_dir, "data", "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Create synthetic dataset with 300 samples
    n_samples = 300
    galaxy_types = ["spiral", "elliptical", "irregular"]
    galaxy_labels = np.random.choice(range(len(galaxy_types)), size=n_samples)
    
    # Create synthetic images and save them
    image_paths = []
    for i in range(n_samples):
        # Create a random image for demonstration
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Add some structure based on the galaxy type to make it more realistic
        if galaxy_labels[i] == 0:  # Spiral
            # Create spiral pattern
            center = (64, 64)
            for r in range(10, 60, 2):
                for theta in np.linspace(0, 8*np.pi, 100):
                    x = int(center[0] + r * np.cos(theta) * theta/20)
                    y = int(center[1] + r * np.sin(theta) * theta/20)
                    if 0 <= x < 128 and 0 <= y < 128:
                        img[y, x] = [200, 200, 255]
        elif galaxy_labels[i] == 1:  # Elliptical
            # Create elliptical pattern
            cv2.ellipse(img, (64, 64), (40, 30), 0, 0, 360, (200, 200, 255), -1)
            img = cv2.GaussianBlur(img, (15, 15), 0)
        else:  # Irregular
            # Create irregular pattern with random blobs
            for _ in range(5):
                radius = np.random.randint(5, 20)
                center = (np.random.randint(20, 108), np.random.randint(20, 108))
                cv2.circle(img, center, radius, (200, 200, 255), -1)
            img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Add noise and stars
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add random stars
        for _ in range(20):
            x, y = np.random.randint(0, 128, 2)
            brightness = np.random.randint(200, 256)
            img[y, x] = [brightness, brightness, brightness]
        
        # Save the image
        img_path = os.path.join(image_dir, f"galaxy_{i:04d}.jpg")
        cv2.imwrite(img_path, img)
        image_paths.append(img_path)
    
    # Create a dataframe with image paths and labels
    df = pd.DataFrame({
        'image_path': image_paths,
        'label_idx': galaxy_labels,
        'label': [galaxy_types[i] for i in galaxy_labels]
    })
    
    # Save the dataframe
    df.to_csv(os.path.join(base_dir, "data", "galaxy_data.csv"), index=False)
    
    print(f"Created synthetic dataset with {n_samples} images")
    print(f"Distribution: {df['label'].value_counts().to_dict()}")
    
    return df

# Split the data
def split_data(galaxy_df):
    train_df, temp_df = train_test_split(galaxy_df, test_size=0.3, stratify=galaxy_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df

# Data preprocessing and augmentation
def preprocess_image(img_path, target_size=(128, 128)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

# Data generator function
def data_generator(dataframe, batch_size=32, augment=False, target_size=(128, 128)):
    indices = np.arange(len(dataframe))
    num_classes = len(dataframe['label'].unique())
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_df = dataframe.iloc[batch_indices]
            
            batch_x = np.array([preprocess_image(path, target_size) for path in batch_df['image_path']])
            batch_y = to_categorical(batch_df['label_idx'], num_classes=num_classes)
            
            if augment:
                # Apply random augmentations
                augmented_batch = []
                for img in batch_x:
                    # Apply random rotation
                    if np.random.random() > 0.5:
                        angle = np.random.uniform(-30, 30)
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                        img = cv2.warpAffine(img, M, (w, h))
                    
                    # Apply random brightness adjustment
                    if np.random.random() > 0.5:
                        brightness = np.random.uniform(0.7, 1.3)
                        img = np.clip(img * brightness, 0, 1)
                    
                    # Apply random flip
                    if np.random.random() > 0.5:
                        img = cv2.flip(img, 1)
                    
                    augmented_batch.append(img)
                batch_x = np.array(augmented_batch)
            
            yield batch_x, batch_y

# Build CNN model - IMPORTANT: Using a Functional API approach instead of Sequential
def build_galaxy_cnn(input_shape=(128, 128, 3), num_classes=3):
    # Using functional API to be able to get intermediate layers more easily for Grad-CAM
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Second convolutional block
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Third convolutional block
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define callbacks
def create_callbacks(base_dir):
    checkpoint = ModelCheckpoint(
        os.path.join(base_dir, "models", "galaxy_cnn_best.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]

# Plot training history
def plot_training_history(history, base_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "results", "training_history.png"))
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(y_true_classes, y_pred_classes, class_names, base_dir):
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(base_dir, "results", "confusion_matrix.png"))
    plt.close()

# FIXED: Implement Grad-CAM for model explainability
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we need to find the last convolutional layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # Create a model that maps the input image to the activations
    # of the last conv layer and the output predictions
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # This is the gradient of the output neuron with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.maximum(np.max(heatmap), 1e-10)
    
    return heatmap

def display_gradcam(image_path, heatmap, alpha=0.4):
    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    superimposed_img = jet_heatmap * alpha + img / 255.0 * (1 - alpha)
    
    return superimposed_img

# Display Grad-CAM for sample images
def visualize_gradcam_samples(model, test_df, class_names, base_dir):
    os.makedirs(os.path.join(base_dir, "results", "gradcam"), exist_ok=True)
    
    # The last convolutional layer will be used for Grad-CAM
    last_conv_layer_name = 'conv3_2'  # Now we use the named layer from functional API
    
    # Get a sample from each class
    samples_per_class = 3
    fig, axes = plt.subplots(len(class_names), samples_per_class, figsize=(15, 5*len(class_names)))
    
    for i, class_name in enumerate(class_names):
        class_samples = test_df[test_df['label'] == class_name].sample(min(samples_per_class, len(test_df[test_df['label'] == class_name])))
        
        for j, (_, row) in enumerate(class_samples.iterrows()):
            # Load and preprocess image
            img_path = row['image_path']
            img_array = np.array([preprocess_image(img_path)])
            
            # Get prediction and class index
            preds = model.predict(img_array)
            pred_index = np.argmax(preds[0])
            pred_class = class_names[pred_index]
            confidence = preds[0][pred_index]
            
            # Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            
            # Display original image
            orig_img = cv2.imread(img_path)
            orig_img = cv2.resize(orig_img, (128, 128))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Display superimposed heatmap
            superimposed_img = display_gradcam(img_path, heatmap)
            superimposed_img = cv2.cvtColor(superimposed_img.astype(np.float32), cv2.COLOR_BGR2RGB)
            
            # Plot images side by side
            axes[i, j].imshow(orig_img)
            axes[i, j].set_title(f"True: {class_name}\nPred: {pred_class} ({confidence:.2f})")
            axes[i, j].axis('off')
            
            # Save individual Grad-CAM image
            output_path = os.path.join(base_dir, "results", "gradcam", f"{class_name}_{j}_gradcam.png")
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(orig_img)
            plt.title(f"Original - True: {class_name}")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(superimposed_img)
            plt.title(f"Grad-CAM - Pred: {pred_class} ({confidence:.2f})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "results", "gradcam_overview.png"))
    plt.close()

def main():
    print("\nGalaxy Classification Project using Deep Learning")
    print("=" * 50)
    
    # Download/create dataset
    galaxy_df = download_galaxy_zoo_sample()
    
    # Split the data
    train_df, val_df, test_df = split_data(galaxy_df)
    
    # Get class names
    class_names = galaxy_df['label'].unique()
    num_classes = len(class_names)
    
    # Build the model with functional API instead of Sequential
    model = build_galaxy_cnn(num_classes=num_classes)
    model.summary()
    
    # Define callbacks
    callbacks = create_callbacks(base_dir)
    
    # Train the model
    batch_size = 32
    epochs = 50

    train_generator = data_generator(train_df, batch_size=batch_size, augment=True)
    val_generator = data_generator(val_df, batch_size=batch_size)

    steps_per_epoch = max(1, len(train_df) // batch_size)
    validation_steps = max(1, len(val_df) // batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history, base_dir)
    
    # Evaluate on test set
    test_batch_x = np.array([preprocess_image(path) for path in test_df['image_path']])
    test_batch_y = to_categorical(test_df['label_idx'], num_classes=num_classes)
    
    test_loss, test_accuracy = model.evaluate(test_batch_x, test_batch_y)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions for test set
    y_pred = model.predict(test_batch_x)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_batch_y, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names, base_dir)
    
    # Run Grad-CAM visualization
    visualize_gradcam_samples(model, test_df, class_names, base_dir)
    
    print("Project completed! Results are saved in the 'galaxy_classification/results' directory.")

if __name__ == "__main__":
    main()