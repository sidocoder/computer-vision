import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: LOAD & PREPROCESS DATA ---
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Preprocessing: Normalization (Scale pixels to 0-1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- STEP 2: CLASSICAL MODEL (SVM) ---
print("Training Classical Model (SVM)... This may take a minute...")
# Flatten images for SVM (32x32x3 -> 3072 features)
X_train_flat = train_images[:2000].reshape(2000, -1) 
y_train_flat = train_labels[:2000].flatten()
X_test_flat = test_images[:500].reshape(500, -1)
y_test_flat = test_labels[:500].flatten()

clf = svm.SVC(kernel='linear') # Optimization: Linear Kernel for speed
clf.fit(X_train_flat, y_train_flat)
svm_preds = clf.predict(X_test_flat)
svm_acc = accuracy_score(y_test_flat, svm_preds)

# --- STEP 3: DEEP LEARNING MODEL (CNN) ---
print("Training Deep Learning Model (CNN)...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'), # Optimization: Hidden Dense Layer
    layers.Dense(10) # 10 Output classes
])

model.compile(optimizer='adam', # Optimization: Adam Optimizer
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train for 5 epochs for demonstration
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels), verbose=1)

# --- STEP 4: VISUALIZE RESULTS ---
plt.figure(figsize=(12, 5))

# Plot CNN Training Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='CNN Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'CNN Validation Accuracy')
plt.title('Deep Learning Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Display Model Comparison
plt.subplot(1, 2, 2)
models_names = ['SVM (Classical)', 'CNN (Deep Learning)']
accuracies = [svm_acc, history.history['val_accuracy'][-1]]
plt.bar(models_names, accuracies, color=['blue', 'green'])
plt.title('Classical vs. Deep Learning Accuracy')
plt.ylabel('Accuracy Score')

plt.tight_layout()
plt.show()