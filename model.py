import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# Paths
train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

# Hyperparameters
input_shape = (128, 128, 3)
batch_size = 32
epochs = 50
fine_tune_at = -2  # Unfreeze last 2 layers
learning_rate = 1e-4  # Increased for better adaptation

# Class Name Fix
corrected_class_names = [
    "A Nageative", "A Positive", "AB Negative", 
    "AB Positive", "B Negative", "B Positive", "O Negative", "O Positive"
]

# Load dataset counts for class balancing
def get_class_weights(directory):
    class_counts = Counter()
    for class_name in os.listdir(directory):
        class_counts[class_name] = len(os.listdir(os.path.join(directory, class_name)))
    total_samples = sum(class_counts.values())
    class_weights = {i: total_samples / (len(class_counts) * count) for i, (class_name, count) in enumerate(class_counts.items())}
    return class_weights

# Compute class weights
class_weights = get_class_weights(train_dir)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load Data
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=batch_size, class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(128, 128), batch_size=batch_size, class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(128, 128), batch_size=batch_size, class_mode='categorical', shuffle=False
)

# Load Pretrained VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last 2 layers
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# Custom Head
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)  # Increased neurons
output = Dense(len(corrected_class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile with adjusted learning rate
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
tensorboard_callback = TensorBoard(log_dir='./logs')

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // batch_size),
    class_weight=class_weights,  # Apply class balancing
    callbacks=[lr_scheduler, tensorboard_callback]
)

# Save the model in Keras format
model.save('blood_group_model_vgg16.keras')

# Save training history
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Generate predictions
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=corrected_class_names, yticklabels=corrected_class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as 'confusion_matrix.png'.")

# Classification Report
class_report = classification_report(y_true, y_pred_classes, target_names=corrected_class_names, zero_division=0)
print("\nClassification Report:\n", class_report)

# Save report
with open("classification_report.txt", "w") as f:
    f.write(class_report)

print("✅ Classification report saved as 'classification_report.txt'.")
