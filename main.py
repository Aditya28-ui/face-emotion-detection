# Imports
import zipfile
import os

# Define the path to the zip file and the extraction directory
zip_file_path = '/content/archive.zip'
extraction_path = '/content/fer2013_dataset'

# Create the extraction directory if it doesn't exist
os.makedirs(extraction_path, exist_ok=True)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

print(f"'{zip_file_path}' extracted to '{extraction_path}' successfully.")

import os
import subprocess # For robust shell command execution

# Define the path to the zip file and the extraction directory
zip_file_path = '/content/archive.zip'
extraction_path = '/content/fer2013_dataset'

# Create the extraction directory if it doesn't exist
os.makedirs(extraction_path, exist_ok=True)

try:
    # Use subprocess to run the unzip command, capturing output and checking return code
    # -q for quiet mode, -o for overwrite existing files without prompting
    result = subprocess.run(
        ['unzip', '-q', '-o', zip_file_path, '-d', extraction_path],
        capture_output=True,
        text=True,
        check=True # This will raise a CalledProcessError if the command returns a non-zero exit code
    )
    print(f"'{zip_file_path}' extracted to '{extraction_path}' successfully using unzip command.")
    if result.stdout:
        print("Unzip stdout:", result.stdout.strip())
    if result.stderr:
        print("Unzip stderr:", result.stderr.strip())

except subprocess.CalledProcessError as e:
    print(f"Error during extraction using unzip command: Command failed with exit code {e.returncode}.")
    print(f"Stderr: {e.stderr.strip()}")
    print("Please ensure '/content/archive.zip' is a valid zip file and try again.")
except FileNotFoundError:
    print("Error: 'unzip' command not found. Please ensure unzip is installed in your environment.")
except Exception as e:
    print(f"An unexpected error occurred during extraction: {e}")
    print("Please ensure '/content/archive.zip' is a valid zip file and try again.")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 1. Reset the validation_generator
validation_generator.reset()

# 2. Get prediction probabilities
predictions = model.predict(validation_generator, steps=validation_steps + 1)

# 3. Convert probabilities to class labels
y_pred = np.argmax(predictions, axis=1)

# 4. Retrieve true class labels
y_true = validation_generator.classes

# Ensure y_true and y_pred have the same length (handling potential generator truncation)
y_pred = y_pred[:len(y_true)]

# 5. Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 6. Get class names
class_labels = list(validation_generator.class_indices.keys())

# 7. Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)

# 8. Add title and labels
plt.title('Confusion Matrix for Facial Emotion Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 9. Display the plot
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Initialize the Sequential model
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants for image dimensions and batch size
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

# Path to the extracted dataset
data_dir = extraction_path # Using the 'extraction_path' variable from previous steps

# Create an instance of ImageDataGenerator for training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create an instance of ImageDataGenerator for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Use flow_from_directory to load images from the 'train' subdirectory
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Use flow_from_directory to load images from the 'test' subdirectory for validation
validation_generator = validation_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'), # Assuming 'test' directory for validation
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Typically set to False for validation/test data
)

print("Data generators created successfully. Training data prepared for augmentation and validation data prepared for rescaling.")


# Labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48))

        roi = roi_gray / 255.0
        roi = np.reshape(roi,(1,48,48,1))

        prediction = model.predict(roi)
        label = emotion_labels[int(prediction.argmax())]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
