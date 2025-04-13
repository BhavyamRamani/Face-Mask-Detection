from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Path to your dataset
train_dir = 'dataset'  # path to the dataset folder

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   horizontal_flip=True, 
                                   zoom_range=0.2, 
                                   shear_range=0.2)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(224, 224), 
                                                    batch_size=32, 
                                                    class_mode='binary')

# Load MobileNetV2 model, pre-trained on ImageNet
base_model = MobileNetV2(weights='/Users/bhavyamramani/Downloads/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5', include_top=False)


# Create a custom model on top
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))

# Freeze the base model layers
base_model.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=5)

# Save the trained model
model.save('mask_detector.keras')  # Save the trained model
