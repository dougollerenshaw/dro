import keras
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import RMSprop
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19

from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_image_size = (480,640,3)

model = VGG19(
    include_top=False, 
    weights='imagenet', 
    input_shape=(input_image_size), 
)

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)

# creating the final model 
model = Model(input = model.input, output = predictions)

# compile the model 
model.compile(
    loss = "binary_crossentropy", 
    optimizer = Adam(), #SGD(lr=0.0001, momentum=0.9), 
    metrics=["accuracy"]
)

#set up generators
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    directory=r"/ssd1/lick_detection_model/train/",
    target_size=(480,640),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(
    directory=r"/ssd1/lick_detection_model/val/",
    target_size=(480,640),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

## fit
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

model.save("lick_detect_model_weights.h5")