import os
import numpy as np
from argparse import ArgumentParser
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten,  MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import DenseNet121

class PlantNet():
    
    def __init__(self, input_shape = (250, 350, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self.model = self.create_pretrained_model()
        
    
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3,3), activation="relu", input_shape=self.input_shape))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(32, kernel_size=(3,3), activation="relu"))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))    
        
        model.add(Flatten())
        
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(self.num_classes, activation="softmax"))
        
        model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.1), metrics=['accuracy'])
        
        return model
    
    def create_pretrained_model(self):
        model_pretrained = DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3))
        
        model = Sequential()
        for layer in model_pretrained.layers:
            layer.trainable = False
        
        model.add(model_pretrained)
#         model.add(MaxPooling2D((2,2)))
        model.add(GlobalAveragePooling2D())
#         model.add(Dense(100, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        
        
        model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.1), metrics=['accuracy'])
        return model
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    train_path = args.training
#     val_path = args.validation
    
    X_train = np.load(os.path.join(train_path, "training.npz"))["images"]
    X_train = X_train / 255.
    
    y_train = np.load(os.path.join(train_path, "training.npz"))["labels"]
    y_train = to_categorical(y_train, 4)
    
    
    pn = PlantNet()
    model = pn.create_pretrained_model()
    
    model.fit(X_train, y_train, batch_size=16, validation_split=0.2, epochs=2, verbose=1)
    
    
    
    