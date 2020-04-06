import os
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import logging
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten,  MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

BATCH_SIZE = 16
    
def keras_model_fn():
    """Returns pre-trained keras model for further fine-tuning"""
    model_pretrained = DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3))
    
    model = Sequential()
    model.add(model_pretrained)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(250, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
    return model

def serving_input_fn(hyperparameters):
    """Function necessary for Sagemaker TF serving"""
    tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    inputs = {"vgg16_input": tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def train_input_fn(training_dir):
    """Returns training data"""
    return _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_dir=training_dir)


def eval_input_fn(training_dir):
    """Returns validation data"""
    return _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_dir=training_dir)


def _input(mode, batch_size, data_dir):
    """Performs data pre-processing such as augmentation. Type of pre-processing depends on TRAIN or VAL step"""
    if mode == tf.estimator.ModeKeys.TRAIN:
        img_generator = ImageDataGenerator(rescale=1./255, 
                        zoom_range=0.15,
                        width_shift_range=0.15, 
                        height_shift_range=0.15, 
                        rotation_range=360,
                        horizontal_flip=True, 
                        vertical_flip=True)
    elif mode == tf.estimator.ModeKeys.EVAL:
        img_generator = ImageDataGenerator(rescale=1./255)
        
    generator = img_generator.flow_from_directory(data_dir, target_size=(224, 224), batch_size=batch_size)

    return generator



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='Directory for training data.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALIDATION'),
        help='Directory for validation data.')
    parser.add_argument(
        '--eval',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_EVAL'),
        help='Directory for evaluation data.')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Model storage directory.')
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size.')

    
    args = parser.parse_args()
    train_path = args.train
    val_path = args.validation
    model_dir = args.model_dir
    
    ### Create training data
    train_generator = train_input_fn(os.environ.get('SM_CHANNEL_TRAIN'))
    val_generator = eval_input_fn(os.environ.get('SM_CHANNEL_VAL'))
    
    STEPS_TRAIN=(train_generator.n//train_generator.batch_size) * 3
    STEPS_VAL=val_generator.n//val_generator.batch_size
    
    X_train, y_train = train_generator.next()
    for step in range(STEPS_TRAIN-1):
        X_tmp, y_tmp = train_generator.next()
        X_train = np.vstack([X_train, X_tmp])
        y_train = np.vstack([y_train, y_tmp])
    
    X_val, y_val = val_generator.next()
    for step in range(STEPS_VAL-1):
        X_tmp, y_tmp = val_generator.next()
        X_val = np.vstack([X_val, X_tmp])
        y_val = np.vstack([y_val, y_tmp])
    
    # Start model training
    model = keras_model_fn()
    
    checkpoint = ModelCheckpoint(args.output_dir + '/checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True)
    callbacks = [checkpoint]
    
    # Define class weights, rate multiple_diseases higher
    class_weights = {0: 1.,
                    1: 3.,
                    2: 1.,
                    3: 1.}    
    
    model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=callbacks, class_weight=class_weights, shuffle=True, verbose=2)
    
    # Save final keras models
    model = tf.keras.models.load_model(args.output_dir + '/checkpoint.h5')
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    tf.contrib.saved_model.save_keras_model(model, args.model_output_dir)
    logging.info("Model saved at: {}".format(args.model_output_dir))
