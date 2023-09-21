import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import  Adamax
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from src.ChickenDisease.config import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config=config

    
    def get_base_model(self):
        self.base_model=tf.keras.applications.EfficientNetB7(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.base_model)

    @staticmethod
    def prepare_full_model(model,classes,learning_rate):

        model.trainable= True
        x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(model.output)
        x = Dense(1024, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
        x=Dropout(rate=.3, seed=123)(x)
        x = Dense(128, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                        bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
        x=Dropout(rate=.45, seed=123)(x)        
        output=Dense(classes, activation='softmax')(x)
        model=Model(inputs=model.input, outputs=output)
        
        full_model=tf.keras.models.Model(
            inputs=model.input,
            outputs=output
        )

        full_model.compile(
            optimizer=Adamax(learning_rate=learning_rate),
            loss=CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model= self.prepare_full_model(
            model=self.base_model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_model_path, model=self.full_model)

    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        model.save(path)
    
