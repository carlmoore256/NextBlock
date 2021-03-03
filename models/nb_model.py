# NextBlock main CNN model

from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import models.unet_fft as unet_fft

class NB_Model():

    def __init__(self, 
                input_shape, 
                learning_rate,
                loss='mse', 
                model_dir='./saved/', 
                chkpoint_dir='/tmp/checkpoint',
                save_checkpoints=True):

        self.input_shape = input_shape
        self.model_dir = model_dir
        self.chkpoint_dir = chkpoint_dir
        self.loss = loss
        self.save_checkpoints = save_checkpoints
        self.checkpoint = ModelCheckpoint(
            chkpoint_dir,
            monitor='val_loss',
            verbose=1,
            save_freq=1000,
            save_best_only=True)


    def save(self):
        save_model(self.model, self.model_dir)

    def load(self, path, lr):
        print(f"loading model {path}")
        self.model = load_model(path)
        self.compile(lr, self.loss)

    def compile(self, lr, loss='mse', metrics=['mse']):
        self.optimizer = Adam(lr)
        self.model.compile(
            optimizer=self.optimizer, 
            loss=loss, 
            metrics=metrics)

    def create_unet_fft(self, lr, filters, kernel_size, bottleneck, use_bias=False, strides=2, activation='tanh'):
        self.model = unet_fft.create_model(self.input_shape,
                        filters, kernel_size, bottleneck,
                        strides, activation, use_bias)
        self.compile(lr, self.loss)
        self.model_summary()

    def model_summary(self):
        print(self.model.summary())

    def fit(self, generators, epochs, callbacks=[]):
        train_gen = generators.train_DG
        val_gen = generators.val_DG

        if self.save_checkpoints:
            callbacks.append(self.checkpoint)

        steps_per_epoch = train_gen.num_examples//train_gen.batch_size
        validation_steps = val_gen.num_examples//val_gen.batch_size

        history = self.model.fit(train_gen.generate(),
                    validation_data=val_gen.generate(),
                    validation_steps=validation_steps,
                    epochs=epochs, 
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[callbacks])

        return history
