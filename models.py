from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow import summary
from torch.utils.tensorboard import SummaryWriter
import itertools

import tensorflow as tf

def make_generator(number_classes, target_length, noise_dimensions):
    model = Sequential(
        [
            layers.Dense(1024, use_bias=False, input_shape=(noise_dimensions + number_classes,), activation='tanh'),
            layers.Dense(6400, activation='tanh'),
            layers.BatchNormalization(),
            layers.Reshape((50, 128)),
            layers.UpSampling1D(size=2),
            layers.BatchNormalization(),
            layers.Conv1D(64, 5, padding='same', activation='tanh'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(1, 5, padding='same', activation='tanh'),
            layers.Flatten(),
            layers.Dense(target_length),
            layers.Reshape([target_length, 1]),
        ],
        name="generator"
    )
    model.summary()
    return model


def make_discriminator(number_classes, target_length):
    """Make Discriminator Model for ESSL GAN
    NOTE:
        Must include number of true classes (n) plus 1
        number_classes = n + 1 where n is number of true classes
    """
    model = Sequential(
        [
            layers.Conv1D(64, 5, padding='same', activation='relu',input_shape=[target_length, 1], name='CONV1'),
            layers.Conv1D(64, 5, padding='valid', activation='relu', name='CONV2'),
            layers.MaxPool1D(),
            layers.Conv1D(64, 5, padding='valid', activation='relu', name='CONV3'),
            layers.Conv1D(64, 5, padding='valid', activation='relu', name='CONV4'),
            layers.MaxPool1D(),
            layers.Flatten(),
            layers.Dense(1024, name='DEN1'),
            layers.Dense(number_classes),
        ],
        name="discriminator",
    )
    model.summary()
    return model


def make_gan_models(number_classes, target_length, noise_dimensions):
    d = make_discriminator(
        number_classes=number_classes,
        target_length=target_length,
    )
    g = make_generator(
        number_classes=number_classes,
        target_length=target_length,
        noise_dimensions=noise_dimensions,
    )
    return d, g


class EsslMetrics(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer_1 = SummaryWriter(log_dir=self.log_dir)
        self.step = 0

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs):
        dg_loss = dict(itertools.islice(logs.items(), 2))
        d_losses = dict(itertools.islice(logs.items(), 1, 4))
        d_mets = dict(itertools.islice(logs.items(), 4, 6))
        g_mets = dict(itertools.islice(logs.items(), 6, None))

        if batch % 20 == 0:
            self.writer_1.add_scalars(f'losses/Model Losses', dg_loss, self.step)
            self.writer_1.add_scalars(f'losses/D Losses', d_losses, self.step)
            self.writer_1.add_scalars(f'metrics/Discriminator Metrics', d_mets, self.step)
            self.writer_1.add_scalars(f'metrics/Generator Metrics', g_mets, self.step)
            self.step += 1

    def on_train_end(self, logs=None):
        self.writer_1.close()
