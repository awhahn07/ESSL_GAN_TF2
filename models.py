from tensorflow.keras import layers 
from tensorflow.keras import Sequential


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
    """Make Discriminator and Generator models for ESSL GAN
    NOTE:
        Must include number of true classes (n) plus 1
        number_classes = n + 1 where n is number of true classes
    """
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
