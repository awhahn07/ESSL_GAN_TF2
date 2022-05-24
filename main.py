import numpy as np
from ESSL_GAN_Model import EsslGAN
from models import make_gan_models
from IP_DATA import IP_DS
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

dataSet = IP_DS()
dat, lab = dataSet.scaled()

target_length = np.shape(dat)[1]
number_classes = lab.max() + 1
noise_dimensions = 100

label_smoothing = 0.1

loss_fn = CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)
disc_optimizer = Adam(learning_rate=5e-4)
gen_optimizer = Adam(learning_rate=5e-4)

discriminator, generator = make_gan_models(
    number_classes=number_classes,
    target_length=target_length,
    noise_dimensions=noise_dimensions
)

gan = EsslGAN(discriminator=discriminator, generator=generator, noise_dim=noise_dimensions)

gan.compile(discriminator_optimizer=disc_optimizer, generator_optimizer=gen_optimizer, loss_fn=loss_fn)

#%%
lab = to_categorical(lab, num_classes=number_classes)
gan.fit(x=dat, y=lab, batch_size=50)