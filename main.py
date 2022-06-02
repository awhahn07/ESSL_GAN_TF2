import numpy as np
from ESSL_GAN_Model import EsslGAN
from models import make_gan_models, EsslMetrics
from IP_DATA import IP_DS
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import datetime

dataSet = IP_DS()
dat, lab = dataSet.remove_null()
weights = tf.constant(dataSet.get_weights(), 'float32')

target_length = np.shape(dat)[1]
number_classes = lab.max() + 1
noise_dimensions = 100

label_smoothing = 0.1

loss_fn = CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)
disc_optimizer = Adam(learning_rate=.0005)
gen_optimizer = Adam(learning_rate=.0015)

discriminator, generator = make_gan_models(
    number_classes=number_classes,
    target_length=target_length,
    noise_dimensions=noise_dimensions
)

gan = EsslGAN(discriminator=discriminator, generator=generator,
              noise_dim=noise_dimensions,  gamma=0.1, class_weights=weights)

gan.compile(discriminator_optimizer=disc_optimizer, generator_optimizer=gen_optimizer, loss_fn=loss_fn)

#%%
lab = to_categorical(lab, num_classes=number_classes)

dat = tf.data.Dataset.from_tensor_slices(dat)
lab = tf.data.Dataset.from_tensor_slices(lab)

ds = tf.data.Dataset.zip((dat, lab)).shuffle(len(dat)).batch(50)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
essl_metrics = EsslMetrics(log_dir=log_dir)
progbar = tf.keras.callbacks.ProgbarLogger(stateful_metrics=["Generator Loss", "Discriminator Loss"])

gan.fit(ds, epochs=10, callbacks=[essl_metrics, progbar], verbose=1)
