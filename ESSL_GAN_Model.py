# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:59:18 2022

@author: m1226
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

def make_class_weights(labels, class_weights):
    if class_weights == None:
        return 1
    ind = [np.argmax(lab) for lab in labels]
    weights = [class_weights[i] for i in ind]
    return weights


'''
def generator_loss(generated_output, labels, weights=1):
    gen_loss = softmax(labels, generated_output, label_smoothing=0.1, weights=weights)
    return gen_loss

def discriminator_loss(real_output, generated_output, train_lab, weights=1):
    """Computes discriminator loss."""
    # Compute real portion of loss
    real_loss = softmax(train_lab, real_output, label_smoothing=0.1, weights=weights)

    # Compute real portion of loss
    gen_loss_labels = tf.ones(self.batch_size) * (self.nb_class - 1)
    gen_loss_labels = k.utils.to_categorical(gen_loss_labels, num_classes=self.nb_class)

    generated_loss = softmax(gen_loss_labels, generated_output, label_smoothing=0.1)

    total_loss = real_loss + self.gamma * generated_loss
    return total_loss, real_loss, generated_loss
    
def generator_vector(self):
    """ Generate vector of random ints in range of all valid TRUE classes,

    i.e. Classes C_0...C_n. Fake class is C_n+1  """

    gen_labels = np.random.randint(0, self.nb_class - 1, size=self.batch_size)
    gen_labels = k.utils.to_categorical(gen_labels, num_classes=self.nb_class)

    # Generate noise vector of size batch_size,noise_dim
    noise_vector = np.random.normal(size=[self.batch_size, self.noise_dim]).astype('float32')

    # Return labels encoded into noise
    return np.append(noise_vector, gen_labels, axis=1), gen_labels
'''


class EsslGAN(Model):

    def __init__(self, generator, discriminator, noise_dim=100):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.gamma = 0.25
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.loss_fn = None
        self.gen_loss_tracker = tf.keras.metrics.Mean(name='Generator Loss')
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="Discriminator Loss")

    def compile(self, discriminator_optimizer, generator_optimizer, loss_fn):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_fn = loss_fn

    # TODO add additional losses/metrics
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def train_step(self, data):

        real_data, labels = data

        '''
        Generate a vector of size([Batch_Size,Noise_Dim+Num_Classes]).
        This vector has random noise of size Noise_Dim with a Sparse Vector
        that corresponds to a randomly selected label appended. Each vector 
        Z fed into Generator G has a size of Noise_Dim + Class Size.
        Then it is batched into Batch_Size
        '''
        #TODO need to figure out logic for batch size
        batch_size = 50 # tf.shape(real_data)[0]
        num_classes = self.discriminator.output_shape[-1]

        # Labels for generator loss, correspond to real classes and conditional vectors
        gen_labels = np.random.randint(0, num_classes - 1, size=batch_size)
        gen_labels = to_categorical(gen_labels, num_classes=num_classes)

        # Generate noise vector of size batch_size, noise_dim
        noise_vector = np.random.normal(size=[batch_size, self.noise_dim]).astype('float32')

        # Input vector to generator, labels embedded into noise
        batch_noise_labeled = np.append(noise_vector, gen_labels, axis=1)

        # Labels for synthetic Discriminator Loss
        syn_loss_labels = np.ones(batch_size) * (num_classes - 1)
        syn_loss_labels = to_categorical(syn_loss_labels, num_classes=num_classes)

        ''' This is the Tensorflow implementation to control at lower levels
        the forward propogation and subsequent backprop of gradients.
        The Training step is implemented in a context manager with statement, 
        where the gradients of D and G are stored in "GradientTapes" '''

        real_loss_weights = make_class_weights(labels, class_weights=None)
        syn_loss_weights = make_class_weights(gen_labels, class_weights=None)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            '''Generate Images from G network'''
            generated_images = self.generator(batch_noise_labeled, training=True)

            '''Evaluate both Real and Synthetic images, D(x) and D(G(z))'''
            real_predictions = self.discriminator(real_data, training=True)
            synthetic_predictions = self.discriminator(generated_images, training=True)

            '''Compute Generator Loss using Categorical Crossentropy'''
            gen_loss = self.loss_fn(gen_labels,
                                    synthetic_predictions,
                                    sample_weight=syn_loss_weights
                                    )

            ''' Compute Discriminator Loss using categorical crossentropy
            Combine real + synthetic loss for total discriminator loss '''
            real_loss = self.loss_fn(labels,
                                     real_predictions,
                                     sample_weight=real_loss_weights
                                     )

            synthetic_loss = self.loss_fn(syn_loss_labels,
                                          synthetic_predictions,
                                          sample_weight=syn_loss_weights
                                          )

            disc_loss = real_loss + self.gamma * synthetic_loss

        '''calculate gradients'''
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)

        '''Using Adam optimizer, backpropogate gradients'''
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_weights)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_weights)
        )

        # Monitor loss.
        self.gen_loss_tracker.update_state(gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }
