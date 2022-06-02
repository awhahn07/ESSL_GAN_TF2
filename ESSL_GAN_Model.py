# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:59:18 2022

@author: m1226
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model


@tf.function
def make_class_weights(labels, class_weights):
    if class_weights is None:
        return 1
    else:
        ind = tf.math.argmax(labels, axis=1)
        weights = tf.map_fn(lambda t: class_weights[t], ind, fn_output_signature=tf.float32)
        return weights


class EsslGAN(Model):

    def __init__(self, generator, discriminator, noise_dim=100, gamma=0.25, class_weights=None):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.gamma = gamma
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.loss_fn = None
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="Generator Loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="Discriminator Loss")
        self.d_real_loss_tracker = tf.keras.metrics.Mean(name="Discriminator Loss, Real")
        self.d_syn_loss_tracker = tf.keras.metrics.Mean(name="Discriminator Loss, Synthetic")
        self.pDu = tf.keras.metrics.CategoricalAccuracy(name="Discriminator Accuracy, Unsupervised")
        self.pDs = tf.keras.metrics.CategoricalAccuracy(name="Discriminator Accuracy, Supervised")
        self.pGe = tf.keras.metrics.CategoricalAccuracy(name="Generator Error")
        self.pGp = tf.keras.metrics.CategoricalAccuracy(name="Generator Precision")
        self.synthetic_loss = 0
        self.real_loss = 0
        self.class_weights = class_weights

    def compile(self, discriminator_optimizer, generator_optimizer, loss_fn):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_fn = loss_fn

    # TODO add additional losses/metrics
    @property
    def metrics(self):
        return [
            self.gen_loss_tracker, self.disc_loss_tracker,
            self.d_real_loss_tracker, self.d_syn_loss_tracker,
            self.pDu, self.pDs, self.pGp
        ]

    def train_step(self, data):

        real_data, labels = data

        '''
        Generate a vector of size([Batch_Size,Noise_Dim+Num_Classes]).
        This vector has random noise of size Noise_Dim with a Sparse Vector
        that corresponds to a randomly selected label appended. Each vector 
        Z fed into Generator G has a size of Noise_Dim + Class Size.
        Then it is batched into Batch_Size
        '''
        batch_size = tf.shape(real_data)[0]
        num_classes = self.discriminator.output_shape[-1]

        # Labels for generator loss, correspond to real classes and conditional vectors
        gen_labels = tf.random.uniform(
            minval=0,
            maxval=num_classes - 2,  # only generate labels corresponding to real classes, no n+1
            shape=(batch_size,),
            dtype=tf.int32)

        conditional_vector = tf.one_hot(gen_labels, depth=num_classes)

        # Generate noise vector of size batch_size, noise_dim
        noise_vector = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Input vector to generator, labels embedded into noise
        batch_noise_labeled = tf.concat([noise_vector, conditional_vector], axis=1)

        # Labels for synthetic Discriminator Loss
        syn_loss_labels = tf.cast(tf.ones(batch_size) * num_classes - 1, tf.int32)
        syn_loss_labels = tf.one_hot(syn_loss_labels, depth=num_classes)

        ''' This is the Tensorflow implementation to control at lower levels
        the forward propogation and subsequent backprop of gradients.
        The Training step is implemented in a context manager with statement, 
        where the gradients of D and G are stored in "GradientTapes" '''

        real_loss_weights = make_class_weights(labels, class_weights=self.class_weights)
        syn_loss_weights = make_class_weights(conditional_vector, class_weights=self.class_weights)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            '''Generate Images from G network'''
            generated_images = self.generator(batch_noise_labeled, training=True)

            '''Evaluate both Real and Synthetic images, D(x) and D(G(z))'''
            real_predictions = self.discriminator(real_data, training=True)
            synthetic_predictions = self.discriminator(generated_images, training=True)

            '''Compute Generator Loss using Categorical Crossentropy'''
            gen_loss = self.loss_fn(conditional_vector,
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

            disc_loss = ((1 - self.gamma) * real_loss) + (self.gamma * synthetic_loss)

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
        self.d_real_loss_tracker.update_state(real_loss)
        self.d_syn_loss_tracker.update_state(synthetic_loss)
        self.pDu.update_state(syn_loss_labels, synthetic_predictions)
        self.pDs.update_state(labels, real_predictions)
        self.pGp.update_state(conditional_vector, synthetic_predictions)

        return {m.name: m.result() for m in self.metrics}
