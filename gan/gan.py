from time import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

class GAN:
    """ Implementation of a Generative Adversatial Network in TensorFlow """

    def __init__(self,
        latent_dim    = 10,
        layers        = (5, 5),
        neurons       = (128, 128),
        activation    = ('selu', 'selu'),
        optimizers    = (None, None),
        batch_norm    = (False, False),
        dropout       = (False, False),
        learning_rate = (1e-4, 1e-4),
        ):
        """
        Initialize a new Generator-Discriminator network.
        All parameters other than the `latent_dim` are passed as a tuple
        where the first and second elements are generator and discriminator
        network parameters respectively.

        Parameters
        ----------
        latent_dim : int > 0
            The dimensionality of the latent vector `z`, input to the generator network
        layers : int > 0
            Number of dense layers to use in the network
        neurons : int > 0
            Number of neurons to use for each layer
        activation : keras.layers.Activation or str
            Activation functions to use with each layer
        optimizers : tf.optimizers or str
            Optimizers to use, defaults to `Adam`
        batch_norm : bool
            Use batch normalization at each layer
        dropout : float >= 0
            Use dropout at each layer, disabled if `0`
        learning_rate: float > 0
            Optimizer learning rate
        """
        self.latent_dim  = latent_dim
        self.ga, self.da = activation
        self.gl, self.dl = layers
        self.gn, self.dn = neurons
        self.go, self.do = optimizers
        self.gb, self.db = batch_norm
        self.gd, self.dd = dropout
        self.glr,self.dlr = learning_rate

        self.G, self.D  = None, None


    def get_generator(self, original_dim):
        # Activation function, neurons, layers, batch normalize, dropout
        a, n, l, b, d = self.ga, self.gn, self.gl, self.gb, self.gd
        # Generator Model: Latent dim in > original dim out
        generator = tf.keras.Sequential(name='generator')
        generator.add(keras.Input(shape=(self.latent_dim,)))
        for _ in range(l):
            generator.add(keras.layers.Dense(n, use_bias=True))
            if b:
                generator.add(keras.layers.BatchNormalization())
            generator.add(keras.layers.Activation(a))
            if d:
                generator.add(keras.layers.Dropout(d))
        generator.add(keras.layers.Dense(original_dim, use_bias=True))
        return generator


    def get_discriminator(self, original_dim):
        # Activation function, neurons, layers, dropout
        a, n, l, b, d = self.da, self.dn, self.dl, self.db, self.dd
        # Discriminator Model: original dim in > bool out (real,fake)
        discriminator = tf.keras.Sequential(name='discriminator')
        discriminator.add(keras.Input(shape=(original_dim,)))
        for _ in range(l):
            discriminator.add(keras.layers.Dense(n, activation=a))
            if b:
                discriminator.add(keras.layers.BatchNormalization())
            discriminator.add(keras.layers.Activation(a))
            if d:
                discriminator.add(keras.layers.Dropout(d))
        discriminator.add(keras.layers.Dense(1))
        return discriminator


    def train(self, X, epochs, depochs=1, batch_size=128, wasserstein=True, lam=1., verbose=True):
        """
        Train the GAN.

        Parameters
        ----------
        epochs : int
            Train for a total number of epochs
        depochs : int
            In each epoch, train the discriminator `depchs` times
        batch_size : int
            Train `X` in batches of size `batch_size`
        wasserstein : bool
            Use the Gradient-Penalty Wasserstein loss (https://arxiv.org/pdf/1704.00028.pdf)
            instead of the Jensen-Shannon divergence
        lam : float >= 0
            Gradient penalty regularization factor for the Wasserstein distance loss
        """
        X   = tf.cast(X, tf.keras.backend.floatx())
        dim = np.prod(X[0].shape)
        self.set_models(dim)

        gopt = self.go or tf.optimizers.Adam(self.glr)
        dopt = self.do or tf.optimizers.Adam(self.dlr)

        dloss, gloss = (self.dloss_w, self.gloss_w) if wasserstein else (self.dloss, self.gloss)

        @tf.function
        def train_step(X):
            # Train the Discriminator `depochs` times
            with tf.GradientTape() as dtape:
                _dloss = []
                for _ in range(depochs):
                    Z    = self.latent_sample(len(X))
                    Xhat = self.G(Z,    training=True)
                    yhat = self.D(Xhat, training=True) # Generated vectors to 0
                    y    = self.D(X,    training=True) # Real vectors to  1
                    _loss = dloss(y, yhat)
                    if wasserstein:
                        _loss += lam*self.gradient_penalty(self.D, X, Xhat)
                    _dloss.append(_loss)
                _dloss = tf.reduce_mean(_dloss)
            dgrads = dtape.gradient(_dloss,  self.D.trainable_variables)
            dopt.apply_gradients(zip(dgrads, self.D.trainable_variables))

            # Train the Gnerator
            with tf.GradientTape() as gtape:
                Z    = self.latent_sample(len(X))
                Xhat = self.G(Z,    training=True)
                yhat = self.D(Xhat, training=True) # Generated vectors to 0
                _gloss = gloss(yhat)
            ggrads = gtape.gradient(_gloss,  self.G.trainable_variables)
            gopt.apply_gradients(zip(ggrads, self.G.trainable_variables))

            return float(_dloss), float(_gloss)

        X_ds = tf.data.Dataset.from_tensor_slices(X).shuffle(len(X)).batch(batch_size)
        hist = {'dloss':[], 'gloss':[]}
        t    = time()
        to_gif = []
        for epoch in range(epochs):
            for x in X_ds:
                _dloss, _gloss = train_step(x)

            if verbose:
                if (epoch+1) % 100 == 0 or epoch == 0:
                    print(f"Epoch: {epoch+1:04d}, DLoss: {_dloss:.3e}, GLoss: {_gloss:.3e}, {(time()-t)/(epoch+1):.3f} s/step")
                elif (epoch+1) % 10 == 0:
                    print('.', end='', flush=True)

            to_gif.append(self.G(self.latent_sample(1000)))
            hist['dloss'].append(_dloss)
            hist['gloss'].append(_gloss)

        if verbose:
            print()

        return hist, to_gif


    def latent_sample(self,n):
        # Generator input
        return tf.random.normal((n, self.latent_dim))

    def set_models(self,dim):
        if self.G is None:
            self.G = self.get_generator(dim)
        if self.D is None:
            self.D = self.get_discriminator(dim)

    @staticmethod
    def gradient_penalty(model, reals, fakes):
        alpha  = tf.random.uniform(reals.shape, 0., 1.)
        interp = alpha*reals + (1-alpha)*fakes
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = model(interp, training=True)
        grad = tape.gradient(pred, interp)
        gp   = tf.reduce_mean( (tf.norm(grad, axis=-1)-1)**2 )
        return gp

    @staticmethod
    def gloss_w(fakes):
        return -tf.reduce_mean(fakes) # ideally -1

    @staticmethod
    def dloss_w(reals, fakes):
        real_loss = tf.reduce_mean(reals) # ideally 1
        fake_loss = tf.reduce_mean(fakes) # ideally 0
        return fake_loss - real_loss

    @staticmethod
    def gloss(fakes):
        return tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fakes), fakes)

    @staticmethod
    def dloss(reals, fakes):
        real_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like (reals), reals)
        fake_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fakes), fakes)
        return real_loss + fake_loss

    @staticmethod
    def printhist(hist, path):
        # Print the output of `train()` to path
        with open(path, 'w') as f:
            print("# GLoss DLoss", file=f)
            for g,d in zip(hist['gloss'], hist['dloss']):
                print(f"{g:.3e} {d:.3e}", file=f)
