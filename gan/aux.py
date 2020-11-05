import numpy as np
import matplotlib.pyplot as plt

def scatter(x, *y, path=None, alpha=0.7, bins=50, s=3):
    plt.figure(figsize=(6,6))

    if bins:
        H, m, n = np.histogram2d(x[:,0], x[:,1], bins=bins)
        m, n = (m[1:] + m[:-1])/2, (n[1:] + n[:-1])/2
        plt.contourf(m, n, H.T, levels=10, cmap='bone_r', extend='both')
    else:
        plt.scatter(x[:,0], x[:,1], label='X', alpha=alpha, linewidth=0, s=s)

    for num,i in enumerate(y,1):
        plt.scatter(i[:,0], i[:,1], label=f'Y{num}', alpha=alpha, linewidth=0, s=s, c=f'C{num+2}')
        plt.plot([],[]) # increment color counter
    if len(y) > 1 or (not bins and len(y) > 0):
        plt.legend()

    # Avoid skewing
    lims = np.array(plt.axis())
    plt.xlim(lims.min(), lims.max())
    plt.ylim(lims.min(), lims.max())
    plt.axis('off')
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200)
    else:
        plt.show()


def gaussian_mixture(n, dim, ngaussians=3, seed=None):
    g = []
    _n = int(n/ngaussians)
    np.random.seed(seed)
    for _ in range(ngaussians):
        g.append(  np.random.uniform(.2, .8, (_n,dim))*np.random.randn(_n, dim) + np.random.uniform(-3, 3, dim)  )
    return np.concatenate(g)


def test_hyperparams(
    repeats,
    savedir,
    X,
    EPOCHS,
    BATCH_SIZE,
    NEURONS    = [16, 64, 128, 256],
    LAYERS     = [2, 4, 8, 16],
    DROPOUT    = [False, 0.1],
    NORM       = [False, True],
    ACTIVATION = ['selu', 'relu', 'tanh'],
    LATENT_DIM = [2, 10, 50, 100, 1000],
    LRATE      = [1e-6, 1e-5, 1e-4, 1e-3],
    w_lambda   = [1e-2, .1, 2, 10]
):
    import os
    import pickle

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    log = []
    for __ in range(repeats):
        print(f"Now at {__}")
        _N =    np.random.choice(NEURONS,   size=2, replace=True)
        _L =    np.random.choice(LAYERS,    size=2, replace=True)
        _D =    np.random.choice(DROPOUT,   size=2, replace=True)
        _NORM = np.random.choice(NORM,      size=2, replace=True)
        _A =    np.random.choice(ACTIVATION,size=2, replace=True)
        _LDIM = np.random.choice(LATENT_DIM)
        _LR   = np.random.choice(LRATE,     size=2, replace=True)
        _lam  = np.random.choice(w_lambda)


        log.append(
            {
            'ldim' : _LDIM,
            'neurons':_N,
            'layers':_L,
            'batch_norm':_NORM,
            'activation':[str(i) for i in _A],
            'dropout' : _D,
            'batch_norm' : _NORM,
            'learning_rate': _LR,
            'w_lambda' : _lam,
            }
        )

        t0 = time()
        gan = GAN(
            _LDIM,
            neurons=_N,
            layers=_L,
            activation=_A,
            dropout = _D,
            batch_norm = _NORM,
            learning_rate = _LR,
            )
        hist, to_gif   = gan.train(X, epochs=EPOCHS, depochs=1, batch_size=BATCH_SIZE, wasserstein=True, lam=_lam)
        log[-1]['time'] = time() - t0

        printhist(hist, path=f"{savedir}/hist{__:03d}.dat")
        with open(f'{savedir}/log.pkl', 'wb') as f:
            pickle.dump(log, f)
        with open(f'{savedir}/frames{__:03d}.pkl', 'wb') as f:
            pickle.dump(to_gif, f)


if __name__ == '__main__':
    from gan import GAN
    # dimX == 2, dimZ == 10
    X     = gaussian_mixture(n=10000,dim=2,ngaussians=4,seed=40)
    gan   = GAN(latent_dim=10, layers=(2,2), neurons=(16,16))

    # Generate 3000 new points prior to training:
    gan.set_models(2) # this is called automatically when gan.train() is called, but we set it before training to demonstrate the generator before training
    Z     = gan.latent_sample(3000)
    Xpred = gan.G.predict(Z)
    scatter(X,Xpred, bins=50, path='../doc/gan.X0.png')

    gan.train(X, epochs=2000)

    # Generate 3000 new points after training:
    Z     = gan.latent_sample(3000)
    Xpred = gan.G.predict(Z)
    scatter(X,Xpred, bins=50, path='../doc/gan.Xt.png')
