class System:
    """Class implementing the LQG problem defined by the provided arguments
    Upon initialzation the optimal solution is obtained by running the Riccati equations
    and stored as instance attributes L, P, K, S
    Parameters
    ----------
    A : ndarray, shape (m,m)
        state-transition model
    B : ndarray, shape (m,k)
        control-input model
    C : ndarray, shape (n,m)
        observation model
    V : ndarray, shape (m,m)
        covariance of process noise
    W : ndarray, shape (n,n)
        covariance of observation noise
    Q : ndarray, shape (m,m) [or (T, m, m) if time-dependent]
        matrix in x-dependent part of cost function x'Qx
    R : ndarray, shape (k,k)
        matrix in u-dependent part of cost function u'Ru
    T : int
        episode length, including 0, i.e. t = 0,...,T-1
    P0: ndarray, shape (m,m)
        initial (a priori) estimate covariance
    """

    def __init__(self, A, B, C, V, W, Q, R, T, P0=0):
        self.A, self.B, self.C, self.V, self.W, self.Q, self.R, self.T = A, B, C, V, W, Q, R, T
        self.cholV = np.linalg.cholesky(V)
        self.cholW = np.linalg.cholesky(W)
        self.n, self.m = C.shape  # observed, latent dim
        self.L, self.P, self.K, self.S = Riccati(A, B, C, V, W, Q, R, T, P0)

def PGwithSysID(self, seed, eta=3e-3, etaK=2.2e-6, momentum=.99, sigma=.2, delay=1,
                    episodes=1000, x0=[-1, 0], ABCLhat=None, Khat=None, returnX=False,
                    episodes2=1000, EMAcoeff=1, multiplicative_noise=False):
     """Perform system identification, i.e. learning of A,B,C,L,
           simultaneously with cost minimization using policy gradient method GPOMDP
        Parameters
        ----------
        seed : int
            random seed
        eta : float or 4-tuple of floats
            learning rate(s) for filter weigths A,B,C,L
        etaK : float
            learning rate for control weigths K
        momentum : float
            Momentum [0,1) for control weigths K
        sigma : float
            standard deviation of the Gaussian noise of the actor
        delay : int
            amount of measurement delay
        episodes : int
            number of episodes
        x0 : m-tuple or tuple of m-tuples
            initial position(s); looped over if multiple are provided
        ABCLhat : tuple of ndarrays with shapes ((m,m), (m,k), (n,m), (m,n))
            The initial estimates for weight matrices A, B, C, and L.
        Khat : ndarray, shape (k,m)
            initial estimates for control weight matrix K
        returnX : bool
            whether to return all trajectories executed during training
        episodes2 : int
            number of episodes run after training without noise in actor
        EMAcoeff : float
            coefficient of exponential moving average for reward baseline in PG gradient estimate
        Returns
        -------
        J, J2, mse, Ahat, Bhat, Chat, Lhat, Khat [, Xs]
            J   costs for each episode during training
            J2  costs after training for running episodes2 episodes without noise in actor
            mse  mean squared errors for each episode
            Ahat, Bhat, Chat, Lhat, Khat  final weight matrices after training
            Xs  trajectories for each episode
        """
        A, B, C, Q, R, T = self.A, self.B, self.C, self.Q, self.R, self.T
        cholV, cholW = self.cholV, self.cholW
        n, m = C.shape
        if Khat is None:
            Khat = np.zeros((1, m))
        else:
            Khat = Khat.copy()
        etaA, etaB, etaC, etaL = [eta] * 4 if isinstance(eta, (int, float)) else eta
        np.random.seed(seed)
        if ABCLhat is None:
            Ahat, Bhat, Chat, Lhat = [.1*np.random.randn(*a_.shape) for a_ in (A, B, C, C.T)]
        elif type(ABCLhat) == str:
            Ahat, Bhat, Chat, Lhat = eval(ABCLhat)
        else:
            Ahat, Bhat, Chat, Lhat = deepcopy([a.astype(float) for a in ABCLhat])
        mse = []
        X = np.zeros((T, m))
        U = np.zeros((T-1, B.shape[1]))
        Y = np.zeros((T, n))
        X[0] = x0 if len(np.shape(x0)) == 1 else x0[seed % len(x0)]
        Y[0] = C.dot(X[0])
        Xhat = X.copy()
        J = []
        grad = np.zeros(B.T.shape)
        if returnX:
            Xs = np.empty((episodes, T, m))
        avgJ = 0
        for j in range(episodes):
            Xhat = np.zeros((T, m))
            U = np.zeros((T-1, B.shape[1]))
            if len(np.shape(x0)) == 2:
                X[0] = x0[j % len(x0)]
                Y[0] = C.dot(X[0])
            Xhat[0] = np.linalg.lstsq(Chat, Y[0], rcond=None)[0]
            z = np.zeros(grad.shape)
            e = np.zeros((1+delay, n))
            for t in range(T-1):
                xi = sigma*np.random.randn(B.shape[1])
                U[t] = -Khat.dot(Xhat[t]) + xi
                z += np.outer(xi, Xhat[t])
                noise = cholV.dot(np.random.randn(m))
                if multiplicative_noise:
                    noise *= np.sqrt(U[t].dot(U[t])) * 200
                X[t+1] = A.dot(X[t]) + B.dot(U[t]) + noise
                Y[t+1] = C.dot(X[t+1]) + cholW.dot(np.random.randn(n))
                e[1:] = e[:-1]
                td = t-delay+1
                if td >= 0:
                    e[0] = Y[td] - Chat.dot(Xhat[td])
                Le = Lhat.dot(e[0])
                Xhat[t+1] = Ahat.dot(Xhat[t]) + Bhat.dot(U[t]) + Le

                Ahat += etaA*np.outer(Le, Xhat[t-delay])
                Bhat += etaB*np.outer(Le, U[t-delay])
                Lhat += etaL*np.outer(Le, e[-1])
                Chat += etaC*np.outer(e[0], Xhat[t+1-delay])

                cost = X[t+1].dot(Q).dot(X[t+1]) + U[t].dot(R).dot(U[t]) - avgJ
                grad = momentum*grad + cost * z
                Khat += etaK * grad
            J.append(np.trace(X.T.dot(X).dot(Q)) + np.trace(U.T.dot(U).dot(R)))
            if EMAcoeff != 1:
                avgJ = EMAcoeff*avgJ + (1-EMAcoeff)*J[-1]/T
            e = Y-Xhat.dot(Chat.T)
            mse.append(np.trace(e.T.dot(e))/T)
            if returnX:
                Xs[j] = X
        J2 = []
        Xhat[0] = np.linalg.lstsq(Chat, Y[0], rcond=None)[0]
        for _ in range(episodes2):
            e = np.zeros((1+delay, n))
            for t in range(T-1):
                U[t] = -Khat.dot(Xhat[t])
                noise = cholV.dot(np.random.randn(m))
                if multiplicative_noise:
                    noise *= np.sqrt(U[t].dot(U[t])) * 200
                X[t+1] = A.dot(X[t]) + B.dot(U[t]) + noise
                Y[t+1] = C.dot(X[t+1]) + cholW.dot(np.random.randn(n))
                e[1:] = e[:-1]
                td = t-delay+1
                if td >= 0:
                    e[0] = Y[td] - Chat.dot(Xhat[td])
                Xhat[t+1] = Ahat.dot(Xhat[t]) + Bhat.dot(U[t]) + Lhat.dot(e[0])
            J2.append(np.trace(X.T.dot(X).dot(Q)) + np.trace(U.T.dot(U).dot(R)))
        if returnX:
            return J, J2, mse, Ahat, Bhat, Chat, Lhat, Khat, Xs
        else:
            return J, J2, mse, Ahat, Bhat, Chat, Lhat, Khat