from copy import deepcopy
import multiprocessing
import numpy as np


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """parallel map that works with multiple arguments and anonymous (lambda) functions
       in newer python versions one could maybe use multiprocessing.Pool().starmap() instead
    """
    def fun(f, q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()
    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]


def Riccati(A, B, C, V, W, Q, R, T, P0=0):
    """Runs the forward and backward Ricatti equations for the defined LQG problem

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

    Returns
    -------
    L, P, K, S
        Tuple containing the final 3d-arrays:
        Kalman/filter gain L
        (a priori) estimate covariance P
        feedback/control gain K
        value S
    """
    n, m = C.shape  # observed, latent dim
    assert Q.shape in ((m, m), (T, m, m))
    L = np.zeros((T, m, n))
    P = np.zeros((T, m, m))
    if P0 != 0:
        P[0] = P0
    K = np.zeros((T, B.shape[1], m))
    S = np.zeros((T, m, m))
    S[-1] = Q if Q.ndim == 2 else Q[-1]
    for t in range(T-1):
        L[t] = P[t].dot(C.T).dot(np.linalg.inv(C.dot(P[t]).dot(C.T) + W))
        P[t+1] = A.dot(np.eye(m)-L[t].dot(C)).dot(P[t]).dot(A.T) + V
        K[T-2-t] = np.linalg.solve(B.T.dot(S[T-1-t]).dot(B) + R, B.T.dot(S[T-1-t]).dot(A))
        S[T-2-t] = A.T.dot(S[T-1-t]).dot(A) - A.T.dot(S[T-1-t]).dot(B).dot(K[T-2-t]) + \
            (Q if Q.ndim == 2 else Q[T-2-t])
    t += 1
    L[t] = P[t].dot(C.T).dot(np.linalg.inv(C.dot(P[t]).dot(C.T)+W))
    return L, P, K, S


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

    def LQG(self, seed, ABCLhat=None, delay=1, asymptoticL=False,
            actor=None, update_current=True, x0=[-1, 0], T=None,
            multiplicative_noise=False):
        """Run episode using optimal LQG, and optionally additionally
        return trajectories using learned params and specified actor.

        Parameters
        ----------
        seed : int
            random seed
        ABCLhat : tuple of ndarrays
            The estimates for matrices A, B, C, and L.
        delay : int
            amount of measurement delay
        asymptoticL : bool
            whether to use use asymptotic Kalman gain L
        actor : string, optional
            equation for U[t], default is the optimal LQG controller "-K[t].dot(Xpost[t])"
        update_current : bool
            True:  update current estimate using past prediction error
            False: update past estimate using past prediction error
                   and predict current using model
        x0 : array-like, shape (m,)
            initial state
        T : int
            episode length, defaults to the one used when creating the object

        Returns
        -------
        U, X, Y, Xpre, Xpost [, Xhat , Xpred]
        """
        A, B, C, L, K = self.A, self.B, self.C, self.L, self.K
        cholV, cholW = self.cholV, self.cholW
        if T is None:
            T = self.T
        n, m = C.shape
        np.random.seed(seed)
        X = np.zeros((T, m))
        U = np.zeros((T-1, B.shape[1]))
        Y = np.zeros((T, n))
        X[0] = x0
        Y[0] = C.dot(X[0])
        Xpre = X.copy()
        Xpost = X.copy()
        Xpost[0] += L[0].dot(Y[0] - C.dot(Xpre[0]))
        if ABCLhat is not None:
            Ahat, Bhat, Chat, Lhat = ABCLhat
            Xhat = X.copy()
            Xhat[0] = np.linalg.lstsq(Chat, Y[0], rcond=None)[0]
            if update_current:
                Xpred = None
            else:
                Xpred = Xhat.copy()
        for t in range(T - 1):
            U[t] = -K[t].dot(Xpost[t]) if actor is None else eval(actor)
            noise = cholV.dot(np.random.randn(m))
            if multiplicative_noise:
                noise *= np.sqrt(U[t].dot(U[t])) * 200
            X[t+1] = A.dot(X[t]) + B.dot(U[t]) + noise
            Y[t+1] = C.dot(X[t+1]) + cholW.dot(np.random.randn(n))
            Xpre[t+1] = A.dot(Xpost[t]) + B.dot(U[t])
            Xpost[t+1] = Xpre[t+1] + (L[-1] if asymptoticL else L[t+1]
                                      ).dot(Y[t+1] - C.dot(Xpre[t+1]))
            if ABCLhat is not None:
                Xhat[t+1] = Ahat.dot(Xhat[t]) + Bhat.dot(U[t])
                if update_current:
                    td = t - delay + 1
                    if td >= 0:
                        Xhat[t+1] += Lhat.dot(Y[td] - Chat.dot(Xhat[td]))
                else:
                    Xhat[t+1] += (Lhat if Lhat.ndim == 2 else Lhat[t]).dot(
                        Y[t] - Chat.dot(Xhat[t]))
                    td = max(0, t-delay+2)
                    pred = Xhat[td]
                    for tt in range(td, t+1):
                        pred = Ahat.dot(pred) + Bhat.dot(U[tt])
                    Xpred[t+1] = pred

        if ABCLhat is not None:
            return U, X, Y, Xpre, Xpost, Xhat, Xpred
        else:
            return U, X, Y, Xpre, Xpost

    def SysID(self, Ahat, Bhat, Chat, Lhat, eta=3e-3, delay=1, episodes=2500,
              sigma=.5, init_seed=0, x0=[-1, 0], useL=True, verbose=False):
        """Perform system identification, i.e. learning of A,B,C,L, using
           (preconditioned) stochastic gradient descent

        Parameters
        ----------
        Ahat : ndarray, shape (m,m)
        Bhat : ndarray, shape (m,k)
        Chat : ndarray, shape (n,m)
        Lhat : ndarray, shape (m,n)
            The initial estimates for matrices A, B, C, and L.
        eta : float or tuple of floats (etaA, etaB, etaC, etaL)
            learning rate(s)
        delay : int
            Delay >=1
        episodes : int
            Number of episodes
        sigma : float
            standard deviation of Gaussian noise of actor
        init_seed : int
            initial random seed
        x0 : m-tuple or tuple of m-tuples
            initial state(s); looped over if multiple are provided
        useL: boolean
            True: use local learning rules L e_t v_{t-\tau}', v \in {\hat{x},u,e}
            False: use non-local SGD rules C'e_t v_{t-\tau}', v \in {\hat{x},u,e}
        verbose:
            if True also returns Lhat for each episode and eCLe for each time step

        Returns
        -------
        Ahat, Bhat, Chat, Lhat, mse [, Lhats, eCLe]
            Tuple containing the final matrices Ahat, Bhat, Chat, Lhat,
            as well as the mean squared errors for each episode.
            If verbose then also Lhat for each episode and eCLe for each time step.
        """

        Ahat, Bhat, Chat, Lhat = [a.astype(float) for a in (Ahat, Bhat, Chat, Lhat)]
        A, B, C, T = self.A, self.B, self.C, self.T
        cholV, cholW = self.cholV, self.cholW
        n, m = C.shape
        etaA, etaB, etaC, etaL = [eta]*4 if isinstance(eta, (int, float)) else eta
        mse = []
        if verbose:
            Ls, eCLe = [], []
        for seed in range(init_seed, init_seed+episodes):
            np.random.seed(seed)
            X = np.zeros((T, m))
            Y = np.zeros((T, n))
            X[0] = x0 if len(np.shape(x0)) == 1 else x0[seed % len(x0)]
            Y[0] = C.dot(X[0])
            U = np.zeros((T-1, 1))
            Xhat = X.copy()
            Xhat[0] = np.linalg.lstsq(Chat, Y[0], rcond=None)[0]
            e = np.zeros((1+delay, n))
            for t in range(T-1):
                U[t] = sigma*np.random.randn(1)
                X[t+1] = A.dot(X[t]) + B.dot(U[t]) + cholV.dot(np.random.randn(m))
                Y[t+1] = C.dot(X[t+1]) + cholW.dot(np.random.randn(n))
                e[1:] = e[:-1]
                td = t-delay+1
                if td >= 0:
                    e[0] = Y[td] - Chat.dot(Xhat[td])
                Le = Lhat.dot(e[0]) if useL else Chat.T.dot(e[0])
                if verbose:
                    eCLe.append(e[0].dot(Chat).dot(Lhat.dot(e[0])))
                Xhat[t+1] = Ahat.dot(Xhat[t]) + Bhat.dot(U[t]) + Lhat.dot(e[0])
                Ahat += etaA*np.outer(Le, Xhat[t-delay])
                Bhat += etaB*np.outer(Le, U[t-delay])
                Chat += etaC*np.outer(e[0], Xhat[t+1-delay])
                Lhat += etaL*np.outer(Le, e[-1])
            mse.append(np.mean((Y-Xhat.dot(Chat.T))**2))
            if verbose:
                Ls.append(Lhat.copy())
        if verbose:
            return Ahat, Bhat, Chat, Lhat, np.array(mse), np.array(Ls), np.array(eCLe)
        else:
            return Ahat, Bhat, Chat, Lhat, np.array(mse)

    def PGafterSysID(self, seed=0, etaK=2.3e-6, momentum=.99, sigma=.2, delay=1,
                     episodes=1000, update_current=True, ABCLhat=None, Khat=None,
                     x0=[-1, 0], episodes2=1000):
        """run policy gradient method using parameters ABCLhat obtained from SysID

        Parameters
        ----------
        seed : int
            random seed
        etaK : float
            learning rate
        momentum : float
            Momentum [0,1)
        sigma : float
            standard deviation of the Gaussian noise of the actor
        delay : int
            amount of measurement delay
        episodes : int
            number of episodes
        update_current : bool
            True:  update current estimate using past prediction error
            False: update past estimate using past prediction error
                   and predict current using model
        ABCLhat : tuple of ndarrays with shapes ((m,m), (m,k), (n,m), (m,n))
            The initial estimates for weight matrices A, B, C, and L.
        Khat : ndarray, shape (k,m)
            initial estimates for control weight matrix K
        x0 : array-like, shape (m,)
            initial state
        episodes2 : int
            number of episodes run after training without noise in actor

        Returns
        -------
        J, J2 : (list, list)
            J  costs during training
            J2 costs after training for running 100 episodes without noise in actor
        """
        A, B, C, Q, R, T = self.A, self.B, self.C, self.Q, self.R, self.T
        cholV, cholW = self.cholV, self.cholW
        n, m = C.shape
        if Khat is None:
            Khat = np.zeros((1, m))
        else:
            Khat = Khat.copy()
        np.random.seed(seed)
        if ABCLhat is None:
            Ahat, Bhat, Chat, Lhat = [.1*np.random.randn(*a_.shape) for a_ in (A, B, C, C.T)]
        else:
            Ahat, Bhat, Chat, Lhat = deepcopy([a.astype(float) for a in ABCLhat])
        X = np.zeros((T, m))
        U = np.zeros((T-1, 1))
        Y = np.zeros((T, n))
        X[0] = x0
        Y[0] = C.dot(X[0])
        Xhat = X.copy()
        Xhat[0] = np.linalg.lstsq(Chat, Y[0], rcond=None)[0]
        J = []
        grad = np.zeros((1, m))
        for _ in range(episodes):
            z = 0
            e = np.zeros((1+delay, n))
            for t in range(T-1):
                if update_current:
                    Xpred = Xhat[t]
                else:
                    td = max(0, t-delay+1)
                    Xpred = Xhat[td]
                    for tt in range(td, t):
                        Xpred = Ahat.dot(Xpred) + Bhat.dot(U[tt])
                xi = sigma*np.random.randn(1)
                U[t] = -Khat.dot(Xpred) + xi
                z += xi * Xpred
                X[t+1] = A.dot(X[t]) + B.dot(U[t]) + cholV.dot(np.random.randn(m))
                Y[t+1] = C.dot(X[t+1]) + cholW.dot(np.random.randn(n))
                if update_current:
                    e[1:] = e[:-1]
                    td = t-delay+1
                    if td >= 0:
                        e[0] = Y[td] - Chat.dot(Xhat[td])
                    Xhat[t+1] = Ahat.dot(Xhat[t]) + Bhat.dot(U[t]) + Lhat.dot(e[0])
                else:
                    Xhat[t+1] = Ahat.dot(Xhat[t]) + Bhat.dot(U[t]) + \
                        Lhat.dot(Y[t]-Chat.dot(Xhat[t]))
                cost = X[t+1].dot(Q).dot(X[t+1]) + U[t].dot(R).dot(U[t])
                grad = momentum*grad + cost * z
                Khat += etaK * grad
            J.append(np.trace(X.T.dot(X).dot(Q)) + np.trace(U.T.dot(U).dot(R)))
        J2 = []
        for _ in range(episodes2):
            e = np.zeros((1 + delay, n))
            for t in range(T-1):
                if update_current:
                    Xpred = Xhat[t]
                else:
                    td = max(0, t-delay+1)
                    Xpred = Xhat[td]
                    for tt in range(td, t):
                        Xpred = Ahat.dot(Xpred) + Bhat.dot(U[tt])
                U[t] = -Khat.dot(Xpred)
                X[t+1] = A.dot(X[t]) + B.dot(U[t]) + cholV.dot(np.random.randn(m))
                Y[t+1] = C.dot(X[t+1]) + cholW.dot(np.random.randn(n))
                if update_current:
                    e[1:] = e[:-1]
                    td = t-delay+1
                    if td >= 0:
                        e[0] = Y[td] - Chat.dot(Xhat[td])
                    Xhat[t+1] = Ahat.dot(Xhat[t]) + Bhat.dot(U[t]) + Lhat.dot(e[0])
                else:
                    Xhat[t+1] = Ahat.dot(Xhat[t]) + Bhat.dot(U[t]) + \
                        Lhat.dot(Y[t]-Chat.dot(Xhat[t]))
            J2.append(np.trace(X.T.dot(X).dot(Q)) + np.trace(U.T.dot(U).dot(R)))
        return J, J2

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
