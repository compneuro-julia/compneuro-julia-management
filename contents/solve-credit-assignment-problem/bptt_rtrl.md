RNN

状態
$$
\mathbf{h}(t+1)=\left(1-\frac{1}{\tau}\right)\mathbf{h}(t)+\frac{1}{\tau}f(\mathbf{W}\mathbf{h}(t)+\mathbf{W}_{in}\mathbf{x}(t+1)+\mathbf{b})
$$

出力は
$$
\mathbf{y}(t)=\mathbf{W}\mathbf{h}(t)
$$

BPTT (Backpropagation through time)
backpropagation through time (BPTT) ( Rumelhart et al., 1985) in order to compare it with the learning rules presented above. The derivation here follows Lecun (1988).

RTRL (Real-time recurrent learning)
Williams RJ, Zipser D. 1989. A learning algorithm for continually running fully recurrent neural networks. Neural Computation 1:270–280. DOI: https://doi.org/10.1162/neco.1989.1.2.270



RFRO (Random feedback local online learning)