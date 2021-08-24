using LinearAlgebra, Random, ToeplitzMatrices, PyPlot

vec(X) = vcat(X...)

# Equality Constrained Quadratic Programming
function solveEqualityConstrainedQuadProg(P, q, A, b)
    """
    minimize   : 1/2 * x'*P*x + q'*x
    subject to : A*x = b
    """
    K = [P A'; A zeros(size(A)[1], size(A)[1])] # KKT matrix
    sol = K \ [-q; b]
    return sol[1:size(A)[2]]
end

t1 = 224*1e-3 # time const of eye dynamics (s)
t2 = 13*1e-3  # another time const of eye dynamics (s)
tm = 10*1e-3
dt = 1e-3     # simulation time step (s)
tf = 50*1e-3  # movement duration (s)
tp = 30*1e-3  # post-movement duration (s)
ntf = round(Int, tf/dt)
ntp = round(Int, tp/dt)
nt = ntf + ntp # total time steps
trange = (1:nt) * dt * 1e3 # ms

x0 = zeros(3)       # initial state (pos=0, vel=0, acc=0)
xf = [10, 0, 0] # final state (pos=10, vel=0, acc=0)
α1 = -1/(t1*t2*tm)
α2 = -1/(t1*t2)-1/(t1*tm)-1/(t2*tm)
α3 = -1/t1-1/t2-1/tm
β = 1/tm
Ac = [0 1 0; 0 0 1; α1 α2 α3];
Bc = [0, 0, β]
A = LinearAlgebra.I + Ac * dt
B = Bc*dt
#A = exp(Ac*dt);
#B = Ac^-1 * (eye(3) - A) *Bc; 

# calculation of V
diagV = zeros(nt);
for i=0:nt-1
    if i < ntf
        diagV[i+1] = sum([(A^(k-i-1) * B * B' * A'^(k-i-1))[1,1] for k=ntf:nt-1])
    else
        diagV[i+1] = diagV[i] + (A^(nt-i-2) * B * B' * A'^(nt-i-2))[1,1]
    end
end
diagV /= maximum(diagV) # for numerical stability
V = Diagonal(diagV); 

# calculation of C
C = zeros(3(ntp+1), nt);
for p=1:ntp+1
    for q=1:nt
        if ntf-1+(p-1)-(q-1) >= 0
            idx = 3(p-1)+1:3p
            C[idx, q] = A^(ntf-1-(q-1)+(p-1)) * B # if ntf-1-(q-1)+(p-1) == 0; A^(ntf-1-(q-1)+(p-1))*B equal to B
        end
    end
end

# calculation of d
d = vec([xf - A^(ntf+i) * x0 for i=0:ntp]);

# solution by quadratic programming
u = solveEqualityConstrainedQuadProg(V, zeros(nt), C, d);

# forward solution
x = zeros(3, nt);
x[:,1] = x0;
for k=1:nt-1
    x[:,k+1] = A*x[:, k] + B*u[k]
end

figure(figsize=(10, 3))
subplot(1,3,1)
plot(trange, x[1, :])
ylabel("Eye position (deg)"); xlabel("Time (ms)"); grid()
subplot(1,3,2)
plot(trange, x[2, :])
ylabel("Eye velocity (deg/s)"); xlabel("Time (ms)"); grid()
subplot(1,3,3)
plot(trange, u)
ylabel("Control signal"); xlabel("Time (ms)"); grid()
tight_layout()
