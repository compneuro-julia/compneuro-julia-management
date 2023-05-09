TotNumSpikes = ns 
M = tspike[tspike[:,1]>dt*tcrit,:]
AverageRate = len(M)/(N*(T-dt*tcrit))
print("\n")
print("Total number of spikes : ", TotNumSpikes)
print("Average firing rate(Hz): ", AverageRate)
step_range = 20000
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
for j in range(5):
    plt.plot(np.arange(step_range)*dt,
             REC_v[:step_range, j]/(50-vreset)+j, color="k")
plt.title('Pre-Learning')
plt.xlabel('Time (s)'); plt.ylabel('Neuron Index') 
plt.subplot(1,2,2)
for j in range(5):
    plt.plot(np.arange(nt-step_range, nt)*dt,
             REC_v[nt-step_range:, j]/(50-vreset)+j,
             color="k")
plt.title('Post Learning'); plt.xlabel('Time (s)')
plt.show()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.arange(nt)*dt, zx, label="Target", color="k")
plt.plot(np.arange(nt)*dt, current, label="Decoded output",
         linestyle="dashed", color="k")
plt.xlim(4.5,5.5); plt.ylim(-1.1,1.4)
plt.title('Pre/peri Learning')
plt.xlabel('Time (s)'); plt.ylabel('current') 
plt.subplot(1,2,2)
plt.title('Post Learning')
plt.plot(np.arange(nt)*dt, zx, label="Target", color="k")
plt.plot(np.arange(nt)*dt, current, label="Decoded output",
         linestyle="dashed", color="k")
plt.xlim(14,15); plt.ylim(-1.1,1.4)
plt.xlabel('Time (s)'); plt.legend(loc='upper right')
plt.show()