import numpy as np

def LIF(tau=10, t0=20, t1=40, t2=60, w=0.1, threshold=1.0, reset=0.0):
    # Spike times, keep sorted because it's more efficient to pop the last value off the list
    times = [t0, t1, t2]
    times.sort(reverse=True)
    
    # set some default parameters
    duration = 100  # total time in ms
    dt = 0.1  # timestep in ms
    alpha = np.exp(-dt/tau)  # decay factor per time step
    
    V_rec = []  # list to record membrane potentials
    T_rec = []  # list to record time points
    V = 0.0  # initial membrane potential
    
    T = np.arange(0, duration, dt)  # array of time points
    spikes = []  # list to store spike times

    # run the simulation
    for t in T:
        T_rec.append(t)  # record time
        V_rec.append(V)  # record voltage before update

        # Integrate equations
        V *= alpha
        
        # Check if there is an input spike
        if times and t > times[-1]:
            V += w
            times.pop()  # remove spike time after processing
        
        # If the potential exceeds the threshold, reset
        if V > threshold:
            V = reset
            spikes.append(t)

    # Return the lists of time and voltage values
    return T_rec, V_rec

# Example usage
times, voltages = LIF()

# Now, times contains the x coordinates (time points) and voltages contains the y coordinates (membrane potentials)
print(times[:10])  # First 10 time points
print(voltages[:10])  # First 10 voltage points

