import numpy as np
import matplotlib.pyplot as plt

# Model parameters
A = 3.25  # mV
B = 22  # mV
a = 100  # s^-1
b = 50  # s^-1
C1 = 135
C2 = 0.8 * C1
C3 = 0.25 * C1
C4 = 0.25 * C1
v0 = 6  # mV
e0 = 2.5  # s^-1
r = 0.56  # mV^-1

# Sigmoid function
def Sigm(v):
    return 2 * e0 / (1 + np.exp(r * (v0 - v)))

# Differential equations
def scm(t, y):
    dydt = np.zeros(6)
    
    dydt[0] = y[3]
    dydt[3] = A * a * Sigm(y[1] - y[2]) - 2 * a * y[3] - a**2 * y[0]
    dydt[1] = y[4]
    dydt[4] = A * a * ( C2 * Sigm(C1 * y[0])) - 2 * a * y[4] - a**2 * y[1]
    dydt[2] = y[5]
    dydt[5] = B * b * (C4 * Sigm(C3 * y[0])) - 2 * b * y[5] - b**2 * y[2]

    return dydt


# euler Solve 
def ode(func, t_ini, t_fin, t_step, y_ini):
    n_steps = int((t_fin - t_ini) / t_step)
    ys = np.zeros((n_steps, len(y_ini)))
    ys[0, :] = y_ini
    t = np.arange(t_ini, t_fin, t_step)
    for i in range(1, n_steps):
        dy = func(t[i-1], ys[i-1, :])
        ys[i, :] = ys[i-1, :] + t_step * dy
    return t, ys


# Initial conditions
y_ini = np.zeros(6)

# Time for the simulation
t_ini = 0 
t_fin = 5 # Simulate for 10 seconds
t_step = 0.01 # Evaluate solution at 1000 points

t, y =ode(scm, t_ini, t_fin, t_step, y_ini)



# Plot the solution for y1, y2, and y3 which represent the activities
plt.figure(figsize=(14, 8))
#plt.plot(t, y[:,0], label='y0 (Pyramidal Cells Output)')
plt.plot(t, y[:,1]-y[:,2], label='y1-y2 (Excitatory Interneurons Output)')
##plt.plot(t, y[:,2], label='y2 (Inhibitory Interneurons Output)')
plt.xlabel('Time (s)')
plt.ylabel('Activity(mv)')
plt.title('Activities of the Cortical Column Model Over Time')
plt.legend()
plt.show()

