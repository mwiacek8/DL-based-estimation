import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def trajectory(T, n, k, equation, wiener_process=None):

    """
    Simulates trajectories of a stochastic differential equation (SDE) using the Euler-Maruyama method.

    Parameters:
    - T (float): Total time of the simulation.
    - n (int): Number of time steps.
    - k (int): Number of trajectories to simulate.
    - equation (tuple): Tuple containing the initial value, drift function, and diffusion function.
    - wiener_process (numpy.ndarray or None): Array of pre-generated Wiener process values. If None, it is generated internally.

    Returns:
    - numpy.ndarray: Time values of the trajectories.
    - numpy.ndarray: Simulated values of the trajectories.
    """

    h = float(T / n)
    h_sqrt = np.sqrt(h)
    X_ = []
    T_ = []

    x_0 = equation[0]
    a = equation[1]
    b = equation[2]

    for j in range(k):

        X_temp = x_0
        x = [X_temp]
        t_temp = 0.0
        t = [t_temp]

        for i in range(1, n + 1):

            if wiener_process is not None:
                dW = wiener_process[i - 1]
            else:
                dW = np.random.normal(0, h_sqrt)

            X = X_temp + a(t_temp, X_temp) * h + b(t_temp, X_temp) * dW
            x.append(X)
            t_temp = i * h
            t.append(t_temp)
            X_temp = X

        X_.append(x)
        T_.append(t)

    return np.array(T_).flatten(), np.array(X_).flatten()


def trajectory_intervals(T, n, k, equation, wiener_process=None):

    """
    Simulates trajectories of an SDE with upper and lower confidence intervals.

    Parameters:
    - T (float): Total time of the simulation.
    - n (int): Number of time steps.
    - k (int): Number of trajectories to simulate.
    - equation (tuple): Tuple containing the initial value, drift function, and diffusion function.
    - wiener_process (numpy.ndarray or None): Array of pre-generated Wiener process values. If None, it is generated internally.

    Returns:
    - numpy.ndarray: Time values of the trajectories.
    - numpy.ndarray: Simulated values of the trajectories.
    - numpy.ndarray: Lower confidence interval values.
    - numpy.ndarray: Upper confidence interval values.
    """

    h = float(T / n)
    h_sqrt = np.sqrt(h)
    X_ = []
    X_lower_ = []
    X_upper_ = []
    T_ = []

    x_0 = equation[0]
    a = equation[1]
    b = equation[2]

    for j in range(k):

        X_temp = x_0
        x = [X_temp]
        x_lower = [X_temp]
        x_upper = [X_temp]
        t_temp = 0.0
        t = [t_temp]

        for i in range(1, n + 1):

            if wiener_process is not None:
                dW = wiener_process[i - 1]
            else:
                dW = np.random.normal(0, h_sqrt)

            X = X_temp + a(t_temp, X_temp) * h + b(t_temp, X_temp) * dW
            X_lower = X_temp + a(t_temp, X_temp) * h - 2 * np.abs(b(t_temp, X_temp)) * np.sqrt(h)
            X_upper = X_temp + a(t_temp, X_temp) * h + 2 * np.abs(b(t_temp, X_temp)) * np.sqrt(h)

            x.append(X)
            x_lower.append(X_lower)
            x_upper.append(X_upper)

            t_temp = i * h
            t.append(t_temp)
            X_temp = X

        X_.append(x)
        T_.append(t)

        X_lower_.append(x_lower)
        X_upper_.append(x_upper)

    return np.array(T_).flatten(), np.array(X_).flatten(), np.array(X_lower_).flatten(), np.array(X_upper_).flatten()


def generate_nn_input(t_values, x_values):

    """
    Generates input data for a neural network using time and trajectory values.

    Parameters:
    - t_values (numpy.ndarray): Time values of the trajectories.
    - x_values (numpy.ndarray): Simulated values of the trajectories.

    Returns:
    - tensorflow.Tensor: Input data for the neural network.
    - tensorflow.Tensor: Output data for the neural network.
    """

  x_diff = x_values[1:] - x_values[:-1]
  t_diff = t_values[1:] - t_values[:-1]

  x = np.array([t_values[1:],x_values[1:]]).T
  x = t_values[1:]
  y = np.array([t_values[1:],x_values[1:],x_diff,t_diff]).T

  indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)

  idx = tf.random.shuffle(indices)

  x_data = tf.gather(x, idx)
  y_data = tf.gather(y, idx)

  return x_data, y_data


def check_distribution(T, n, n_traj, trajectories):

    """
    Checks the distribution of real and approximated values at a specific time point.

    Parameters:
    - T (float): Total time of the simulation.
    - n (int): Number of time steps.
    - n_traj (int): Number of trajectories to analyze.
    - trajectories (function): Function to generate trajectories.

    Returns:
    - list: Real values at the specified time point for each trajectory.
    - list: Approximated values at the specified time point for each trajectory.
    """

    t = np.linspace(0,T,n+1)
    Y_1, Y_2 = [], []

    for i in range(n_traj):

        t_values, x_real_values, x_approx_values = trajectories(T,n,1)

        Y_1.append(x_real_values[-1])
        Y_2.append(x_approx_values[-1])

    return Y_1, Y_2



def plot_distribution(Y_1, Y_2):

    """
    Plots the distribution of real and approximated values and prints their mean and standard deviation.

    Parameters:
    - Y_1 (list): Real values.
    - Y_2 (list): Approximated values.
    """

    fig = plt.figure(figsize=(16,5))

    plt.ylabel('Relative Frequency in T')
    plt.title('Distribution of X_real(T) and X_approx(T)')
    plt.hist(Y_1,bins=30,density=1,alpha=0.8, color='darksalmon', label = "true")
    plt.hist(Y_2,bins=150,density=1,alpha=0.8, color='red', label = "approx")
    plt.legend()
    plt.savefig("example_histogram.png")

    print("Mean value for real: ", np.mean(Y_1))
    print("Mean value for approx: ", np.mean(Y_2))

    print("Standard deviation for real: ", np.std(Y_1))
    print("Standard deviation for approx: ", np.std(Y_2))



