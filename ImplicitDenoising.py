import numpy as np
import tensorflow as tf
from numpy.fft import fft, ifft, fftfreq

global_tf_datatype = tf.float32
global_np_datatype = np.float32

rk_methods = ['Backward_Euler', 'Forward_Euler', 'Midpoint', 'Gauss1', 'Trapezoid', 'Gauss2', 'Gauss3']

def RK_loss(X,f,h,A,b,m):
    """
    Input:
        x = [x_j] + [f^{-1}(k_i)]_{i=1}^s + [x_j+1]
        f = vector field
        h = timestep
    Output:

    """
    
    s = len(b)

    X0_indices = tf.constant([(s+1)*j for j in range(m-1)])
    X_middle_indices = [tf.constant([(s+1)*j+1+i for j in range(m-1)]) for i in range(s)]
    X1_indices = tf.constant([(s+1)*j for j in range(1,m)])

    X0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), X0_indices))
    X_middle = [tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), indices)) for indices in X_middle_indices]
    X1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), X1_indices))

    # Prediction accuracy
    loss = tf.nn.l2_loss(X1 - X0 - h*tf.add_n([b[j]*f(X_middle[j]) for j in range(s)]))

    # Midpoint accuracy
    for i in range(s):
        loss = loss + tf.nn.l2_loss(X_middle[i] - X0 - h*tf.add_n([A[i][j]*f(X_middle[j]) for j in range(s)]))

    return loss

def RK_residuals(X,f,h,A,b,bs,m):
    """
    Input:
        x = [x_j] + [f^{-1}(k_i)]_{i=1}^s + [x_j+1]
        f = vector field
        h = timestep
    Output:

    """
    
    s = len(b)

    X0_indices = tf.constant([(s+1)*j for j in range(m-1)])
    X_middle_indices = [tf.constant([(s+1)*j+1+i for j in range(m-1)]) for i in range(s)]
    X1_indices = tf.constant([(s+1)*j for j in range(1,m)])

    X0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), X0_indices))
    X_middle = [tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), indices)) for indices in X_middle_indices]
    X1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), X1_indices))

    # High and low order predictions
    X1_high = X0 + h*tf.add_n([b[j]*f(X_middle[j]) for j in range(s)])
    X1_low = X0 + h*tf.add_n([bs[j]*f(X_middle[j]) for j in range(s)])

    # Resilduals
    Residual_high = X1 - X1_high
    Residual_low = X1 - X1_low
    High_low_diff = X1_high - X1_low

    # Return residual
    return [Residual_high, Residual_low, High_low_diff]

def RK_tables(method = 'Midpoint'):
    """
    Returns Butcher table for a few different types of implicit Runge-Kutta methods
    
    Some of these have a lower order approximation for adaptive step size which for now I'm, not using.
    Perhaps could be used for some sort of UQ?
    """

    if method == 'Backward_Euler':
        
        A = [[1]]
        b = [1]
        bs = None
        c = [1]

    if method == 'Forward_Euler':
        
        A = [[0]]
        b = [1]
        bs = None
        c = [0]

    if method == 'Midpoint' or method == 'Gauss1':
        
        A = [[1/2]]
        b = [1]
        bs = None
        c = [1/2]

    elif method == 'Trapezoid':
        
        A = [[0,0],[1/2,1/2]]
        b = [1/2,1/2]
        bs = [1,0]
        c = [0,1]

    elif method == 'Gauss2':

        A = [[1/4, 1/4-np.sqrt(3)/6],\
             [1/4+np.sqrt(3)/6, 1/4]]
        b = [1/2,1/2]
        bs = [1/2+np.sqrt(3)/2, 1/2-np.sqrt(3)/2]
        c = [1/2-np.sqrt(3)/6, 1/2+np.sqrt(3)/6]

    elif method == 'Gauss3':

        A = [[5/36, 2/9-np.sqrt(15)/15, 5/36-np.sqrt(15)/30],\
             [5/36+np.sqrt(15)/24, 2/9, 5/36-np.sqrt(15)/24],\
             [5/36+np.sqrt(15)/30, 2/9+np.sqrt(15)/15, 5/36]]
        b = [5/18,4/9,5/18]
        bs = [-5/6, 8/3, -5/6]
        c = [1/2-np.sqrt(15)/10, 1/2, 1/2+np.sqrt(15)/10]

    else:

        # Just using trapezoid for now
        A = [[0,0],[1/2,1/2]]
        b = [1/2,1/2]
        bs = [1,0]
        c = [0,1]

    return A,b,bs,c

def approximate_noise(Y, T, lam = 1e-3):

    n,m = Y.shape
    dt = T[1:] - T[:-1]

    D = np.zeros((m-2,m))

    for i in range(m-2):
	    D[i,i] = 2 / (dt[i]*(dt[i]+dt[i+1]))
	    D[i,i+1] = -2/(dt[i]*(dt[i]+dt[i+1])) - 2/(dt[i+1]*(dt[i]+dt[i+1]))
	    D[i,i+2] = 2 / (dt[i+1]*(dt[i]+dt[i+1]))
	    
    X_smooth = np.vstack([np.linalg.solve(np.eye(m) + lam*D.T.dot(D), Y[j,:].reshape(m,1)).reshape(1,m) for j in range(n)])

    N_hat = Y-X_smooth

    return N_hat, X_smooth

def expand_X(X, c):
    """
    Linear interpolation to get approximate midpopints.
    """

    n,m = X.shape
    s = len(c)
    
    Xs = np.zeros((n, (s+1)*(m-1)+1))
    Xs[:,-1] = X[:,-1]

    for i in range(m-1):

        Xs[:,(s+1)*i] = X[:,i]

        for j in range(s):

            Xs[:,(s+1)*i+j+1] = (c[j]*X[:,i+1]) + (1-c[j])*X[:,i]

    return Xs

def derivative_regularizer(X, H):

    # As is, not really set up to work with variable timesteps
    
    n,m = tf.shape(X).eval()
        
    DX = tf.divide(tf.slice(X, [0,0], [n,m-5]) - 4*tf.slice(X, [0,1], [n,m-5]) + \
              6*tf.slice(X, [0,2], [n,m-5]) - 4*tf.slice(X, [0,3], [n,m-5]) + \
              tf.slice(X, [0,4], [n,m-5]), tf.pow(tf.slice(H, [0,2], [1,m-5]), 4))

    return tf.nn.l2_loss(DX)

def create_computational_graph(Y, T, f, method = 'Midpoint', gamma = 1e-8, noise_penalty="L2", reg_derivative = 0):

    n,m = Y.shape
    H = tf.constant(T[1:]-T[:-1], dtype=global_tf_datatype, shape=[1,m-1], name = "H") # timestep lengths
    _, Ys = approximate_noise(Y, T, lam = 1e-5)

    if method in rk_methods:

        # Use Runge-Kutta timestepper

        A,b,bs,c = RK_tables(method)
        
        # Expanded state variable
        X_extended = tf.get_variable("X_extended", initializer = expand_X(Ys,c).astype(global_np_datatype))

        # Indices corresponding to measurements
        measurement_indices = tf.constant([(len(c)+1)*j for j in range(m)])
        X_hat = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X_extended), measurement_indices))

        # Cost from fit to implicit timestepper
        timestepper_cost = RK_loss(X_extended, f, H, A, b, m)
        residual_cost = RK_residuals(X_extended,f,H,A,b,bs,m)[0]

    else: raise ValueError('Method not recognized.')

    # Cost from to magnitude of noise
    if noise_penalty == "L2": noise_cost = tf.nn.l2_loss(tf.constant(Y.astype(global_np_datatype)) - X_hat)
    elif noise_penalty == "L1": noise_cost = tf.reduce_sum(tf.abs(tf.constant(Y.astype(global_np_datatype)) - X_hat))

    cost = timestepper_cost + gamma*noise_cost

    # Cost from fourth order derivative
    if reg_derivative != 0: cost = cost + reg_derivative*derivative_regularizer(X_hat, H)

    # L-BFGS-B optimizer via scipy
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, options={'maxiter': 50000, 
                                                                      'maxfun': 50000,
                                                                      'ftol': 1e-15, 
                                                                      'gtol' : 1e-11,
                                                                      'eps' : 1e-15,
                                                                      'maxls' : 100})

    return optimizer, X_hat, [cost, timestepper_cost, gamma*noise_cost, residual_cost]

###########################################################################################################
###########################################################################################################
###########################################################################################################

def expand_X_2(X, c, f):
    """
    Linear interpolation to get approximate midpopints.
    """

    n,m = X.shape
    s = len(c)
    
    Xs = np.zeros((n, (s+1)*(m-1)+1))
    Xs[:,-1] = X[:,-1]

    for i in range(m-1):

        Xs[:,(s+1)*i] = X[:,i]

        for j in range(s):

            Xs[:,(s+1)*i+j+1] = f((c[j]*X[:,i+1]) + (1-c[j])*X[:,i])

    return Xs

def RK_loss_2(X,f,h,A,b,m):
    """
    Same as above but for optimization over k

    Input:
        x = [x_j] + [f^{-1}(k_i)]_{i=1}^s + [x_j+1]
        f = vector field
        h = timestep
    Output:

    """
    
    s = len(b)

    X0_indices = tf.constant([(s+1)*j for j in range(m-1)])
    K_indices = [tf.constant([(s+1)*j+1+i for j in range(m-1)]) for i in range(s)]
    X1_indices = tf.constant([(s+1)*j for j in range(1,m)])

    X0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), X0_indices))
    K = [tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), indices)) for indices in K_indices]
    X1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), X1_indices))

    # Prediction accuracy
    loss = tf.nn.l2_loss(X1 - X0 - h*tf.add_n([b[j]*K[j] for j in range(s)]))

    # Midpoint accuracy
    for i in range(s):
        loss = loss + tf.nn.l2_loss(K[i] - f(X0 + h*tf.add_n([A[i][j]*K[j] for j in range(s)] )))

    return loss

def RK_residuals_2(X,f,h,A,b,bs,m):
    """
    Input:
        x = [x_j] + [f^{-1}(k_i)]_{i=1}^s + [x_j+1]
        f = vector field
        h = timestep
    Output:
    """
    
    s = len(b)

    X0_indices = tf.constant([(s+1)*j for j in range(m-1)])
    K_indices = [tf.constant([(s+1)*j+1+i for j in range(m-1)]) for i in range(s)]
    X1_indices = tf.constant([(s+1)*j for j in range(1,m)])

    X0 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), X0_indices))
    K = [tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), indices)) for indices in K_indices]
    X1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X), X1_indices))

    # High and low order predictions
    X1_high = X0 + h*tf.add_n([b[j]*K[j] for j in range(s)])
    X1_low = X0 + h*tf.add_n([bs[j]*K[j] for j in range(s)])

    # Resilduals
    Residual_high = X1 - X1_high
    Residual_low = X1 - X1_low
    High_low_diff = X1_high - X1_low

    # Return residual
    return [Residual_high, Residual_low, High_low_diff]

def create_computational_graph_2(Y, T, f, f_np, method = 'Midpoint', gamma = 1e-8, noise_penalty="L2", reg_derivative = 0):
    """
    similar to above but optimizes over k_j^i instead of x_j^i

    """

    n,m = Y.shape
    H = tf.constant(T[1:]-T[:-1], dtype=global_tf_datatype, shape=[1,m-1], name = "H") # timestep lengths
    _, Ys = approximate_noise(Y, T, lam = 1e-5)

    if method in rk_methods:

        # Use Runge-Kutta timestepper

        A,b,bs,c = RK_tables(method)
        
        # Expanded state variable
        X_extended = tf.get_variable("X_extended", initializer = expand_X_2(Ys,c,f_np).astype(global_np_datatype))

        # Indices corresponding to measurements
        measurement_indices = tf.constant([(len(c)+1)*j for j in range(m)])
        X_hat = tf.transpose(tf.nn.embedding_lookup(tf.transpose(X_extended), measurement_indices))

        # Cost from fit to implicit timestepper
        timestepper_cost = RK_loss_2(X_extended, f, H, A, b, m)
        residual_cost = RK_residuals_2(X_extended,f,H,A,b,bs,m)[0]

    else: raise ValueError('Method not recognized.')

    # Cost from to magnitude of noise
    if noise_penalty == "L2": noise_cost = tf.nn.l2_loss(tf.constant(Y.astype(global_np_datatype)) - X_hat)
    elif noise_penalty == "L1": noise_cost = tf.reduce_sum(tf.abs(tf.constant(Y.astype(global_np_datatype)) - X_hat))

    cost = timestepper_cost + gamma*noise_cost

    # Cost from fourth order derivative
    if reg_derivative != 0: cost = cost + reg_derivative*derivative_regularizer(X_hat, H)

    # L-BFGS-B optimizer via scipy
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, options={'maxiter': 50000, 
                                                                      'maxfun': 50000,
                                                                      'ftol': 1e-15, 
                                                                      'gtol' : 1e-11,
                                                                      'eps' : 1e-15,
                                                                      'maxls' : 100})

    return optimizer, X_hat, [cost, timestepper_cost, gamma*noise_cost, residual_cost]

