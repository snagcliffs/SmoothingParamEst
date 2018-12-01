import numpy as np
import numpy.linalg as linalg
from numpy.linalg import svd as svd

def Matrix_sqrt(X):
	U, Sigma, V = svd(X)
	return V.T.dot(np.diag(np.sqrt(Sigma))).dot(U.T)

def Matrix_inv(X):
	U, Sigma, V = svd(X, full_matrices=False)
	return V.T.dot(np.diag(Sigma**-1)).dot(U.T)

def EnKF(Y, f, h, x0, P0, R, Q, N):
	"""
	Square-root Ensemble Kalman Filter
	"""

	nx = x0.shape[0]
	ny, m = Y.shape

	# Forecast and analysis ensembles
	Ea = np.zeros((nx,N,m))
	Ef = np.zeros((nx,N,m))

	# Square roots of initial data, process noise, and measurement noise covariance
	sqrt_P0 = Matrix_sqrt(P0)
	sqrt_Q = Matrix_sqrt(Q)
	sqrt_R = Matrix_sqrt(R)
        
	# Timestepping loop
	for j in range(m):

		# Forecast previous analysis
		if j == 0: Ef[:,:,j] = np.tile(x0.reshape(nx,1), (1,N)) + sqrt_P0.dot(np.random.randn(nx,N))
		else: Ef[:,:,j] = f(Ea[:,:,j-1]) + sqrt_Q.dot(np.random.randn(nx,N))
		Ey = h(Ef[:,:,j])

		# Compute anomolies and gain
		Af = Ef[:,:,j] - np.tile(Ef[:,:,j].mean(1).reshape(nx,1), (1,N))
		Ay = Ey - np.tile(Ey.mean(1).reshape(ny,1), (1,N))
		K = Af.dot(Ay.T).dot(Matrix_inv(Ay.dot(Ay.T)+(N-1)*R))
        
		# Assimilate
		Ea[:,:,j] = Ef[:,:,j] + K.dot(np.tile(Y[:,j], (N,1)).T - Ey - sqrt_R.dot(np.random.randn(ny,N)))

	return Ea, Ef

def EnRTS(Y, f, h, x0, P0, R, Q, N):
    """
    For reference see:
    On the ensemble Rauch-Tung-Striebel smootherand its equivalence to the ensemble Kalman smoother
    Patrick Nima Raanesab
    Quarterly Journal of the Royal Meteorological Society 
    """
    
    Ea, Ef = EnKF(Y, f, h, x0, P0, R, Q, N)
    
    nx = x0.shape[0]
    ny, m = Y.shape
    
    # Smoothed ensemble
    Es = np.zeros((nx,N,m))
    Es[:,:,m-1] = Ea[:,:,m-1]
        
    for j in np.arange(m-2,-1,-1):

        Aa = Ea[:,:,j] - np.tile(Ea[:,:,j].mean(1).reshape(nx,1), (1,N))
        Af = Ef[:,:,j+1] - np.tile(Ef[:,:,j+1].mean(1).reshape(nx,1), (1,N))
        J = Aa.dot(Matrix_inv(Af))
        Es[:,:,j] = Ea[:,:,j] + J.dot(Es[:,:,j+1]-Ef[:,:,j+1])
        
    return Ef, Ea, Es