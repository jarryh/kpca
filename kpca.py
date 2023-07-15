import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA
from scipy.sparse.linalg import eigsh


##KERNEL PCA
class kPCA(object):
	def __init__(self, ns, nf, fr): 
		self.n_samples = ns
		self.n_features = nf
		self.features_reduced = fr
		
		
	def kernel(self, X, ker):
		K1 = np.zeros((self.n_samples, self.n_samples))
		K2 = np.zeros((self.n_samples, self.n_samples))
		
		if ker == "linear":
			for i in range(0, self.n_samples):
				for j in range(0, self.n_samples):
					K1[i, j] = np.dot(X[i], X[j])
			return K1
		
		if ker == "rbf":
			
			gamma=1/self.n_features
			
			for i in range(0, self.n_samples):
				for j in range(0, self.n_samples):
					K2[i, j] = np.exp(-gamma*self.euk_dist(X[i], X[j])**2)
			return K2
		
		if ker == "sigmoid":
			gamma=1/self.n_features
			coef0 = 1
			for i in range(0, self.n_samples):
				for j in range(0, self.n_samples):
					K1[i, j] = np.tanh(gamma*np.dot(X[i], X[j]) + coef0)
			return K1
		
		return -1
	##

	##Eukleidovska vzdalenost
	def euk_dist(self, x,y):
		return np.sqrt(np.dot(x.transpose(), x) - 2 * np.dot(x, y) + np.dot(y, y.transpose()))
	##

	##Centrovani matice
	def centering(self, K):
		I = 1/self.n_samples*np.ones(self.n_samples)
		return K - np.dot(I,K)-np.dot(K,I)+np.dot(I,np.dot(K,I))
	##

	##Nahodny vzorek dat
	def randomSample(self):
		X1 = np.random.randn(int(np.floor(self.n_samples/2)), self.n_features)
		X2 = np.random.randn(int(np.ceil(self.n_samples/2)), self.n_features)
		X2[:,1] = 2*np.cos(X2[:,1] + 8*np.pi) + 5
		X2[:,0] = 3*np.sin(X2[:,0] - 8*np.pi) + 5
		return np.row_stack((X1,X2))
	##
		
	##Vypocet vlastnich vektoru, vlastnich cisel a serazeni podle velikosti
	def transform_data(self, X, ker_name):
		K = self.kernel(X, ker_name)
		K_c = self.centering(K)
		
		eigvals, eigvecs = np.linalg.eigh(K_c)
		idx = eigvals.argsort()[::-1]   
		eigvals = eigvals[idx]
		eigvecs = eigvecs[:,idx]

		##Zvolim d prvnich vlastnich vektory a podelim odmocninou z vlastniho cisla
		u_red =  eigvecs[:,0:self.features_reduced]/np.sqrt(eigvals[0:self.features_reduced])
		z_ker = np.zeros((self.n_samples, self.features_reduced))
		for i in range (0, self.features_reduced):
			z_ker[:, i] = np.dot(u_red[:,i].transpose(),K_c)
		return z_ker
	
	##Vizualizace
	def plot_init_data(self, X):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:,0], X[:,1], X[:,2])
		plt.show()
		
	def plot_new_data(self, z_ker):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(z_ker[:,0], z_ker[:,1])
		plt.show()

	def plot_all(self, X, z_ker):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(max(z_ker[:,0])*X[:,0]/max(X[:,0]), z_ker[:,1]*X[:,1]/max(X[:,1]), X[:,2])
		ax.scatter(z_ker[:,0], z_ker[:,1])
		plt.show()
	##
	
  

mk = kPCA(200, 3, 2)
X = mk.randomSample()
z_ker = mk.transform_data(X, "rbf")

mk.plot_init_data(X)
mk.plot_new_data(z_ker)
mk.plot_all(X, z_ker)

