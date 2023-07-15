import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils.extmath import svd_flip

class normalPCA(object):
	
	#Inicializuje pocet dat, trid a koncovou dimenzi
	def __init__(self, ns, nf, fr):
		self.n_samples = ns
		self.n_features = nf
		self.features_reduced = fr
		self.X = np.zeros((ns, nf))
		self.e_vec1 = np.zeros((nf, nf))
		self.e_val = np.zeros((nf, 1))
		self.e_vec2 = np.zeros((nf, nf))
		
	def getEig(self, CM):
		self.e_vec1, self.e_val, self.e_vec2 = np.linalg.svd(CM, full_matrices=True)
		return [self.e_vec1, self.e_val, self.e_vec2]
		
	#Transformuje data
	def eval(self, X):
		X_c = np.zeros((self.n_samples, self.n_features))
		for i in range (0,self.n_features):
			X_c[:,i] = X[:,i] - np.mean(X[:,i])
			cov_matrix = np.cov(X_c.transpose())
			
		U, S, V = self.getEig(cov_matrix)
		#U, V = svd_flip(U, V)
		u_red = U[:,0:self.features_reduced]
		return np.dot(X_c,u_red)
		
	#Vytvori dva nahodne shluky
	def randomSample(self):
		X1 = np.random.randn(int(np.floor(self.n_samples/2)), self.n_features)
		X2 = np.random.randn(int(np.ceil(self.n_samples/2)), self.n_features)
		X2[:,1] = X2[:,1] + 5
		X2[:,0] = X2[:,0] + 5
		self.X = np.row_stack((X1,X2))
		return self.X
		
	#Vizualizace
	def plot_all(self, X, z_ker):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:,0], X[:,1], X[:,2])
		ax.scatter(z_ker[:,0], z_ker[:,1])
		plt.show()
	
	def plot_init(self, X):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:,0], X[:,1], X[:,2])
		plt.show()
	
	def plot_new(self, z_ker):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(z_ker[:,0], z_ker[:,1])
		plt.show()

npca = normalPCA(100, 3, 2)
X = npca.randomSample()
z = npca.eval(X)

npca.plot_init(X)
npca.plot_new(z)
npca.plot_all(X, z)


