from cvxopt.solvers import qp
from cvxopt.base import matrix, spdiag
import numpy, pylab, random, math

kerneltype="linear"

# Define kernels, the kernel is a measure of similarity between points
def kernel(type,xi,xj):
	_SIGMA=2
	_DIMENSION=3
	if(type=="linear"):
		k=(matrix(xi).trans() * matrix(xj) + 1)
	if(type=="polynomial"):
		k=math.pow((matrix(xi).trans() * matrix(xj) + 1)[0], _DIMENSION)
	if(type=="rbf"):
		k=math.exp(-1 * math.pow(math.sqrt(math.pow((matrix(xi) - matrix(xj))[0], 2) + math.pow((matrix(xi) - matrix(xj))[1], 2)), 2) / 2 * math.pow(_SIGMA, 2))
	return k

# Build covariance matrix
# P = t_i * t_j * K(x, y)
def buildP(data):
	N = len(data)
	P = list()

	for i in range(N):
		r = list()
		xi = data[i][0:2]
		ti = data[i][2]
		
		for j in range(N):			
			xj = data[j][0:2]
			tj = data[j][2]

			r.append(ti * tj * kernel(kerneltype,xi, xj))

		P.append(r)

	return matrix(P)

def getSupportVectors(alpha):
	threshold = math.pow(10, -5)
	
	zeros = list()
	for i in range(len(alpha)):
		if alpha[i] < threshold:
			zeros.append(i)

	if len(zeros) == 0:
		return list()

	support_vectors = list()
	for i in range(len(alpha)):
		if i not in zeros:
			support_vectors.append(i)

	return support_vectors

# The indicator function assesses how similar the point is to each support vector and weighs them with alpha value, and makes class prediction based on this and target class of sv
def indicator(x, y, data, alpha, sv):
	ind = 0
	for i in sv:
		ind += alpha[i] * data[i][2] * kernel(kerneltype,matrix([x, y]), data[i][0:2])
	return ind

# Generate data
# classA = 1
# classB = -1
classA = [(random.normalvariate(-1.5, 1), 
	random.normalvariate(0.5, 1), 
	1.0)
	for i in range(5)] + \
	[(random.normalvariate(1.5, 1),
	random.normalvariate(0.5, 1),
	1.0)
	for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5),
	random.normalvariate(-0.5, 0.5),
	-1.0)
	for i in range(10)]

data = classA + classB
random.shuffle(data)

# Build P,q,h and G
P=buildP(data)
N = 20
q = matrix(-1.0, (N, 1))			# Vector of size N containing only -1
G = matrix(spdiag(N * [-1.0]))		# Diagonal matrix with -1 on the diagonal, 0 otherwise
h = matrix(0.0, (N, 1))				# Vector of size N containing only 0


# QP Optimizer
r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])

# Get support vectors
sv=getSupportVectors(alpha)

# Plot data
pylab.hold(True)
pylab.plot([p[0] for p in classA],
[p[1] for p in classA],
'bo')
pylab.plot([p[0] for p in classB],
[p[1] for p in classB],
'ro')

# Plot boundaries
xrange=numpy.arange(-4, 4, 0.05)
yrange=numpy.arange(-4, 4, 0.05)
grid = matrix([[indicator(x, y, data, alpha, sv) 
	for y in yrange]
	for x in xrange])
pylab.contour(xrange, yrange, grid,
	(-1.0, 0.0, 1.0),
	colors=('red', 'black', 'blue'),
	linewidths =(1, 3, 1))

pylab.show()