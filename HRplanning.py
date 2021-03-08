import numpy

# Forecasting number of available workers
D = numpy.array([[15,0,0],[0,20,0],[0,0,50]])
M = numpy.array([[5,2,0],[0,10,0],[5,5,45]])
b = numpy.array([[4],[0],[0]])
a = M@numpy.linalg.inv(D)@b
print(a)

# Forecasting number of needed workers
X = numpy.array([[],[],[],[],[],[],[],[],[]])
Y = numpy.array([[],[],[],[],[],[],[],[],[]])
XTX = numpy.matmul(X.T,X)
print(XTX)
inverse = numpy.linalg.inv(XTX)
print(inverse)
XTY = numpy.matmul(X.T,Y)
print(XTY)
betas = numpy.matmul(inverse,XTY)
print(betas)