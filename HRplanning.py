import numpy

# Forecasting number of available workers
D = numpy.array([[15,0,0],[0,20,0],[0,0,50]])
M = numpy.array([[5,2,0],[0,10,0],[5,5,45]])
b = numpy.array([[4],[0],[0]])
a = M@numpy.linalg.inv(D)@b
print(a)

# Forecasting number of needed workers
X = numpy.array([[15,1,0,0],[17,1,0,0],[14,1,0,0],[6,0,1,0],[6,0,1,0],[5,0,1,0],[23,0,0,1],[27,0,0,1],[25,0,0,1]])
Y = numpy.array([[4],[6],[5],[13],[12],[13],[7],[7],[9]])
XTX = numpy.matmul(X.T,X)
print(XTX)
inverse = numpy.linalg.inv(XTX)
print(inverse)
XTY = numpy.matmul(X.T,Y)
print(XTY)
betas = numpy.matmul(inverse,XTY)
print(betas)
y = numpy.array([[10,1,0,0],[4,0,1,0],[20,0,0,1]])@betas
print(y)