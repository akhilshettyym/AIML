import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point,xmat,k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T(-2.0*k**2))
    return weights

def localWeights(point,xmat,ymat,k):
    wei = kernel(point,xmat,k)
    weights = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    return weights

def localweightregression(xmat,ymat,k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] + localWeights(xmat[i],xmat,ymat,k)

def graphPlot(X,ypred):
    sortIndex = X[:,1].argsort(0)
    xsort = X[sortIndex][:,0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(bill,tip,color='green')
    ax.plot(xsort[:,1],ypred[sortIndex],color='red',linewidth=5)
    plt.xlabel('Total Bill')
    plt.ylabel('Tip')
    plt.show()

data = pd.read_csv('/home/student/4.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T,mbill.T))
ypred = localweightregression(X,mtip,5)
graphPlot(X,ypred)