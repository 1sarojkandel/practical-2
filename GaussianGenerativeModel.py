from scipy.stats import multivariate_normal
import numpy as np
import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib.colors as c
from numpy import linalg
from numpy import matrix

from collections import defaultdict



# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance
        
        #self.mean_k=np.zeros((15,177))
        self.Num_k=np.zeros((1,15))
        self.mean_k = []
        
        self.covarianceMatrix=np.zeros((15,15))
        self.covarianceMatrixSeparate=[]


    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        T=[]
        nClasses = max(Y) + 1
        numX = len(X)
                
        for i in range(len(self.Y)):
            T.append([0]*nClasses)
            T[i][self.Y[i]]=1
            
        self.mean_k=[[0]*len(X[0])]*nClasses
        self.covarianceMatrix=[[0]*len(X[0])]*len(X[0])
        self.Num_k=np.sum(T,axis=0)
        
        '''for i in range(nClasses):
            for j in range(numX):
                a = self.mean_k[i]
                b = (1.0/self.Num_k[i])*T[j][i]*self.X[j]
                
                self.mean_k[i]=np.add(a,b)
                
                add_a = self.covarianceMatrix
                add_b = T[j][i]*matrix(np.add(self.X[j],-self.mean_k[i])).T \
                            *matrix(np.add(self.X[j],-self.mean_k[i]))
                self.covarianceMatrix=np.add(add_a,add_b)
                
        self.covarianceMatrix=self.covarianceMatrix/float(numX)'''
        



        # Filter by class, and calculate mean and covariance of each.
        for c in range(nClasses):
            rows_in_class = X[Y == c]
            self.mean_k.append(np.mean(rows_in_class, axis=0))
            if self.isSharedCovariance:
                Cov_i = np.cov(rows_in_class.T)
                self.covarianceMatrix += Cov_i*rows_in_class.shape[0]
            else:
                self.covarianceMatrixSeparate.append(np.cov(rows_in_class.T))
        
                
    # TODO: Implement this method!

    def __separateClasses(self, X, Y):
        result = defaultdict(list)
        
        for i in range(len(Y)):
            xVals = X[i]
            if Y[i] not in result:
                result[Y[i]] = []
                
            result[Y[i]].append(xVals)
            
        return result        
    
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        N=len(X_to_predict)
        result= np.zeros((len(X_to_predict),15))
        Y = []
        
               
        for n in range(N):
            for k in range(15):
                if self.isSharedCovariance == True:
                    result[n][k] = -0.5*np.log(np.linalg.det(self.covarianceMatrix)) - 0.5*matrix((X_to_predict[n]
                            -self.mean_k[k]))*matrix(self.covarianceMatrix).I*matrix((X_to_predict[n]-self.mean_k[k])).T+np.log(float(self.Num_k[k])/N)
                else:
                    ls=self.__separateClasses(self.X, self.Y)[k]
                    #ls = self.X[self.Y==k]
                    saroj = zip(*ls)
                    self.covarianceMatrixSeparate.append(np.cov(saroj))
                    result[n][k] = -0.5*np.log(np.linalg.det(self.covarianceMatrixSeparate[k])) - 0.5*matrix((X_to_predict[n]
                            -self.mean_k[k]))*matrix(self.covarianceMatrixSeparate[k]).I*matrix((X_to_predict[n]-self.mean_k[k])).T+np.log(float(self.Num_k[k])/N)
                    
                    '''result[n][k] = multivariate_normal.pdf(X_to_predict[n], mean=self.mean_k[k], 
                                    cov=self.covarianceMatrixSeparate[k])'''
                
            maxI = 0
            maxProb = -float("inf")
            
            for j in range(15):
                if (result[n][j] >= maxProb):
                    maxProb = result[n][j]
                    maxI = j
            Y.append(maxI)
        
        
        return np.asarray(Y)
        
    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
           plt.show()




























'''









# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance
        self.summaryOfAttr = {}
        

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __classifyIntoClass (self, X, Y):
        separated = {}
        
        for i in range(len(Y)):
            data = X[i]
            if Y[i] not in separated:
                separated[Y[i]] = []
                
            separated[Y[i]].append(data)
            
        return separated
        
    def __mean(self, X):
        return sum(X)/float(len(X))
        
    def __stdDev(self, X):
        mean = self.__mean(X)
        variance = sum([pow(x-mean,2) for x in X])/float(len(X)-1)
        return math.sqrt(variance)
        
    def __meanAndStdOfAttribute (self, ls):
        meanAndStd = [(self.__mean(attr), self.__stdDev(attr)) for attr in zip(*ls)]
        return meanAndStd
        
    def __calculatePDF (self, x, mean, stdev):
        exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))
        return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent


    def __calculateClassProbabilities(self, meanAndStdData, X):
        probabilities = {}
        for classValue, classSummaries in meanAndStdData.iteritems():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = X[i]
                probabilities[classValue] *= self.__calculatePDF(x, mean, stdev)
                
        return probabilities

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        
        separatedIntoClass = self.__classifyIntoClass(X, Y)     

        for i in separatedIntoClass:
            self.summaryOfAttr[i] = self.__meanAndStdOfAttribute(separatedIntoClass[i])
        
        #print self.summaryOfAttr

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        prob_x_data = []
        
        for i in range(len(X_to_predict)):
            classProbabilities = self.__calculateClassProbabilities(self.summaryOfAttr, X_to_predict[i])
            prob_x_data.append(classProbabilities)
        
        #print "prob_x_data", prob_x_data
        
        Y = []
        for item in prob_x_data:
            maximum = -1
            bestProb = -1
            for num, prob in item.iteritems():
                if prob > bestProb:
                    maximum = num
                    bestProb = prob
            
            Y.append(maximum)
        
        #print Y
        return np.array(Y)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()'''
            
        
