# perceptron.py
# Written by: Wes Cox for IS485/485
# Oct 31, 2019
# Starting point for Question 1 of Programming Problem Set #7
#
# For a randomly generated set of data, perform the perceptron learning algorithm
# to correctly classify the points

import numpy
import matplotlib.pylab as plt
import pandas as pd
import sys
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Number of data points
N = 200


def getRand():
    return numpy.random.uniform(-1,1)

def getY(xVal1, xVal2):
    
    
    std_dev = 0.2
    
    # x1 = -1, x2 = -1, y = -1
    # x1 = 0, x2 = 0, y = 0
    # x1 = 1, x2 = 1, y = 1
    point = numpy.random.normal(xVal1/2 + xVal2/2, std_dev)  
    
    # Offset the generated data using a normal distribution to give it some noise
    #y = numpy.random.normal(lineY, std_dev)

    return point

def plotData(currentData, delay):
    # Plot the randomly generated data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(currentData["x1"],currentData["x2"],currentData["yp"], marker=".")
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("yp")

    plt.draw()
    plt.pause(delay)
    plt.clf()
    
    
def createData():
    
    # Initialize an empty dictionary to hold our data
    dataDict = {"x1": [], "x2": [], "yp":[]}

    # Create N datapoints
    for i in range(N):
        x1 = getRand()
        x2 = getRand()
        y = getY(x1, x2)

        dataDict['x1'].append(x1)
        dataDict['x2'].append(x2)
        dataDict['yp'].append(y)

    data = pd.DataFrame(dataDict)
    return data

input_data = createData()

print(input_data)

# Save generated data to a JSON file (CSV probably would have made better sense)
input_data.to_json("linear_data_2D.txt")

plotData(input_data, 30)