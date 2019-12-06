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

# Number of data points
N = 200


def getRand():
    return numpy.random.uniform(-1,1)

def sign(x1, x2):
    
    valY = x1*x1 + x2*x2
    
    radius = 0.3
    
    if valY > radius:
        return 1
    else:
        return -1

def plotData(currentData, delay):
    # Plot the randomly generated data
    plt.scatter(currentData[currentData["yp"] == 1]["x1"],currentData[currentData["yp"] == 1]["x2"], marker="o")
    plt.scatter(currentData[currentData["yp"] == -1]["x1"],currentData[currentData["yp"] == -1]["x2"], marker="o")
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("x")
    plt.ylabel("y")

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
        y = sign(x1,x2)

        dataDict['x1'].append(x1)
        dataDict['x2'].append(x2)
        dataDict['yp'].append(y)

    data = pd.DataFrame(dataDict)
    return data

input_data = createData()

print(input_data)

# Save generated data to a JSON file (CSV probably would have made better sense)
input_data.to_json("linear_data_circle.txt")

plotData(input_data, 30)