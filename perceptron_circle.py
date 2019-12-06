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
import copy

#    w0    w1  w2
initial_w = [0.1, 0.1, 0.1]

def plotCurrentW(ww, currentData, delay):
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

    # Create data points corresponding the the weights so that they can be plotted on the graph
    wDict = {'x1':[], 'x2':[]}
    leftx2 = (ww[1] - ww[0])/ww[2]
    rightx2 = (-ww[1] - ww[0])/ww[2]

    wDict["x1"].append(-1.0)
    wDict["x2"].append(leftx2)
    wDict["x1"].append(1.0)
    wDict["x2"].append(rightx2)

    resultW = pd.DataFrame(wDict)

    # Plot the corresponding classification separating line
    plt.plot(resultW.x1, resultW.x2)


def linearize(data):

    new_data = copy.deepcopy(data)
    new_data["x1"] = new_data["x1"]*new_data["x1"]
    new_data["x2"] = new_data["x2"]*new_data["x2"]
    
    return new_data
    
def perceptron(data):

    # Implement the perceptron learning algorithm to determine the weights w that will fit the data
    w = initial_w
    
    
    
    #print(data)
    
    #for row in data.iterrows():
    index = 0
    count = 0
    
    while index < len(data):
        print("Index {0} of {1}".format(index, len(data)))
        count += 1
        
        if count %20 == 0:
            plotCurrentW(w, data, 0.0001)
        row = data.iloc[index]
        
        #print(row)
        #sys.exit()
        #print(row[1]["x1"])
        h = w[0]*1 + w[1]*row["x1"] + w[2]*row["x2"]
        
        if numpy.sign(h) != numpy.sign(row["yp"]):
            # Mismatch
            # Update w values
            
            #w[w0 w1 w2] = w + yp * data
            w[0] = w[0] + row["yp"]*1
            w[1] = w[1] + row["yp"]*row["x1"]
            w[2] = w[2] + row["yp"]*row["x2"]
            index = 0
        else:
            index += 1
    # Steps:
    # - See if the current weights will correctly predict the yp values in the DataFrame for all rows
    # - If so, done.
    # - If not, choose a row that isnt predicted correctly, and update the weights by the scalar product of yp with [1, x1, x2]
    # - Repeat until all rows are correctly predicted


    return w # Return the predicting weights

input_data = pd.read_json("linear_data_circle.txt")

lin_data = linearize(input_data)

# Perform the learning
result_w = perceptron(lin_data)

print(result_w)

plotCurrentW(result_w, lin_data, 30)