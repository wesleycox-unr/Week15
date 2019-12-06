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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


#    w0    w1  w2
initial_w = [0.0, 0.0, 0.0]

def plotCurrentW(ww, currentData, delay):
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

    # Create data points corresponding the the weights so that they can be plotted on the graph
    wDict = {'x1':[], 'x2':[], 'yp':[]}
    
    # y = yIntercept + gradient*x
    #   = w[0] + w[1]*x
    # To plot a straight line, we need the x and y at the far left (x=-1) and far right (x=+1)
    #0 = w[0] + w[1]*x1 + w[2]*x2
    #-w[0] = w[1]*x1 + w[2]*x2
    #x2 = (-w[0] -w[1]*x1)/w[2]
    
    leftleftY = ww[0] + ww[1]*(-1.0) + ww[2]*(-1.0)
    leftrightY = ww[0] + ww[1]*(-1.0) + ww[2]*(1.0)
    rightleftY = ww[0] + ww[1]*(1.0) + ww[2]*(-1.0)
    rightrightY = ww[0] + ww[1]*(1.0) + ww[2]*(1.0)

    wDict["x1"].append(-1.0)
    wDict["x2"].append(-1.0)
    wDict["yp"].append(leftleftY)
    
    wDict["x1"].append(-1.0)
    wDict["x2"].append(1.0)
    wDict["yp"].append(leftrightY)
    
    wDict["x1"].append(1.0)
    wDict["x2"].append(-1.0)
    wDict["yp"].append(rightleftY)
    
    wDict["x1"].append(1.0)
    wDict["x2"].append(1.0)
    wDict["yp"].append(rightrightY)
    
    # Convert to a dataframe so it can be plotted like the other data
    resultW = pd.DataFrame(wDict)

    xx1, xx2 = numpy.meshgrid(resultW.x1, resultW.x2)
    
    z = ww[0] + ww[1]*xx1 + ww[2]*xx2


    # Plot the corresponding classification separating line
    #plt.plot(resultW.x, resultW.yp)
    
    ax.plot_surface(xx1, xx2, z, alpha=0.2)

    plt.draw()
    plt.pause(delay)
    plt.clf()
    
def getEin(ww, data):
    Ein = 0
    # Sum the error for each datapoint to get the overall in sample error, Ein
    for index in range(0,len(data)):
        h = ww[0] + ww[1]*data["x1"][index] + ww[2]*data["x2"][index]
        Ein += (h - data["yp"][index])*(h - data["yp"][index])
    
    # Needs to be scaled by the number of datapoints
    Ein = Ein/len(data)

    return Ein
    
    
def regression(data):

    # Copy the initial weights so we can change them. Copy by value so they dont point to the same thing
    w = copy.deepcopy(initial_w)
    
    # How many times to we want to keep trying to update, configured by trial and error
    max_iterations = 300
    iteration = 0
    
    # How quickly should the weights update, configured by trial and error
    learning_rate = 1
    
    # Initial values so we can store the best fit
    #bestW = copy.deepcopy(initial_w)
    #minEin = getEin(initial_w, data)
    
    while iteration < max_iterations:
        
        print("Iteration: {0}".format(iteration))
        
        # Plotting is the slowest part of each iteration. We want to see it updating, so lets 
        # just check after every 10 updates
        if iteration%10 == 0:
            plotCurrentW(w, data, 0.1)
        
        derivative = [0,0,0]
        
        for i in range(0, len(data)):
            
            # Based on our weights, predict the y value
            h = w[0] + w[1]*data["x1"][i] + w[2]*data["x2"][i]
        
            # Create a vector so that we can do multiplication with the weight vector more easily
            x = [1, data['x1'][i], data['x2'][i]]
            
            # Derivative is 2x(h-y) for each datapoint, scaled by total number of points
            dw = [2*elem*(h - data["yp"][i]) for elem in x] # Use a lambda function, could be done in a regular loop
            derivative[0] += dw[0]
            derivative[1] += dw[1]
            derivative[2] += dw[2]
            
        derivative[0] = derivative[0]/len(data)
        derivative[1] = derivative[1]/len(data)
        derivative[2] = derivative[2]/len(data)
            
        # Update the weights
        w[0] = w[0] - learning_rate*derivative[0]
        w[1] = w[1] - learning_rate*derivative[1]
        w[2] = w[2] - learning_rate*derivative[2]
        
        print(w)
        
        ## Get the current in sample error for these weights
        #Ein = getEin(w, data)
        
        #if Ein < minEin:
        #    # Least error seen so far
        #    minEin = Ein
        #    bestW = copy.deepcopy(w)
           
        
        
        
        iteration += 1
    
    
    return w # Return the predicting weights

input_data = pd.read_json("linear_data_2D.txt")
#print(input_data)

# Perform the learning
result_w = regression(input_data)

print(result_w)

#initialEin = getEin(initial_w, input_data)
#bestEin = getEin(result_w, input_data)

print("Initial Ein {0} result Ein {1}".format(initialEin, bestEin))

plotCurrentW(result_w, input_data, 60)

