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
import copy
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#    w0    w1  w2
initial_w = [0.0, 0.0, 0.0]


def plotCurrentW(ww, values, delay):
	# Plot the randomly generated data
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(values["x1"],values["x2"],values["yp"], marker=".")
	ax.set_xlim(-1.1,1.1)
	ax.set_ylim(-1.1,1.1)
	ax.set_zlim(-1.1,1.1)
	ax.set_xlabel("x1")
	ax.set_ylabel("x2")
	ax.set_zlabel("yp")

	# Create data points corresponding the the weights so that they can be plotted on the graph
	wDict = {'x1':[], 'x2':[], 'yp':[]}
	# [-1,-1]
	leftleftY = ww[0] + ww[1]*(-1) + ww[2]*(-1)
	leftrightY = ww[0] + ww[1]*(-1) + ww[2]*(+1)
	rightleftY = ww[0] + ww[1]*(+1) + ww[2]*(-1)
	rightrightY = ww[0] + ww[1]*(+1) + ww[2]*(+1)

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

	resultW = pd.DataFrame(wDict)

	xx1, xx2 = numpy.meshgrid(resultW.x1, resultW.x2)

	z = ww[0] + ww[1]*xx1 + ww[2]*xx2

	print(resultW.yp)
	# Plot the corresponding classification separating line
	ax.plot_surface(xx1, xx2, z, alpha=0.2)

	plt.draw()
	plt.pause(delay)
	plt.clf()

def calculateEin(data, this_w):
	Ein = 0

	N = len(data)

	for index in range(0,N):
		h = this_w[0] + this_w[1]*data["x1"][index] + this_w[2]*data["x2"][index]

		Ein += (h - data["yp"][index])*(h - data["yp"][index])

	Ein = Ein/N
	return Ein

def totalEin(ww, data):
	Ein = 0
	for index in range(len(data)):
		h = ww[0] + ww[1]*data['x1'][index] + ww[2]*data['x2'][index]
		Ein += (h - data['yp'][index])*(h - data['yp'][index])

	return Ein

def regression(data):

	# Implement the perceptron learning algorithm to determine the weights w that will fit the data
	w = copy.deepcopy(initial_w)

	learning_rate = 0.1


	max_interations = 300
	iteration = 0

	minEin = totalEin(initial_w, data)
	bestW = copy.deepcopy(initial_w)

	while iteration < max_interations:
		index = numpy.random.randint(0,len(data))
		print("Iteration {0} index {1} with w {2}".format(iteration, index, w))

		if iteration % 10 == 0:
			plotCurrentW(w, data, 0.0001)	

		

		# new_Ein = totalEin(w, data)

		# if new_Ein < minEin:
		# 	minEin = new_Ein
		# 	bestW = copy.deepcopy(w)



		derivative = [0,0,0]

		for i in range(0, len(data)):
			h = w[0] + w[1]*data['x1'][i] + w[2]*data['x2'][i]
			x = [1, data['x1'][i], data['x2'][i]]


			dw = [2*elem*(h - data["yp"][i]) for elem in x]
			derivative[0] += dw[0]
			derivative[1] += dw[1]
			derivative[2] += dw[2]

		derivative[0] = derivative[0]/len(data)
		derivative[1] = derivative[1]/len(data)
		derivative[2] = derivative[2]/len(data)

		w[0] = w[0] - learning_rate*derivative[0]
		w[1] = w[1] - learning_rate*derivative[1]
		w[2] = w[2] - learning_rate*derivative[2]



		iteration += 1


	# return bestW # Return the predicting weights
	return w

input_data = pd.read_json("linear_data_2D.txt")

# Perform the learning
w_result = regression(input_data)

print("Initial Ein: {0} predicted Ein: {1}".format(totalEin(initial_w, input_data), totalEin(w_result, input_data)))

plotCurrentW(w_result, input_data, 60)