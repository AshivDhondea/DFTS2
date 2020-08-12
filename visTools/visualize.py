import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import os
import yaml

def userInterface():
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--params', help='path to plots config file')
	args = vars(ap.parse_args())

	fileName = args['params']
	with open(fileName) as c:
		config = yaml.load(c)

	return config

def producePlots(paramsDict):
	yVals = paramsDict['yVals']
	yLen = len(yVals[0])

	if len(paramsDict['xVals'])!=yLen or len(paramsDict['labels'])!=len(yVals):
		raise ValueError('x and y coordinates cannot be resolved into pairs')
	yCoords = []
	for i in range(len(yVals)):
		temp = []
		for j in range(len(yVals[i])):
			a = np.load(yVals[i][j])
			a = np.mean(a, axis=0)
			# print(a)
			temp.append(a[-1])
		yCoords.append(temp)
	for i in range(len(yCoords)):
		plt.plot(paramsDict['xVals'], yCoords[i], label=paramsDict['labels'][i])
	plt.xlabel(paramsDict['xlabel'])
	plt.ylabel(paramsDict['ylabel'])
	plt.legend(loc='lower right')

	if not os.path.exists(paramsDict['plotDir']):
		try:
			os.makedirs(paramsDict['plotDir'])
		except OSError as exc:
			if exc.errno != errno.EXIST:
				raise
	plt.savefig(os.path.join(paramsDict['plotDir'], paramsDict['fileName']))


if __name__=='__main__':
	paramsDict = userInterface()
	producePlots(paramsDict)