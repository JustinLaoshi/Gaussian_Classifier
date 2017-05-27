from __future__ import division
import numpy as np
import scipy.io
import math
import itertools
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt 


class GaussianClassifier:

	def __init__(self, meanDict, avgCov=None, covDict=None, priorDict=None, ldaClass=False, spam=False):
		self.meanDict = meanDict
		self.avgCov = avgCov
		self.covDict = covDict
		self.priorDict = priorDict
		self.ldaClass = ldaClass
		if ldaClass:
			if spam:
				k = 32
			else:
				k = 784			
			for i in range(0, k):
				for j in range(0, k):
					if i == j:
						self.avgCov[i][j] += 1e-6
		else:
			if spam:
				k = 32
			else:
				k = 784
			for digit in self.covDict:	
				for i in range(0, k):
					for j in range(0, k):
						if i == j:
							self.covDict[digit][i][j] += 1e-6


	def predict(self, images):
		predLabels = np.empty((images.shape[0], 1))
		i = 0
		ldaDict = {}
		qdaDict = {}
		if self.ldaClass:
			invCov = np.linalg.inv(self.avgCov)
		term1Dict = {}
		term2Dict = {}
		term3Dict = {}
		invDict = {}
		detDict = {}
		for digit in self.meanDict:
			mean = self.meanDict[digit]
			if self.ldaClass:
				term1Dict[digit] = np.dot(invCov, mean)
				term2Dict[digit] = np.dot(mean.T, np.dot(invCov, mean))
			else:
				invDict[digit] = np.linalg.inv(self.covDict[digit])
				term1Dict[digit] = np.dot(invDict[digit], mean)
				term2Dict[digit] = np.dot(mean.T, np.dot(invDict[digit], mean))
				term3Dict[digit] = np.linalg.slogdet(self.covDict[digit])[0] * np.exp(np.linalg.slogdet(self.covDict[digit])[1]) 
				
		for img in images:
			for digit in self.meanDict:
				prior = self.priorDict[digit]
				if self.ldaClass:
					lda = np.dot(img.T, term1Dict[digit]) - 0.5 * term2Dict[digit] + np.log(prior)
					ldaDict[digit] = lda
				else:
					qda = (-.5 * np.dot(img.T, np.dot(invDict[digit], img))) + np.dot(img.T, term1Dict[digit]) - (.5 * term2Dict[digit]) - (.5 * term3Dict[digit]) + np.log(prior)
					qdaDict[digit] = qda
			if self.ldaClass:
				predLabels[i] =  max(ldaDict, key=ldaDict.get)
			else:
				predLabels[i] =  max(qdaDict, key=qdaDict.get)
			i += 1

		return predLabels



