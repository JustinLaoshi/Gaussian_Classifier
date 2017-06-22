from __future__ import division
from gaussianClassifier import GaussianClassifier
import numpy as np
import scipy.io
import math
import sklearn
from sklearn import preprocessing
import itertools
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal


def run():	
	def contrastNormalize(images):
		new = np.zeros((60000, 784))
		for i in range(0, 60000):
			new[i] += images[i]
			imgSum = 0.0
			for j in range(0, 784):
				imgSum += np.power(new[i][j], 2)
			imgSum = np.sqrt(imgSum)
			new[i] /= imgSum
		return new	


	train = scipy.io.loadmat('data/digit_dataset/train.mat')
	test = scipy.io.loadmat('data/digit_dataset/test.mat')
	spam = scipy.io.loadmat('data/spam_dataset/spam_data.mat')
	spamData = spam['training_data']
	spamLabels = spam['training_labels']
	spamTest = spam['test_data']
	testImages = test['test_images']
	testImages = sklearn.preprocessing.normalize(testImages, 'l2', 1, True)
	images = train['train_images'].transpose()
	images = images.reshape(60000, 784)

	images = sklearn.preprocessing.normalize(images, 'l2', 1, True)

	labels = train['train_labels'].ravel()
	
	indexDict = {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
	priorDict = {}
	meanDict = {}
	covDict = {}

	
	def findCovMatrix(images, start, end):
		subset = images[start:end]
		return np.cov(subset.T)

	def findMean(images, start, end):
		subset = images[start:end]
		return np.mean(subset, 0)

	############# ABC #######################


	start = 0
	end = indexDict[0]
	for digit in indexDict:
		mean = findMean(images, start, end)
		cov = findCovMatrix(images, start, end)
		# plt.imshow(cov)
		# plt.show()
		meanDict[digit] = mean
		covDict[digit] = cov
		priorDict[digit] = float(end - start - 1) / 60000.0
		start = end
		if digit == 9:
			break
		end += indexDict[digit+1]


	############## D #########################

	def findPrior(labels, spam=False):
		priorDict = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0, 9:0.0}
		if spam:
			priorDict = {0:0.0, 1:0.0}
		total = labels.shape[0]
		for label in labels:
			if spam:
				priorDict[label[0]] += 1.0
			else:
				priorDict[label] += 1.0

		for digit in priorDict:
			priorDict[digit] /= total
		return priorDict

	def findSampleMean(sample, size, label, spam=False):
		if spam:
			meanDict = {0:[], 1:[]}
			count = {0:0.0, 1:0.0}
		else:
			meanDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
			count = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0, 9:0.0}
		
		for i in range(0, size):
			currLabel = label[i]
			if spam:
				meanDict[currLabel[0]].append(sample[i])
			else:
				meanDict[currLabel].append(sample[i])

		for digit in meanDict:
			meanDict[digit] = np.array(meanDict[digit])
			meanDict[digit] = np.mean(meanDict[digit], 0)
			#meanDict[digit] /= count[digit]
		return meanDict

	def findCov(sample, size, label, spam=False):
		if spam:
			covDict = {0:np.zeros((32,32)), 1:np.zeros((32,32))}
			spamDict = {0: [], 1: []}
			for i in range(0, size):
				currLabel = label[i]
				spamDict[currLabel[0]].append(sample[i])
			for spam in spamDict:
				spamDict[spam] = np.array(spamDict[spam])
				covDict[spam] = np.cov(spamDict[spam].T)
			return covDict
		else:
			covDict = {0:np.zeros((784, 784)), 1:np.zeros((784, 784)), 2:np.zeros((784, 784)), 3:np.zeros((784, 784)), 4:np.zeros((784, 784)), 5:np.zeros((784, 784)), 6:np.zeros((784, 784)), 7:np.zeros((784, 784)), 8:np.zeros((784, 784)), 9:np.zeros((784, 784))}
			digitDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
			for i in range(0, size):
				currLabel = label[i]
				digitDict[currLabel].append(sample[i])
			for digit in digitDict:
				digitDict[digit] = np.array(digitDict[digit])
				covDict[digit] = np.cov(digitDict[digit].T)
			return covDict

	def findOverallCov(covDict, spam=False):
		if spam:
			avg = np.zeros((32,32))
			for sp in covDict:
				avg += covDict[sp]
			avg /= 2.0
			return avg

		else:
			avg = np.zeros((784, 784))
			for digit in covDict:
				avg += covDict[digit]
			avg /= 10.0
			return avg



	sampleSizes = [100,200,500,1000,2000,5000,10000] # This is the fully train the classifier 60000]
	errorDict1 = {}
	errorDict2 = {}
	images, labels = sklearn.utils.shuffle(images, labels)
	validationSet = images[50000:]
	errRateList1 = []
	errRateList2 = []
	test1 = findSampleMean(images, 100, labels)
	test2 = findMean(images, 0, 100)

	for size in sampleSizes:
		sample = images[:size]
		label = labels[:size]
		priorDict = findPrior(label)
		covDict = findCov(sample, size, label)
		overallCov = findOverallCov(covDict)
		meanDict = findSampleMean(sample, size, label)


		## this block is d) i)
		# gc1 = GaussianClassifier(meanDict, overallCov, covDict, priorDict, True)
		# pred_labels1 = gc1.predict(validationSet)
		# a, b = benchmark(pred_labels1.T, labels[50000:].reshape(1, 10000))
		# errorDict1[size] = a
		# errRateList1.append(a)


		# this block is d) ii)
		gc2 = GaussianClassifier(meanDict, overallCov, covDict, priorDict, False)
		pred_labels2 = gc2.predict(validationSet)
		c, d = benchmark(pred_labels2.T, labels[50000:].reshape(1, 10000))
		errorDict2[size] = c
		errRateList2.append(c)

	# plotting code
	plt.plot(errRateList2, sampleSizes, 'ro')
	plt.ylabel('Number of Training Examples')
	plt.xlabel('Error Rate')
	axes = plt.gca()
	axes.set_xlim([0.0, 1.0])
	axes.set_ylim([0, 12000])
	plt.show()

	testLabels = gc2.predict(testImages)
	csvList = [['Id,Category']]
	for i in range(1, 10001):
	    csvList.append([i, int(testLabels[i-1][0])])

	with open('problem5digits.csv', 'w', newline='') as fp:
	    a = csv.writer(fp, delimiter=',')
	    a.writerows(csvList)



	sample = spamData
	label = spamLabels.T
	priorDict = findPrior(label, True)
	covDict = findCov(sample, 5172, label, True)
	overallCov = findOverallCov(covDict, True)
	meanDict = findSampleMean(sample, 5172, label, True)
	gcSpam = GaussianClassifier(meanDict, overallCov, covDict, priorDict, False, True)
	testLabelsSpam = gcSpam.predict(spamTest)
	csvList = [['Id,Category']]
	for i in range(1, 5858):
	    csvList.append([i, int(testLabelsSpam[i-1][0])])

	with open('problem5spam.csv', 'w', newline='') as fp:
	    a = csv.writer(fp, delimiter=',')
	    a.writerows(csvList)
      
      
run()
    
