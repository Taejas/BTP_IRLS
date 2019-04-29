import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# File name
FILE_NAME = 'sos1_49.csv'
# FILE_NAME = 'artificial_5.csv'
DELTA = 1e-5
NUM_ITER = 10000
# PREV_TIME = 1

ACC_LIST = [0.01, 0.05, 0.1, 0.2]

if __name__ == '__main__':

	# Reading the data; rows represent time points and columns represent genes.
	df = pd.read_csv(FILE_NAME)
	data = df.values
	#data = (data - data.min(0)) / data.ptp(0)
	#data = data / data.max(0)

	data = (data - data.mean(0)) / data.std(0)
	N = len(data[0])
	genes = N
	T = len(data)

	X = data
	C = np.dot(X.T, X) / len(X)
	mc = np.linalg.det(C)
	print(mc)

	vif = [variance_inflation_factor(X, i) for i in range(N)]

	X = np.c_[X, (X[:, 1] + X[:, 3] + X[:, 4]) / 3]
	X = np.delete(X, 4, 1)
	X = np.delete(X, 3, 1)
	X = np.delete(X, 1, 1)

	N = len(X[0])
	# X = (X - X.mean(0)) / X.std(0)

	C2 = np.dot(X.T, X) / len(X)
	mc = np.linalg.det(C)
	print(mc)

	vif = [variance_inflation_factor(X, i) for i in range(N)]

	data2 = X

	X_list = []
	Y_list = []

	# t - 1
	X_list.append(data2[: -1])
	Y_list.append(data[1 :])

	# # t - 2
	# X_list.append(data2[: -2])
	# Y_list.append(data[2 :])

	# # t - 3
	# X_list.append(data2[: -3])
	# Y_list.append(data[3 :])

	# # Prev 2 avg
	# X = np.zeros((len(data2) - 2, len(data2[0])))
	# for j in range(len(data2) - 2):
	# 	X[j] = data2[j : j + 2, :].mean(0)
	# X_list.append(X)
	# Y_list.append(data[2 :])

	# # Prev 3 avg
	# X = np.zeros((len(data2) - 3, len(data2[0])))
	# for j in range(len(data2) - 3):
	# 	X[j] = data2[j : j + 3, :].mean(0)
	# X_list.append(X)
	# Y_list.append(data[3 :])

	result_list = []
	binary_result_list = []
	counts_list = []
	W_list = []

	for ds in range(len(X_list)):
		
		X = X_list[ds]
		N = len(X[0])
		T = len(X)
		X = np.c_[X, np.ones(len(X))]

		result = np.zeros((genes, N + 1))
		counts = np.zeros((genes, len(ACC_LIST)))
		
		for i in range(genes):

			y = Y_list[ds][:, i]
			W = np.identity(T)

			for outer in range(NUM_ITER):

				beta = np.linalg.multi_dot([np.linalg.inv(np.linalg.multi_dot([X.T, W, X])), X.T, W, y])
				for k in range(T):
					W[k][k] = 1 / max(DELTA, abs(y[k] - np.dot(X[k], beta)))

			result[i] = beta

			for j in range(len(ACC_LIST)):
				count = 0
				for k in range(T):
					if(1 / W[k][k] <= abs(y[k] * ACC_LIST[j])):
						count += 1
				counts[i][j] = count
			# for acc in [100, 20, 10, 5]
			# 	counts.append(sum(sum(W >= acc)))

			W_list.append(W)
		
		result = result[:, : -1]
		result_list.append(result)
		
		binary_result_list.append(abs(result) > abs(result).mean(1).reshape(len(result), 1))
		
		# binary_result_list.append(abs(result - result.mean(1).reshape(len(result), 1)) > result.std(1).reshape(len(result), 1))
		
		# binary_result_list.append(abs(result) > abs(result).mean(1).reshape(len(result), 1) + abs(result).std(1).reshape(len(result), 1) / 2)
		# print(abs(result).mean(1))
		# print(abs(result).std(1) / 2)
		# print(abs(result).mean(1) + abs(result).std(1) / 2)
		
		counts_list.append(counts)