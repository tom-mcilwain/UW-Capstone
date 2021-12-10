import numpy as np
import pandas as pd
import trial
import os

'''
KALMAN
------
This class represents the Kalman filter of a cursor. Each cursor has it's own Kalman filter. This class contains
all the matrices as well as methods to do the calculations.

This class also includes all of the closed-loop decoder adaptation algorithms.

'''


class KalmanFilter():

    num_states = 3
    num_observations = trial.NUM_SENSORS

    a = 0.95
    beta = 0.5

    # Algorithm Parameters
    rho = 0.95 # SmoothBatch weighting parameter
    alpha = 0.999834979 # Adaptive KF weighting parameter
    mu = 0.15 # Step size

    batch_length = None
    BATCH_DURATION = 300 # Duration of batch in seconds for the Batch algorithm
    SMOOTH_BATCH_DURATION = 80 # Duration of batch in seconds for the SmoothBatch algorithm

    A = [[a, 0, 0], [0, a, 0], [0, 0, 1]]
    W = np.identity(num_states)
    C = np.empty((num_observations, num_states))
    Q = np.empty((num_observations, num_observations))

    P = []
    K = []
    xt = [] #[] # predicted velocities

    algorithm = ''

    def __init__(self, algorithm):
        self.P = np.zeros((self.num_states, self.num_states))
        self.K = np.zeros((self.num_states, self.num_observations))
        self.xt = np.asarray([[0], [0], [1]])
        self.algorithm = algorithm
        if algorithm == 'batch':
            self.batch_length = self.BATCH_DURATION
        elif algorithm == 'smooth_batch':
            self.batch_length = self.SMOOTH_BATCH_DURATION

    # Fitting KF matrices with calibration data
    def initialize(self, file_name):
        data = pd.read_csv(file_name)
        data = trial.standardize(data)
        states = data.iloc[:, 1:trial.STATE_INDEX]
        velocities = data.iloc[:, trial.INTENDED_INDEX:]
        Y = np.transpose(states)
        X = self.add_constant(np.transpose(velocities))
        end = X.shape[1]
        X1 = X[:, 0:end-1]
        X2 = X[:, 1:end]
        difference = np.subtract(X2, np.matmul(self.A, X1))
        self.W = np.matmul(difference, np.transpose(difference)) / (end - 1)
        self.batch(states, velocities)

    # Shuffling data given by file_name
    def randomize(self, file_name):
        data = pd.read_csv(file_name)
        states = data.iloc[:, 1:trial.STATE_INDEX].copy()
        kinematics = data.iloc[:, trial.STATE_INDEX:].copy()
        shuffled_states = states.sample(frac=1)
        shuffled_data = pd.DataFrame(columns=data.columns[1:])
        for i in range(len(data)):
            row = np.append(shuffled_states.iloc[i,:], kinematics.iloc[i,:])
            shuffled_data.loc[len(shuffled_data)] = row
        print(shuffled_data)

        file_name = f"shuffled_{file_name}"
        if os.path.exists(file_name):
            header = False
        else:
            header = True
        with open(file_name, 'a') as file:
            shuffled_data.to_csv(file, header=header)
        return shuffled_data

    # Center the model
    def center(self):
        self.xt = np.asarray([[0], [0], [1]])

    # Perform a KF iteration
    def predict(self, state):
        self.time_update()
        self.measurement_update(state)
        return self.xt[0:2]

    # A priori estimate
    def time_update(self):
        self.xt = np.matmul(self.A, self.xt) # + wt # Projecting the kinematic state
        self.P = np.matmul(self.A, np.matmul(self.P, np.transpose(self.A))) + self.W # Projecting the error

    # Second update of the KF iteration
    def measurement_update(self, state):
        self.gain()
        self.update_estimate(state)
        self.update_error()

    # Calculating Kalman gain
    def gain(self):
        ptc = np.matmul(self.P, np.transpose(self.C))
        sum = np.add(np.matmul(self.C, ptc), self.Q)
        inv = np.linalg.pinv(sum)
        self.K = np.matmul(ptc, inv)

    # Updating estimate of cursor kinematics
    def update_estimate(self, state):
        weighted_estimate = np.matmul(self.C, self.xt)
        residual = np.subtract(state.reshape(self.num_observations, 1), weighted_estimate)
        self.xt = np.add(self.xt, np.matmul(self.K, residual))

    # Updating prediction error
    def update_error(self):
        product = np.matmul(self.K, self.C)
        identity = np.identity(product.shape[0])
        diff = identity - product
        self.Pt = np.matmul(diff, self.P)

    # Adding a constant to keep mean constant
    def add_constant(self, x):
        size = x.shape[1]
        x = np.r_[x, np.ones((1, size))]
        return x

    # Perform maximum likelihood estimation, updating the C and Q matrices
    def MLE(self, Y, X):
        Y = np.asarray(np.transpose(Y))
        X = np.asarray(np.transpose(X))
        X = self.add_constant(X)

        # Calculating C
        product = np.matmul(X, np.transpose(X))
        inverse = np.linalg.pinv(product)
        C = np.matmul(Y, np.transpose(X))
        C = np.matmul(C, inverse)

        # Calculating Q
        difference = np.subtract(Y, np.matmul(C, X))
        Q = np.matmul(difference, np.transpose(difference))
        Q = Q / Y.shape[1]

        return C, Q

    # Perform batch update
    # Y = glove states, X = intended velocities
    def batch(self, Y, X):
        self.C, self.Q = self.MLE(Y, X)

    # Perform adaptive KF update (in real-time at every iteration)
    def adaptive(self, yt, xt):
        xt = np.reshape(xt, (2, 1))
        yt = np.reshape(yt, (self.num_observations, 1))
        xt = self.add_constant(xt)

        # Calculating C
        temp_C = np.subtract(np.matmul(self.C, xt), yt)
        temp_C = np.matmul(self.mu * temp_C, np.transpose(xt))
        self.C = np.subtract(self.C, temp_C)

        # Calculating Q
        temp_Q = np.subtract(yt, np.matmul(self.C, xt))
        temp_Q = (1 - self.alpha) * np.matmul(temp_Q, np.transpose(temp_Q))
        self.Q = np.add((self.alpha * self.Q), temp_Q)

    # Perform SmoothBatch update
    def smooth_batch(self, Y, X):
        C, Q = self.MLE(Y, X)
        self.C = np.add((1 - self.rho)*C, self.rho*C)
        self.Q = np.add((1 - self.rho)*Q, self.rho*Q)