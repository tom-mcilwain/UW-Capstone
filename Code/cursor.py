import trial
import math
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal


'''
CURSOR
------
This class controls the cursor movement, iteration, updating, etc. This class keeps track of all the
cursor-related stuff like what target it corresponds to, the data of the cursor, etc.

An instance of this class is made for each cursor.

'''


class Cursor():

    # Initializing fields
    model = None
    target_loc = None
    calibrating = None
    cursor_name = ""
    cursor_num = None
    target_label = None
    position = None
    dimensions = 2
    subject_id = None
    calibration_signal = None

    # Building a Pandas DataFrame
    data = pd.DataFrame(columns=range(22))
    data = pd.DataFrame(columns=data.columns.tolist() + ['x', 'y', 'vx', 'vy', 'ix', 'iy'])

    # Creating event vectors
    events = []
    event_indeces = []

    # Initialize the cursor
    def __init__(self, model, calibrating, dimensions, cursor_num, subject_id):
        self.cursor_num = cursor_num
        self.position = trial.center[cursor_num]
        self.cursor_name = f"Cursor{str(cursor_num)}"
        self.calibrating = calibrating
        if calibrating:
            num_samples = trial.freq * trial.calibration_duration
            self.calibration_signal = signal.gaussian(num_samples, math.sqrt(num_samples))
            self.calibration_signal = (4 / np.sum(self.calibration_signal)) * self.calibration_signal
        self.dimensions = dimensions
        self.model = model
        self.subject_id = subject_id

    def center(self):
        self.position = trial.center[self.cursor_num]
        self.model.center()

    # Perform an iteration of the cursor, i.e. predict the velocity based on the state
    # Returns the position, velocity of the cursor
    def iterate(self, state):

        # if calibrating, cursor moves in straight line to target
        if self.calibrating:
            # Setting velocity as a gaussian profile
            diff = np.subtract(self.target_loc, trial.center[self.cursor_num]) / 4
            if trial.to_center[self.cursor_num]:
                diff = -diff
            predicted_velocity = self.calibration_signal[trial.trial_iter] * diff
            predicted_position = self.position + predicted_velocity
        else:

            state = state.to_numpy()

            # Finding predicted velocity
            predicted_velocity = self.model.predict(state).reshape(2)
            predicted_position = np.add(self.position, predicted_velocity)

            # Constraining the kinematics so the cursor stays within the task space
            predicted_position[0] = self.constrain(predicted_position, 0)
            predicted_position[1] = self.constrain(predicted_position, 1)
            predicted_velocity = predicted_position - self.position
            self.model.xt = np.append(predicted_velocity, 1).reshape(3,1)

            # Cursor only moves in 1 direction if one-dimensional
            if self.dimensions == 1:
                predicted_velocity[0] = 0

        target_pos = self.target_loc
        if trial.to_center[self.cursor_num]:
            target_pos = trial.center[self.cursor_num]

        intended_velocity = self.estimate_intention(predicted_velocity, target_pos)

        # Formatting and creating a row representing an iteration to save
        iteration = np.append(state, predicted_position)
        iteration = np.append(iteration, predicted_velocity)
        iteration = np.append(iteration, intended_velocity)

        # Saving iteration in dataframe
        data_length = len(self.data.index)
        self.data.loc[data_length] = iteration

        # Updating kalman filter parameters **only if algorithm type is adaptive, real time**
        if self.model.algorithm == 'adaptive' and not self.calibrating:
            self.model.adaptive(state, intended_velocity)

        # Setting position to the predicted position
        if self.dimensions == 1:
            self.position = [self.position[0], predicted_position[1]]
        else:
            self.position = predicted_position

        # Returning data
        return predicted_position, predicted_velocity


    # Method to estimate the intention of the user (CursorGoal)
    def estimate_intention(self, predicted_velocity, target_pos):
        x_pos = self.position[0]
        y_pos = self.position[1]
        x_vel = predicted_velocity[0]
        y_vel = predicted_velocity[1]
        magnitude = np.sqrt(np.add(np.square(x_vel), np.square(y_vel)))
        x_diff = np.subtract(target_pos[0], x_pos)
        y_diff = np.subtract(target_pos[1], y_pos)
        angle = math.atan2(y_diff, x_diff)
        x_unit = math.cos(angle) #1 / math.sqrt(1 + (math.tan(angle)**2))
        y_unit = math.sin(angle) #x_unit * math.tan(angle)
        intended_velocity = np.multiply(np.asarray([x_unit, y_unit]), magnitude)
        return intended_velocity


    # Updates parameters at every batch length
    # start = starting index of batch
    def batch_update(self, start):
        states = self.data.iloc[int(start):,0:22]
        intended_vel = self.data.iloc[int(start):, 26:]
        if self.model.algorithm == 'batch':
            self.model.batch(states, intended_vel)
        elif self.model.algorithm == 'smooth_batch':
            self.model.smooth_batch(states, intended_vel)
        print("Updated decoder parameters.")


    # Constrains the position of the cursor to remain within the screen
    def constrain(self, position, index):
        if abs(position[index] - trial.center[self.cursor_num][index]) > 5:
            position[index] = trial.center[self.cursor_num][index] + (5 * np.sign(position[index] - (index * 0.5)))
        return position[index]


    # Saves data to the relevant file
    def save_data(self):
        dt = datetime.now()
        date = f"{dt.month}_{dt.day}_{dt.year}"
        time = f"{dt.hour}_{dt.minute}"
        file_name = f"subject{self.subject_id}_{self.cursor_name}"
        events_file = ""
        time_file = ""
        if self.calibrating:
            data_file = f"{file_name}_calibration_data.csv"
        else:
            data_file = f"{file_name}_{date}_{time}_data.csv"
            events_file = f"{file_name}_{date}_{time}_events.csv"
            time_file = f"{file_name}_{date}_{time}_event_times.csv"
        if os.path.exists(file_name):
            header = False
        else:
            header = True
        with open(data_file, 'a') as data_file:
            self.data.to_csv(data_file, header=header)
            print(f"Data saved successfully at {dt.hour}:{dt.minute}.")
        with open(events_file, 'a') as events_file:
            pd.DataFrame(np.asarray(self.events)).to_csv(events_file, header=header)
        with open(time_file, 'a') as time_file:
            pd.DataFrame(np.asarray(self.event_indeces)).to_csv(time_file, header=header)
