import numpy as np
import math
import random
import graphics
from time import sleep
from datetime import datetime
import time
import matplotlib.pyplot as plt


'''
TEST_TRIAL
-----
This script is similar to trial.py, the backbone behind the entire kinematic interface.
The difference is that this script uses pre-existing glove movements rather than real-time data.

This script handles everything to do with running the session as well as the calibration session.
This script uses the classes created (cursor, kalman, and graphics) to run through a session.

'''


# Defining distances and positions of the workspace
dist = 4/math.sqrt(2)
center = [[0, 0], [0, 0]]
target_positions = [[0, 4],
                    [dist, dist],
                    [4, 0],
                    [dist, -dist],
                    [0, -4],
                    [-dist, -dist],
                    [-4, 0],
                    [-dist, dist]]


# Event codes
# Target labels range from 0 - 7
TARGET_REACHED = 11
TIME_EXPIRED = 12
CENTER_REACHED = 13
SESSION_OVER = 14
UPDATED_PARAMS = 15

# Defining values of the trial
time_start = datetime.now().microsecond
freq = 32
loop_duration = 1.0 / freq
trial_duration = 5
calibration_duration = 1

MAX_DISTANCE = 0.4 # Distance the cursor has to be away from the target/center to count
HOLD_TIME = 0.35 # How long the cursor has to be at the target/center
NUM_SENSORS = 22 # Number of sensors of the glove
NUM_TARGETS = 8
STATE_INDEX = 23 #      For
VELOCITY_INDEX = 25 #   DataFrame
INTENDED_INDEX = 27 #   Indexing

# Values for standardizing
means = np.empty(0)
stds = np.empty(0)
standardized = False

to_center = []

trial_iter = None  # Number of iterations in the trial so far


# Run through a trial with a model. save_data = True if you want to save data to a csv file
def run(glove_data, cursors, save_data, session_length, dimensions):

    screen = graphics.Graphics(num_cursors=len(cursors))

    # Initializing counters
    global trial_iter
    trial_iter = np.zeros(len(cursors))  # Number of samples in the trial so far
    total_count = np.zeros(len(cursors)) # Number of total trials
    success = np.zeros(len(cursors)) # Number of successful trials
    state_iter = 0 # Number of total iterations in the session
    batch_iter = 0 # Number of iterations in batch so far
    batch_start = 0 # Index where the next batch update begins

    target_reached = []
    for i in range(len(cursors)):
        target_reached.append(True)
    pos_count = np.zeros(len(cursors))

    plt.pause(0.0025)
    countdown()
    prev = time.time()

    times = []

    # Using saved data instead of glove
    for index in range(glove_data.shape[0]):
        current_state = glove_data.iloc[index, 1:STATE_INDEX]

        # If the training data is standardized, do the same to the glove state
        if standardized:
            global diff
            diff = np.subtract(current_state, means)
            current_state = np.divide(diff, stds)

        # Perform an iteration for each cursor
        velocities = []
        for i in range(len(cursors)):

            cursor = cursors[i]

            # If this is the beginning of the session...
            if state_iter == 0:
                get_target(cursor, dimensions)
                show_target(screen, cursor)
                cursor.events.append(cursor.target_label)
                cursor.event_indeces.append(state_iter)
                total_count[i] += 1
                print(f"Trial #{int(total_count[i])} for Cursor #{cursor.cursor_num + 1}")

            # Perform iteration
            pos, vel = cursor.iterate(current_state)
            screen.move(screen.cursors[cursor.cursor_num], pos)
            velocities = np.append(velocities, vel)

            # Perform a batch update when the time has reached the batch length
            if cursor.model.algorithm != 'adaptive':
                if batch_iter // freq >= cursor.model.batch_length:
                    cursor.batch_update(batch_start)
                    cursor.events.append(UPDATED_PARAMS)
                    cursor.event_indeces.append(state_iter)
                    batch_start = state_iter
                    batch_iter = 0

            # If reach time has expired, remove target and show center
            if trial_iter[i] > trial_duration * freq and not to_center[i]:
                screen.hide(screen.targets[cursor.cursor_num])
                show_center(screen, cursor)
                to_center[i] = True
                target_reached[i] = False
                cursor.events.append(TIME_EXPIRED)
                cursor.event_indeces.append(state_iter)
                print("FAIL")

            # else, check if the cursor is within proximity of the target/center
            else:

                trial_iter[i] += 1

                # If the cursor reaches the center to initialize a new trial...
                colors = ['lime', 'lime']
                if within_proximity(pos, center[i]) and to_center[i]:
                    pos_count[i] += 1
                    screen.set_color(screen.centers[i], 'pink')
                    if pos_count[i] / freq >= HOLD_TIME:
                        screen.hide(screen.centers[i])
                        to_center[i] = False
                        get_target(cursor, dimensions)
                        show_target(screen, cursor)
                        cursor.events.append(CENTER_REACHED)
                        cursor.event_indeces.append(state_iter)
                        cursor.events.append(cursor.target_label)
                        cursor.event_indeces.append(state_iter)
                        total_count[i] += 1
                        print(f"Trial #{int(total_count[i])} for Cursor #{cursor.cursor_num + 1}")
                        pos_count[i] = 0
                        trial_iter[i] = 0

                # If the cursor reaches the target in time...
                elif within_proximity(pos, cursor.target_loc) and not to_center[i]:
                    pos_count[i] += 1
                    screen.set_color(screen.targets[cursor.cursor_num], colors[i])
                    if pos_count[i] / freq >= HOLD_TIME:
                        screen.hide(screen.targets[cursor.cursor_num])
                        show_center(screen, cursor)
                        to_center[i] = True
                        target_reached[i] = True
                        success[i] += 1
                        pos_count[i] = 0
                        cursor.events.append(TARGET_REACHED)
                        cursor.event_indeces.append(state_iter)
                        print("SUCCESS")
                else:
                    colors = ['lightskyblue', 'salmon']
                    screen.set_color(screen.targets[cursor.cursor_num], colors[i])
                    screen.set_color(screen.centers[cursor.cursor_num], 'deeppink')

            # Check if session time is up
            if state_iter / freq >= session_length:
                cursor.events.append(SESSION_OVER)
                cursor.event_indeces.append(state_iter)

        # Calculate how much time to sleep
        now = time.time()
        current_time_diff = now - prev
        pause_time = loop_duration - current_time_diff
        if pause_time < 0:
            pause_time = 1e-10
        plt.pause(pause_time)
        temp = time.time()
        times.append(temp - prev)
        prev = temp

        # Check if time is up
        if state_iter / freq >= session_length:
            break
        else:
            state_iter += 1
            batch_iter += 1

    # Saving data to file
    if save_data:
        for cursor in cursors:
            cursor.save_data()


# Used to generate calibration data
def calibrate(stream, cursors, calibration_length):
    screen = graphics.Graphics(num_cursors=1)
    cursor = cursors[0]
    cursor.target_label = 0
    trial_count = 0

    global trial_iter
    trial_iter = calibration_duration * freq
    global to_center
    to_center[0] = True

    countdown()
    prev = time.time()

    for state in stream:

        # Checking if time is up
        # If so, the cursor has reached either the target or the center
        if trial_iter >= calibration_duration * freq:
            if to_center[0]:
                screen.hide(screen.centers[0])
                to_center[0] = False
                if trial_count + 1 > calibration_length:
                    print("Calibration complete.")
                    break
                get_target(cursor, 2)
                show_target(screen, cursor)
                trial_count += 1
                print(f"Trial #{trial_count}")
            else:
                screen.hide(screen.targets[cursor.cursor_num])
                show_center(screen, cursor)
                to_center[0] = True
            trial_iter = 0

        # Perform a cursor iteration and move the cursor
        pos, vel = cursor.iterate(state)
        screen.move(screen.cursors[0], pos)

        # Controlling loop timing
        now = time.time()
        current_time_diff = now - prev
        pause_time = loop_duration - current_time_diff
        if pause_time < 0:
            print("long")
            pause_time = 10e-10
        plt.pause(pause_time)
        prev = time.time()
        plt.pause(pause_time)
        trial_iter += 1


# Show the target of the cursor on the screen
def show_target(screen, cursor):
    colors = ['lightskyblue', 'salmon']
    screen.set_color(screen.targets[cursor.cursor_num], colors[cursor.cursor_num])
    screen.move(screen.targets[cursor.cursor_num], cursor.target_loc)
    screen.show(screen.targets[cursor.cursor_num])


# Show the center circle of the screen
def show_center(screen, cursor):
    screen.move(screen.centers[cursor.cursor_num], center[cursor.cursor_num])
    screen.show(screen.centers[cursor.cursor_num])


# Generate a random target for the cursor passed in
def get_target(cursor, dimensions):
    index = np.random.randint(0, NUM_TARGETS)
    cursor.target_label = index
    constant = 0
    if dimensions == 1:
        index = int((NUM_TARGETS / 2) * random.randint(0, 1))
        constant = dist * ((cursor.cursor_num * 2) - 1)
    cursor.target_loc = [target_positions[index][0] + constant, target_positions[index][1]]


# Move cursor to center
def center_cursor(cursor):
    x_vel = center[cursor.cursor_num][0] - cursor.position[0]
    y_vel = center[cursor.cursor_num][1] - cursor.position[1]
    vel = np.append(x_vel, y_vel)
    cursor.center()
    return vel


# Check if the cursor is within proximity of the target or center
def within_proximity(cursor_pos, target_pos):
    x_res = cursor_pos[0] - target_pos[0]
    y_res = cursor_pos[1] - target_pos[1]
    dist = math.sqrt(x_res**2 + y_res**2)
    if dist < MAX_DISTANCE:
        return True
    else:
        return False


# Standardizing given data
def standardize(data):
    global means
    global stds
    global standardized

    standardized = True
    standardized_data = data.copy()

    means = standardized_data.iloc[:,1:STATE_INDEX].mean(axis=0)
    stds = standardized_data.iloc[:,1:STATE_INDEX].std(axis=0)
    stds[stds == 0] = 1e-6
    standardized_data.iloc[:,1:STATE_INDEX] = standardized_data.iloc[:,1:STATE_INDEX].subtract(means).divide(stds)

    return standardized_data


# Counting down for the start of the session
def countdown():
    print("Session begins in:")
    for i in range(3):
        sleep(1)
        print(3-i)
    sleep(1)
    print("Go!")
