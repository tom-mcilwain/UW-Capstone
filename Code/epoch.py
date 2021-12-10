import connect as ct
import trial
from cursor import Cursor
from kalman import KalmanFilter


'''
EPOCH
-----
This is the main script used to control all the settings and run the sessions.

In this script, you can choose session options such as whether or not you are calibrating, whether
to use shuffled data, whether to save the data, how many cursors, how many dimensions, etc.

'''


# Calibration settings
calibrating = False
calibration_length = 150  # Number of trials when calibrating

# Model settings
use_shuffled_data = True
shuffle = False  # True if randomizing data
algorithm = 'adaptive'  # Model algorithm to use (batch, smooth_batch, or adaptive)

# Trial settings
save_data = False
session_length = 60 * 30  # Length of a session in seconds
num_cursors = 1  # Number of cursors (1 for low dimensions, 2 for high dimensions)
num_dimensions = 2  # Number of dimensions the cursor(s) moves in
subject_id = 99  # ID of Subject

# Initializing set of cursors
cursors = []

# Opening the glove
port = '/dev/tty.usbserial'  # This differs between OS X and Windows
glove = ct.open_glove(port, baudrate=9600)
stream = glove.stream_sensors()

# Creating cursor(s)
for i in range(num_cursors):

    # Creating the model for the cursor
    model = KalmanFilter(algorithm=algorithm)
    if not calibrating:
        file_name = f"subject{subject_id}_Cursor{str(i)}_calibration_data.csv"
        if shuffle:
            model.randomize(file_name)
        if use_shuffled_data:
            file_name = f"shuffled_{file_name}"
        model.initialize(file_name)

    if num_dimensions == 1 and not calibrating:
        x = trial.dist * ((i * 2) - 1)
        trial.center[i] = [x, 0]

    # Creating the cursor
    cursor = Cursor(model, calibrating, num_dimensions, i, subject_id)
    cursors.append(cursor)
    trial.to_center.append(False)


# Running the session
try:
    if calibrating:
        trial.calibrate(stream, cursors, calibration_length)
    else:
        trial.run(stream, cursors, save_data, session_length, num_dimensions)
finally:
    glove.close()
