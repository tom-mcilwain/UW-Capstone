import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import trial
from sklearn.metrics import mean_squared_error


'''
ANALYSIS
--------
This script is used to analyze all of the saved data, and includes methods to plot the path, event rate,
event percentage, run times, etc.

'''


def plot_times(events, event_times):

    events = np.asarray(events)[:,1]
    event_times = np.asarray(event_times)[:, 1]

    event_indices = event_times * (events == trial.UPDATED_PARAMS)
    event_indices = event_indices[event_indices != 0]

    times = np.load('times.npy')

    plt.vlines(event_indices, 0, np.max(times), colors='lightskyblue', linewidth=2)
    plt.plot(range(len(times)), times, 'k')#)
    plt.title('Time per loop iteration')
    plt.ylabel('Time (s)')
    plt.xlabel('Samples')
    plt.show()


def plot_event_rates(events, event_times, event_type):

    samples_per_min = trial.freq * 60
    data_length = event_times.iloc[len(events)-1, 1]
    num_bins = int(data_length / samples_per_min)
    event_rates = np.zeros(num_bins)

    for index, row in events.iterrows():
        event = row[1]
        event_time = event_times.iloc[index,1]
        if event == event_type:
            bin = int(event_time / samples_per_min)
            event_rates[bin] += 1
    x = np.linspace(1, num_bins, num_bins)
    plt.plot(x, event_rates, linewidth=2)
    plt.ylabel('Events per min')
    plt.xlabel('Time (mins)')
    plt.show()


def plot_event_percent(data, events, event_times, event_type):

    #data_length = event_times.iloc[len(events), 1]

    window_width = 75 # Number of trials
    num_trials = sum(events <= 7)
    trial_num = 0
    #percentages = np.zeros(math.ceil(num_trials / window_width)
    percentages = np.zeros(num_trials)

    for iter, row in events.iterrows():
        event = row[1]

        if event <= 7:

            trial_num += 1

    if num_trials < window_width:
        percentages[0] = sum(events == event_type) / window_width

    # for i in range(len(percentages)):
    #     start = max(0, i - (window_width / 2))
    #     end = min(i + (window_width / 2), data_length - 1)
    #
    #     event_idx = np.asarray(events)
    #     print(event_idx)
    #     percentages[i] = sum(event_idx)
    #
    # print(percentages)


def plot_state_comparison(data):
    prev_state = np.zeros(2)
    prev_vel = [0, 0]
    state_mses = []
    vel_mses = []
    states = []
    vels = []
    for iter, row in data.iterrows():
        state = np.asarray(row[1:23])
        state = state[10:12]
        vel = np.asarray(row[27:29])
        state_mses.append(mean_squared_error(state, prev_state))
        percent_diff_state = (np.subtract(state_mses[iter], state_mses[iter - 1])) / state_mses[iter - 1] * 100
        states.append(percent_diff_state)
        vel_mses.append(mean_squared_error(vel, prev_vel))
        percent_diff_vel = (np.subtract(vel_mses[iter], vel_mses[iter - 1])) / vel_mses[iter - 1] * 100
        vels.append(percent_diff_vel)

        prev_state = state.copy()
        prev_vel = vel.copy()
    x = range(len(states))
    plt.plot(x, states, label='states')
    plt.figure()
    plt.plot(x, vels, label='vels')
    plt.legend()
    plt.show()


# Plot paths of cursor
def plot_paths(cursor_num, data, events, event_times, show_intention, paths):

    fig, ax = plt.subplots()

    center = plt.Circle((trial.center[cursor_num][0], trial.center[cursor_num][1]), radius=0.4, alpha=0.5,
                        color='m')
    ax.add_artist(center)

    for path in paths:
        current_path = 0
        target_label = 0
        start = 0
        end = 0
        ax.set(xlim=(-5, 5), ylim=(-5, 5))
        ax.set_aspect('equal', adjustable='box')

        for index, row in events.iterrows():
            event = row[1]
            event_time = event_times.iloc[index,1]
            if event <= 7:
                target_label = event
                start = event_time
            elif event == trial.CENTER_REACHED:
                current_path += 1
                end = event_time
                if current_path == path:
                    break

        if path > current_path:
            break

        x = trial.target_positions[target_label][0]
        y = trial.target_positions[target_label][1]
        target = plt.Circle((x, y), radius=0.4, alpha=0.5, color='b')
        ax.add_artist(target)

        positions = data.iloc[start:end, 23:25].to_numpy()

        plt.plot(positions[:,0], positions[:,1])
        if show_intention:
            intentions = data.iloc[start:end, 27:29]
            for iter in range(len(positions)):
                if iter % 3 == 0:
                    intended_position = positions[iter,:] + (intentions.iloc[iter,:]*3)
                    plt.plot([positions[iter,0], intended_position[0]], [positions[iter,1], intended_position[1]])
                    iter += 1

    plt.xticks([])
    plt.yticks([])
    plt.show()


def not_zero(num):
   if num == 0: num = 1
   return num


# Example use:
# date = '5_26_2020'
# time = '21_49'
# trial_num = 1
# num_of_path = 400
# data_file_name = f'subject99_Cursor0_{date}_{time}_data.csv'
# events_file_name = f'subject99_Cursor0_{date}_{time}_events.csv'
# event_times_file_name = f'subject99_Cursor0_{date}_{time}_event_times.csv'
# data = pd.read_csv(data_file_name)
# events = pd.read_csv(events_file_name)
# event_times = pd.read_csv(event_times_file_name)
