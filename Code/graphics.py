import matplotlib
import matplotlib.pyplot as plt
import trial


'''
GRAPHICS
--------
This class describes the graphics of the kinematic interface.

Use this class to control the cursor, target(s), and center(s) on the screen.

'''


class Graphics():

    centers = []
    targets = []
    cursors = []

    def __init__(self, num_cursors):

        # Setting up the figure
        matplotlib.use('Qt5Agg')
        plt.ion()
        fig, ax = plt.subplots()
        ax.set(xlim=(-5, 5), ylim=(-5, 5))
        ax.axis('square')
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        fig.patch.set_facecolor('k')

        # Creating the circle objects
        colors = ['mediumblue', 'red']
        for i in range(num_cursors):
            self.centers.append(plt.Circle((trial.center[i]), radius=0.4))
            self.centers[i].set_facecolor('deeppink')
            ax.add_artist(self.centers[i])
            self.hide(self.centers[i])
            self.targets.append(plt.Circle(trial.center[i], radius=0.4))
            self.cursors.append(plt.Circle(trial.center[i], radius=0.25))
            self.targets[i].set_facecolor('red')
            self.cursors[i].set_facecolor(colors[i])
            ax.add_artist(self.targets[i])
            ax.add_artist(self.cursors[i])
            self.hide(self.targets[i])


    # Method to move the given circle (target or center) to the given position
    def move(self, circ, position):
        circ.center = position[0], position[1]

    # Moving the cursor to the given position
    def move_cursor(self, position):
        self.cursors[0].center = position[0], position[1]

    def show(self, circ):
        circ.set_visible(True)
        
    def hide(self, circ):
        circ.set_visible(False)

    def set_color(self, circ, color):
        circ.set_facecolor(color)