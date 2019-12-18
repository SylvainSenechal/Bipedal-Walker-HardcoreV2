### FROM : https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

# PARAMETER data should be like this :
# data = [['STAIRS', 'STUMP', 'HEIGHT'],
#         ('Basecase', [
#             [0.2, 0.2, 0.8],
#             [0.4, 0.0, 0.5],
#             [0.6, 0.4, 0.9]])]
def plotSpider(data, nbAgent):
    N = len(data[0])
    print(data)
    print(N)
    theta = radar_factory(N, frame='circle')

    spoke_labels = data.pop(0)
    title, case_data = data[0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    # ax.set_title(title,  position=(0.5, 1.1), ha='center')

    for d in case_data:
        line = ax.plot(theta, d)
        ax.plot(theta, d,  alpha=0.0)
    ax.set_varlabels(spoke_labels)
    # labels = ('direct', 'oui')
    # legend = ax.legend(labels, loc=(0.9, .95),
    #                    labelspacing=0.1, fontsize='small')
    plt.title('Maximum difficulty solved poet pair n°' + str(nbAgent))
    plt.show()

def plotDifficultyReached():
    for i in range(5):
        difficulties = []
        with open("savedAgent/difficultyLastPOET_V2_" + str(i) + ".txt", "r") as file:
            for difficulty in file:
                difficulties.append(float(difficulty.strip()))
        difficulties.append(1.0) # ROUGHNESS
        difficulties.append(0.0) # GAP_WIDTH
        data = [['STAIRS', 'STUMP', 'HEIGHT', 'ROUGHNESS', 'GAP_WIDTH'],
                ('Basecase', [
                    difficulties,
                    ])]
        plotSpider(data, i)

def plotListDifficulties():
    listIterations, listStairs, listStumps, listHeight, listSumDifficulties = [], [], [], [], []
    iterations, stairs, stumps, heights, SumDifficulties = [], [], [], [], []
    cursor = 0
    sum = 0
    with open("savedAgent/listDifficulty_V2.txt", "r") as file:
        for info in file:
            cursor += 1
            if info.strip() == '#':
                listIterations.append(iterations)
                listStairs.append(stairs)
                listStumps.append(stumps)
                listHeight.append(heights)
                listSumDifficulties.append(SumDifficulties)
                iterations = []
                stairs = []
                stumps = []
                heights = []
                SumDifficulties = []
                cursor = 0
                sum = 0
            elif cursor == 1:
                stairs.append(info.strip())
                sum += float(info.strip())
            elif cursor == 2:
                stumps.append(info.strip())
                sum += float(info.strip())
            elif cursor == 3:
                heights.append(info.strip())
                sum += float(info.strip())
                SumDifficulties.append(sum)
                print(sum)
                print("##")
                sum = 0
            elif cursor == 4:
                iterations.append(info.strip())
                cursor = 0
    print(listStairs[10])
    print(listStumps[10])
    print(listHeight[10])
    print(listSumDifficulties[10])
    print(len(listSumDifficulties))
    print(len(listStairs))
    for agent in range(len(listIterations)):
        plt.plot(listIterations[agent], listStairs[agent])
    plt.show()
    for agent in range(len(listIterations)):
        plt.plot(listIterations[agent], listStumps[agent])
    plt.show()
    for agent in range(len(listIterations)):
        plt.plot(listIterations[agent], listHeight[agent])
    plt.show()
    for agent in range(len(listIterations)):
        plt.plot(listIterations[agent], listSumDifficulties[agent])
    plt.show()
def plotScoresOverDifficulty(scoresSTAIRS, difficultiesSTAIRS, scoresSTUMP, difficultiesSTUMP, scoresHEIGHT, difficultiesHEIGHT):

    plt.plot(difficultiesSTAIRS, scoresSTAIRS, '-o',label='STAIRS')
    plt.plot(difficultiesSTUMP, scoresSTUMP, '-o',label='STUMP')
    plt.plot(difficultiesHEIGHT, scoresHEIGHT, '-o',label='HEIGHT')
    plt.ylim(-130, 380) # Worst and best score possible
    plt.legend(loc='upper right')
    plt.xlabel('Difficulty')
    plt.ylabel('Score')
    plt.title('Agent Score over environment difficulty, direct learning')
    plt.show()
