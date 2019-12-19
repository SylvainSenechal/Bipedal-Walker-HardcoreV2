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
    theta = radar_factory(N, frame='circle')

    spoke_labels = data.pop(0)
    title, case_data = data[0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    # ax.set_title(title,  position=(0.5, 1.1), ha='center')

    for id, d in enumerate(case_data):
        if id == 0:
            line = ax.plot(theta, d, color='r', label="POET")
        elif id == 1:
            line = ax.plot(theta, d, color='b', label='CURRICULUM')
        else:
            line = ax.plot(theta, d, color='b')
        ax.plot(theta, d,  alpha=0.0)
    ax.legend()
    ax.set_varlabels(spoke_labels)

    # legend = ax.legend(labels, loc=(0.9, .95),
    #                    labelspacing=0.1, fontsize='small')
    plt.title('Maximum difficulty solved, POET vs CURRICULUM, poet pair n°' + str(nbAgent))
    plt.show()

def plotDifficultyReached():
    curriculumDiff1, curriculumDiff2 = [], []
    with open("savedAgent/difficultyLastCURRICULUM_V4_.txt") as file:
        for difficulty in file:
            curriculumDiff1.append(float(difficulty))
        curriculumDiff1.append(1.0) # ROUGHNESS
        curriculumDiff1.append(0.0) # GAP_WIDTH
    with open("savedAgent/difficultyLastCURRICULUM_V5_.txt") as file:
        for difficulty in file:
            curriculumDiff2.append(float(difficulty))
        curriculumDiff2.append(1.0) # ROUGHNESS
        curriculumDiff2.append(0.0) # GAP_WIDTH
    for i in range(5):
        difficulties = []
        with open("savedAgent/difficultyLastPOET_V3_" + str(i) + ".txt", "r") as file:
            for difficulty in file:
                difficulties.append(float(difficulty.strip()))
        difficulties.append(1.0) # ROUGHNESS
        difficulties.append(0.0) # GAP_WIDTH

        data = [['STAIRS', 'STUMP', 'HEIGHT', 'ROUGHNESS', 'GAP_WIDTH'],
                ('Basecase', [
                    difficulties,
                    curriculumDiff1,
                    curriculumDiff2
                    ])]
        plotSpider(data, i)

def plotListDifficulties():
    listIterations, listStairs, listStumps, listHeight, listSumDifficulties = [], [], [], [], []
    iterations, stairs, stumps, heights, SumDifficulties = [], [], [], [], []
    cursor = 0
    sum = 0
    with open("savedAgent/listDifficulty_V2.txt", "r") as file:
    # with open("savedAgent/listDifficultyCURRICULUM_V5.txt", "r") as file:
        for info in file:
            cursor += 1
            if info.strip() == '#':
                listIterations.append(iterations)
                listStairs.append(stairs)
                listStumps.append(stumps)
                listHeight.append(heights)
                listSumDifficulties.append(SumDifficulties)
                iterations, stairs, stumps, heights, SumDifficulties = [], [], [], [], []
                cursor = 0
                sum = 0
            elif cursor == 1:
                stairs.append(float(info.strip()))
                sum += float(info.strip())
            elif cursor == 2:
                stumps.append(float(info.strip()))
                sum += float(info.strip())
            elif cursor == 3:
                heights.append(float(info.strip()))
                sum += float(info.strip())
                SumDifficulties.append(sum/3)
                print(sum)
                print("##")
                sum = 0
            elif cursor == 4:
                iterations.append(float(info.strip()))
                cursor = 0
    plt.ylim(0, 1)

    for agent in range(len(listIterations)):
        plt.title('Pairs generated and solved by POET')
        plt.xlabel('POET iteration')
        plt.ylabel('STAIRS difficulty solved')
        plt.plot(listIterations[agent], listStairs[agent])
    plt.show()
    for agent in range(len(listIterations)):
        plt.title('Pairs generated and solved by POET')
        plt.xlabel('POET iteration')
        plt.ylabel('STUMPS difficulty solved')
        plt.plot(listIterations[agent], listStumps[agent])
    plt.show()
    for agent in range(len(listIterations)):
        plt.title('Pairs generated and solved by POET')
        plt.xlabel('POET iteration')
        plt.ylabel('HEIGHT difficulty solved')
        plt.plot(listIterations[agent], listHeight[agent])
    plt.show()
    for agent in range(len(listIterations)):
        plt.title('Pairs generated and solved by POET')
        plt.xlabel('POET iteration')
        plt.ylabel('MEAN difficulty solved')
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
