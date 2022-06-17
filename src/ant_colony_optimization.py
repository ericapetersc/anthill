import random as rd
from concurrent import futures
from turtle import pos
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

COLORS_MAP = {
    0: "b",
    1: "r",
    2: "y",
    3: "g",
    4: "m",
    5: "c",
}


class Datum:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def get_label(self):
        return self.label

    def reduce_dimensionality(self):
        return np.mean(self.data)

    def calculate_distance(self, datum):
        return np.linalg.norm(self.data - datum.data)


class Grid:
    def __init__(self, n_rows, n_columns, objects):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.objects = objects
        self.grid = self.create_grid()
        self.populate_grid()

    def get_grid(self):
        return self.grid

    def create_grid(self):
        return np.empty((self.n_rows, self.n_columns), dtype=object)

    def is_empty_cell(self, x, y):
        return self.grid[x][y] is None

    def is_valid_cell(self, x, y):
        return 0 < x <= self.n_rows - 1 and 0 < y <= self.n_columns - 1

    def empty_cell(self, x, y):
        self.grid[x][y] = None

    def fill_cell(self, x, y, obj):
        self.grid[x][y] = obj

    def get_coordinates(self):
        return [(x, y) for x in range(self.n_rows) for y in range(self.n_columns)]

    def get_objects_coordinates(self):
        coordinates = []
        for x in range(self.n_rows):
            for y in range(self.n_columns):
                if self.grid[x][y] is not None:
                    coordinates.append((x, y))
        return coordinates

    def get_objects_labels(self):
        labels = []
        for x in range(self.n_rows):
            for y in range(self.n_columns):
                if self.grid[x][y] is not None:
                    l = self.grid[x][y].get_label()
                    labels.append(l)
        return labels

    def get_neighbours(self, r, x, y):

        objs = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if (
                    (i != 0 or j != 0)
                    and self.is_valid_cell(x + i, y + j)
                    and not self.is_empty_cell(x + i, y + j)
                ):
                    objs.append(self.grid[x + i][y + j])
        return objs

    def get_neighbours_similarity(self, x, y, r, alpha, data=None):
        neighbours = self.get_neighbours(r, x, y)
        data = self.grid[x][y] if not data else data
        sum_ = 0
        for n in neighbours:
            sum_ += 1 - data.calculate_distance(n) / alpha
        return max(0, 1 / pow((r * 2) + 1, 2) * sum_)

    def populate_grid(self):
        coordinates = self.get_coordinates()
        rd.shuffle(coordinates)
        for i, obj in enumerate(self.objects):
            x, y = coordinates[i]
            self.grid[x][y] = obj


class Ant:
    def __init__(self, **kwargs):
        self.pos = kwargs["pos"]
        self.grid = kwargs["grid"]
        self.vision_ratio = kwargs["vision_ratio"]
        self.kp = kwargs["kp"]
        self.kd = kwargs["kd"]
        self.alpha = kwargs["alpha"]
        self.number_movements = 0
        self.item = None
        self.pick()

    def get_position(self):
        return self.pos

    def is_carrying_item(self):
        return self.item is not None

    def move(self):
        while True:
            x = rd.randint(-1, 1) + self.pos[0]
            y = rd.randint(-1, 1) + self.pos[1]
            if self.grid.is_valid_cell(x, y):
                break
        self.pos = (x, y)
        self.number_movements += 1

    def pick(self):
        self.item = self.grid.get_grid()[self.pos[0]][self.pos[1]]
        self.grid.empty_cell(self.pos[0], self.pos[1])

    def drop(self):
        self.grid.fill_cell(self.pos[0], self.pos[1], self.item)
        self.item = None

    def pick_proba(self):
        f = self.grid.get_neighbours_similarity(
            self.pos[0], self.pos[1], self.vision_ratio, self.alpha
        )
        return pow(self.kp / (self.kp + f), 2) > 0.5

    def drop_proba(self):
        f = self.grid.get_neighbours_similarity(
            self.pos[0], self.pos[1], self.vision_ratio, self.alpha, self.item
        )
        if f < self.kd:
            return (2 * f) > 0.5
        return True

    def behaviour(self):
        self.move()
        if self.is_carrying_item():
            if self.grid.is_empty_cell(self.pos[0], self.pos[1]) and self.drop_proba():
                self.drop()
        else:
            if (
                not self.grid.is_empty_cell(self.pos[0], self.pos[1])
                and self.pick_proba()
            ):
                self.pick()


class AntColonyOptimization:
    def __init__(self, **kwargs):
        self.data = kwargs["data"]
        self.n_rows = kwargs["rows"]
        self.n_columns = kwargs["columns"]
        self.n_ants = kwargs["ants"]
        self.vision_ratio = kwargs["ratio"]
        self.kp = kwargs["kp"]
        self.kd = kwargs["kd"]
        self.alpha = kwargs["alpha"]
        self.max_iterations = kwargs["iterations"]
        self.grid = Grid(self.n_rows, self.n_columns, self.data)
        self.ants = self.create_ants()

    def create_ants(self):
        ants = []
        coordinates = self.grid.get_objects_coordinates()
        rd.shuffle(coordinates)
        for i in range(self.n_ants):
            ants.append(
                Ant(
                    **{
                        "pos": coordinates[i],
                        "grid": self.grid,
                        "vision_ratio": self.vision_ratio,
                        "kp": self.kp,
                        "kd": self.kd,
                        "alpha": self.alpha,
                    }
                )
            )
        return ants

    def generator_function(self):
        for i in range(self.max_iterations):
            with futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_executer = {
                    executor.submit(
                        a.behaviour,
                    ): a
                    for a in self.ants
                }
            for future in futures.as_completed(future_executer):
                _ = future.result()

            objs_coordinates = self.grid.get_objects_coordinates()
            objs_labels = self.grid.get_objects_labels()
            ants_coordinates = [a.get_position() for a in self.ants]

            if i % 200 == 0:
                yield (objs_coordinates, objs_labels, ants_coordinates)

    def animate(self, info):
        plt.clf()
        plt.xlim([0, self.n_columns])
        plt.ylim([0, self.n_rows])

        objs_coordinates, objs_labels, ants_coordinates = info
        xo, yo = zip(*objs_coordinates)
        xa, ya = zip(*ants_coordinates)

        plt.scatter(
            xo, yo, c=list(np.vectorize(COLORS_MAP.get)(objs_labels)), marker="x"
        )
        plt.scatter(
            xa,
            ya,
            c="k",
            marker="o",
        )

    def execute(self):
        self.fig, self.ax = plt.subplots()

        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=self.generator_function,
            interval=100,
            repeat=False,
        )

        plt.show()
