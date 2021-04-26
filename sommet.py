import numpy as np

class Sommet:

    def __init__(self, position, speed, distance, depth, parent):
        self.position = position
        self.speed = speed
        self.distance = distance
        self.depth = depth
        self.parent = parent
        return

    def is_root(self):
        return self.parent is None

    def get_depth(self):
        return self.depth

    def get_position(self):
        return self.position

    def get_speed(self):
        return self.speed

    def get_distance(self):
        return self.distance

    def get_parent(self):
        return self.parent

    def set_data(self, distance, depth, parent):
        self.distance = distance
        self.depth = depth
        self.parent = parent
        return

    def is_same(self, position, speed):
        return self.position[0] == position[0] and self.position[1] == position[1] and self.speed[0] == speed[0] and self.speed[1] == speed[1]
