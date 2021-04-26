import numpy as np


class Pile:

    def __init__(self, queue, indice_start, indice_end, len):
        self.queue = queue
        self.indice_start = indice_start
        self.indice_end = indice_end
        self.len = len

    def __init__(self):
        self.queue = np.zeros(256)
        self.indice_start = -1
        self.indice_end = 0
        self.len = 256

    def popleft(self):
        self.indice_start += 1
        return self.queue[self.indice_start - 1]

    def append(self, valeur):
        if self.indice_end < self.len - 1:
            self.indice_end += 1
            self.queue[self.indice_end] = valeur

        else:
            self.queue = np.concatenate((self.queue, np.zeros(2*self.len)), axis=None)
            self.len = 3 * self.len
            self.indice_end += 1
            self.queue[self.indice_end] = valeur
            print("######### ON MULTIPLIE PAR TROIS LA TAILLE DE LA PILE ##########", self.len)

    def size(self):
        return self.indice_end - self.indice_start

    def start(self):
        return self.indice_start


