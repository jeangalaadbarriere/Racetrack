#on définit une classe pour les grilles
#dans le champ grille, un 1 est dans le circuit, 0 est hors-limite
#arrivée est représenté par les deux points extrêmes du segment
import numpy as np

class Grille:
    def __init__(self, n, p, depart, arrivee):
        self.grid = np.array((n,p))
        self.depart = depart
        self.arrivee = arrivee