import numpy as np
from sommet import Sommet


class Arbre:
    #stockage en keyword tree mais au moins utilise un numpy optimisé

    def __init__(self, position):
        self.list = np.full(256, None)
        self.list[0] = Sommet(position, [0, 0], 0, 0, None) #on crée la racine
        self.size = 256
        self.indice_act = 1
        self.start_current_depth = 0
        self.compteur = 0
        return

    def get_sommet(self, indice):
        return self.list[indice]

    def create_sommet(self, position, speed, distance, depth, parent):
        if self.indice_act < self.size:
            self.list[self.indice_act] = Sommet(position, speed, distance, depth, parent)
            self.indice_act += 1
        else:
            self.list = np.concatenate((self.list, np.zeros(self.size)), axis=None)
            self.size = self.size*2
            self.list[self.indice_act] = Sommet(position, speed, distance, depth, parent)
            self.indice_act += 1
        return

    def create_sommet_unique(self, position, speed, distance, depth, parent):
        #on check d'abord si ledit sommet n'existe pas déjà
        #pour cela, on cherche dans ceux de même depth
        if depth > self.start_current_depth:
            self.start_current_depth = self.indice_act

        indice = self.start_current_depth


        while indice < self.indice_act and self.list[indice].is_same(position, speed) is not True:
            indice += 1

        if indice < self.indice_act: #alors on a un qui est True
            if self.list[indice].get_distance() <= distance: #aucun intéret à créer un nouveau sommet
                return 0
            else:
                self.list[indice].set_data(distance, parent) #on remplace l'ancien sommet par le nouveau
                return 0

        else:
            if self.indice_act < self.size:
                self.list[self.indice_act] = Sommet(position, speed, distance, depth, parent)
                self.indice_act += 1
            else:
                self.list = np.concatenate((self.list, np.zeros(self.size)), axis=None)
                self.size = self.size*2
                self.list[self.indice_act] = Sommet(position, speed, distance, depth, parent)
                self.indice_act += 1
            return 1

    def get_indice_act(self):
        return self.indice_act

    def ret_chemin(self, indice):
        node = self.get_sommet(indice)
        dep = node.get_depth()+1
        res = np.full(dep, None)
        for i in range(dep):
            res[dep-i-1] = node.get_position()
            node = self.list[node.get_parent()]
        return res


    def create_sommet_unique_bis(self, position, speed, distance, depth, parent):

        if self.indice_act >= self.size:
            self.list = np.concatenate((self.list, np.zeros(self.size)), axis=None)
            self.size = self.size * 2

        compteur = 0
        while compteur < self.indice_act and self.list[compteur].is_same(position, speed) == False:
            compteur += 1

        if compteur < self.indice_act: #on a trouvé le meme avant
            if self.list[compteur].get_depth() >= depth:
                if self.list[compteur].get_distance() >= distance:
                    #on fait un test bas de gamme:
                    #on regarde tous les enfants, et on remplace toutes les locd,,,,d,d
                    self.list[compteur].set_data(distance, depth, parent)
                    return 0
            #else: #on ne fait rien, on rajoutera le notre

        self.list[self.indice_act] = Sommet(position, speed, distance, depth, parent)
        self.indice_act += 1

        return 1
