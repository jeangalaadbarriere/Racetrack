# on définit un obejet pour représenter une configuration possible pour arriver à l'instant actuel


class Configuration:
    # deux champs: move représente le dernier tour; path est une liste de tous les moves utilisés pour arriver là
    # dist permet de définir la distance parcourue pour rajouter des critères d'optimalité
    def __init__(self, move, path, dist):
        self.move = move
        self.chemin = path
        self.distance = dist

    # Une fonction pour afficher les chemins parcourus
    def affiche(self):
        print("Chemin : ")
        n = len(self.chemin)
        for k in range(n):
            M = self.chemin[k]
            print("Tour ", k, " : déplacement ", M.deplacement, "à la position ", M.position)
        print("Distance = ", + self.distance)
