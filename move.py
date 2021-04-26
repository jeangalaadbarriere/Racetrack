# On définit une classe pour les mouvements des voitures
import numpy as np

class Move:
    # deux champs: une position, et un vecteur déplacement représentant le dernier tour
    # une position est un vecteur des coordonnées dans la grille
    def __init__(self, position, deplacement):
        self.position = position
        self.deplacement = deplacement

    # on définit une fonction pour déterminer si le déplacement intersecte un segment entre A et B
    # on considère que le segment est soit vertical, soit horizontal
    def intersect(self, A, B):
        '''
        Parameters
        ----------
        A : point, de coordonnées (x_A, y_A)
        B : point, de coordonnées (x_b, y_b)

        Returns
        -------
        None.

        '''
        A2 = toCoord(A)
        B2 = toCoord(B)
        M_pos, M_dep = toCoord(self.position), toCoord(self.deplacement)
        x, y = M_pos[0,0], M_pos[1,0]
        u, v = M_dep[0,0], M_dep[1,0]

        x_A, y_A = A2[0,0], A2[1,0]
        x_B, y_B = B2[0,0], B2[1,0]

        #on le met au début car c'est le cas le plus probable donc on diminue la complexité en moyenne
        if (x_A == x and y_A == y) or (x_B == x and y_B == y):
            return True

        if (x_A == x_B):  #s'il s'agit d'un segment vertical
            if u == 0:
                if min(y_A,y_B) <= y and y <= max(y_A, y_B) and x == x_A:
                    return True
                return False
            elif ((x_A <= x and x - u < x_A) or (x_A >= x and x - u > x_A)) and (min(y_A, y_B) <= y - v*(x - x_A)/u  and y - v*(x - x_A)/u <= max(y_A, y_B)) :
                #ATTENTION: LES CONDITIONS SERVENT A INDIQUER SI LE DEPLACEMENT FAIT TRAVERSER LA FRONTIERE, C EST DONC DES INEGALITES INVERSEES
                #IL FAUT DONC COMPARER LA POSITION INITIALE X-U QUI DOIT DONC ETRE DE L AUTRE COTE DE XA
                return True
            return False


        #il s'agit donc d'un segment horizontal
        if v == 0:
            if min(x_A, x_B) <= x and x <= max(x_A, x_B) and y == y_B:
                return True
            return False
        elif ((y_A <= y and y - v < y_A) or (y_A >= y and y - v > y_A)) and (min(x_A, x_B) <= (x - u*(y - y_A)/v) and (x - u*(y - y_A)/v) <= max(x_A, x_B)) :
            return True
        return False

    #une fonction pour vérifier que le mouvement reste dans les bornes du circuit
    def estValable(self, grille):

        '''

        Parameters
        ----------
        grille :  Grille

        Returns
        -------
        true or false :  boolean
            indique si le déplacement respecte les règles imposées par le jeu

        '''
        #on considère que la position actuelle est valable, sans quoi nous ne pourrions y être
        #ATTENTION NONONONONONONON LA POSITION DE L ITERATION EST VALABLE ET IL FAUT JUSTEMENT VERIFIER QUE LA NOUVELLE VA BIEN
        #LA POSITION QUI VA BIEN EST DONC X-A ET Y-B
        #x, y = self.position[0,0], self.position[1,0]
        M_pos, M_dep = toCoord(self.position), toCoord(self.deplacement)
        x, y = M_pos[0,0], M_pos[1,0]
        a, b = M_dep[0,0], M_dep[1,0]
        sign_a = int(np.sign(a))
        if(sign_a == 0):
            sign_a = 1
        sign_b = int(np.sign(b))
        if(sign_b ==0):
            sign_b = 1

        #On peut facilement éliminer la ]position (x,y) si elle n'est pas dans la grille
        if x >= np.shape(grille.grid)[0] or y >= np.shape(grille.grid)[1] or x < 0 or y < 0:
            return False

        #évidemment, on élimine aussi la position si on tombe sur une case interdite
        if grille.grid[int(x)][int(y)] == 0:
            return False

        #on test le cas particulier d'un vecteur vertical/horizontal et on peut simplifier directement:
        if a == 0:
            for j in range(int(y-b), int(y), sign_b):
                if grille.grid[int(x)][j] == 0:
                    return False
            return True

        if b == 0:
            for i in range(int(x-a), int(x), sign_a):
                if grille.grid[i][int(y)] == 0:
                    return False
            return True


        # On test maintenant si le dernier vecteur intercept un trajet interdit, càd reliant deux positions interdites
        for i in range(int(x-a), int(x), sign_a):
            for j in range(int(y-b), int(y), sign_b):
                #inutile de tester si cette position intermédiare existe, c'est assuré avec le test plus haut de (x,y)
                if grille.grid[i][j] == 0:
                    if grille.grid[i+sign_a][j] == 0:
                        if self.intersect(np.array([[i],[j]]), np.array([[i+sign_a],[j]])):
                            return False
                        return True
                    if grille.grid[i][j+sign_b] == 0:
                        if self.intersect(np.array([[i],[j]]), np.array([[i],[j+sign_b]])):
                            return False
                        return True
        return True

    #détermine si la trajectoire a dépassé la ligne d'arrivée
    def Goal(self, grille):
        return self.intersect(grille.arrivee[0], grille.arrivee[1])

def toCoord(A):
    x, y = A[0], A[1]
    return np.array([[x], [y]])


