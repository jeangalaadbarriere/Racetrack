import svgutils.transform as sg
#import svglib
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from PIL.Image import *
import numpy as np

import time

#import pour BFS
from collections import deque

#imports de code
import svg
from grille import Grille
from move import Move
from configuration import Configuration
#from pile import Pile

#pour le stockage en keyword tree
from treelib import Node, Tree
from arbre import Arbre




#ATTENTION: MODIFICATION PROBABLE A FAIRE PCQ LA FONCTION QUI VERIFIE SI ON A LE DROIT D ETRE A UN ENDROIT CONFOND PEUT ETRE LA OU ON EST ET LA OU ON VA

def main(pas):
    # on définit les coordonnées du centre du plan
    print("################   Avec le pas:", pas)

    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    print("le centre:"+ str(x_centre)+", "+str(y_centre))
    matrice = create_matrice_des_possibles(pas, 212, 300)

    #On crée maintenant la grille sur laquelle l'aglorithme BFS va s'appliquer
    #On définit les coordonnées de départ I0<nbX, J0<nbY, et celles de la ligne d arrivee:
    I0= int(nbX * 2/10)
    J0= int(nbY * 5/10)-1
    print("coordonnées du départ:", I0, J0)
    ArriveeA=np.array([[0],[int(nbY * 1/2)]])
    ArriveeB=np.array([[int(nbX * 1/2)],[int(nbY * 1/2)]])
    G = Grille(nbX, nbY, np.array([[I0],[J0]]), (ArriveeA, ArriveeB))
    G.grid = matrice

    #On utilise BFS, si on veux avoir toutes les solutions de BFS avant de toruver les optimales
    #for config in BFS(G):
        #config.affiche()
        #print("on a une config")

    #bfs = BFS(G)

    #bfs = chemin_opt(G, x_centre, y_centre)
    bfs = BFS(G, x_centre, y_centre)
    #config = bfs[0]
    print("nombre de solutions différentes optimales trouvées:", len(bfs))
    print("en voici une (ouvrir circuit_plus_grille_plus_itin.png):")
    for i, config in enumerate(bfs):


        #On crée le .svg de l'itinéraire optimal EN AJOUTANT LA LIGNE D ARRIVEE
        draw_itin(config, pas, 212, 300, pas*I0, pas*J0, ArriveeA, ArriveeB, matrice)
        #create new SVG figure
        fig = sg.SVGFigure(212, 300)
        #on crée la grille qui va s'appeller grille.svg
        draw_grille(pas, 212, 300)

        fig1 = sg.fromfile('circuit_non_rond.svg')
        fig2 = sg.fromfile('grille.svg')
        fig3 = sg.fromfile('itinerary.svg')

        plot1 = fig1.getroot()
        plot2 = fig2.getroot()
        plot3 = fig3.getroot()

        fig.append([plot1, plot2, plot3])
        fig.save("circuit_plus_grille_plus_itin.svg")
        drawing = svg2rlg("circuit_plus_grille_plus_itin.svg")

        #fichier png qui montre le circuit avec le cadrillage par dessus
        renderPM.drawToFile(drawing, "circuit_plus_grille_plus_itin_"+str(pas)+" option numero"+str(i)+".png", fmt="PNG")

    print("end")
    return

def create_matrice_des_possibles(pas, tailleX, tailleY):

    #On est obligé de convertir notre svg en png pour accéder aux couleurs des pixels
    #Attention: on ouvre le circuit SANS la grille pour ne pas fausser l'analyse de couleur des pixels
    drawing = svg2rlg("circuit.svg")
    renderPM.drawToFile(drawing, "file.png", fmt="PNG")

    image = open("file.png")

    #on garde les bons rapports hauteur/largeur, il faut cependant prendre cela en compte
    alphaX = image.width/tailleX
    alphaY = image.height/tailleY

    nbX = int(tailleX/pas)
    nbY = int(tailleY/pas)

    print("taille de la grille:", nbX, nbY)

    matrice = np.zeros((nbX+1, nbY+1))
    for i in range(nbX):
        for j in range(nbY):
            (rouge, vert, bleu) = image.getpixel((pas*i*alphaX, pas*j*alphaY))
            #par défaut, on enlève, on ne garde que le vert
            matrice[i, j] = 0
            if vert != 0 and rouge == 0 and bleu == 0:
                matrice[i, j] = 1
    return matrice

def draw_grille(pas, tailleX, tailleY):
    s = svg.SVG()
    s.create(tailleX, tailleY)
    k = 0
    while (k*pas < tailleX):
        s.line("#000000", 1, k*pas, 0, k*pas, tailleY)
        k = k+1
    k = 0
    while (k*pas < tailleY):
        s.line("#000000", 1, 0, k*pas, tailleX, k*pas)
        k = k+1
    s.finalize()
    try:
        s.save("grille.svg")
    except IOError as ioe:
        print(ioe)

def toCoord(A):
    x, y = A[0], A[1]
    return np.array([[x], [y]])

def draw_itin(config, pas, tailleX, tailleY, x0, y0, arriveeA, arriveeB, matrice):
    s = svg.SVG()
    s.create(tailleX, tailleY)
    n = len(config.chemin)
    x = x0
    y = y0

    #on ajoute la ligne d'arrivée
    s.line("#ffff00", 5, arriveeA[0][0]*pas, arriveeA[1][0]*pas, arriveeB[0][0]*pas, arriveeB[1][0]*pas)

    s.circle("#0000ff", 1, "#0000ff", 5, x0, y0)
    for k in range(1,n):
        M = config.chemin[k]
        pos=toCoord(M.position)
        valeurpoint = matrice[int(x0/pas)][int(y0/pas)]
        if valeurpoint == 0:
            #ce cas n'est que pour signaler si le bfs nous fait passer par un point interdit
            #On a pas le droit d'être sur cette case pourtant on y est..., on indique ceci sur la graphe en mettant une pastille orange
            s.circle("#0000ff", 1, "#ff7f00", 5, x0, y0)
        else:
            s.circle("#0000ff", 1, "#0000ff", 5, x0, y0)
        s.line("#0000ff", 5, x0, y0, pos[0][0][0]*pas, pos[1][0][0]*pas)
        x0 = pos[0][0][0]*pas
        y0 = pos[1][0][0]*pas
    s.circle("#ffff00", 1, "#ffff00", 5, x0, y0)
    s.finalize()
    try:
        s.save("itinerary.svg")
    except IOError as ioe:
        print(ioe)

def BFS(grille, x_centre, y_centre):
    '''

    Parameters
    ----------
    grille : type Grille, représente le circuit

    Returns
    -------
    chemins_possibles : liste d'éléments de type Configuration, avec point d'arrivée, chemin parcouru, et distance parcourue
        retourne tous les chemins optimaux de manière temporelle(tours minimaux)

    '''
    delta = np.array([[[0],[0]] for k in range(9)])
    for i in ([-1, 0, 1]):
        for j in [-1, 0, 1]:
            delta[(i+1)*3+j+1][0,0] = i
            delta[(i+1)*3+j+1][1,0] = j
    Q = deque() #file des configurations explorés
    chemins_possibles = [] #on énumère les chemins possibles
    M = Move(grille.depart, [[0],[0]])
    Q.append(Configuration(M, [], 0))
    last_turn = -1 #permet de terminer la boucle lorsqu'au moins un chemin optimal est trouvé
    while (len(Q) != 0):
        config = Q.popleft() #on explore à partir de la première configuration
        M, chemin, dist = config.move, config.chemin, config.distance
        chemin.append(M)
        if (last_turn != -1 and len(chemin) > last_turn): #si un chemin optimal a déjà été trouvé, on n'explore que ceux de même longueur
            return chemins_possibles
        for a in delta: #on explore chaque mouvement possible
            C = Move(M.position + a + M.deplacement, a + M.deplacement)

            #on rajoute une condition qui empèche un retour en arrière
            # x0, y0 sont les coordonnées du centre de l'obstacle: on le centre au milieu du plan
            if C.deplacement[1,0]*(C.position[0,0]-x_centre) >= C.deplacement[0,0]*(C.position[1,0]-y_centre):
                #optimisation qui borne chaque composante de la vitesse par la racine carrée de la distance qu'il reste
                #avec le bord de cette coordonnée
                if C.deplacement[0,0]<=np.sqrt(2 + 2*x_centre - C.position[0,0]) and C.deplacement[1,0]<=np.sqrt(2 + 2*y_centre - C.position[1,0]):
                    if(C.estValable(grille)): #Si le mouvement est valable, on le retient
                        if C.deplacement[0,0] != 0 or C.deplacement[1,0] != 0: #on enlève le cas de vitesse nulle pour mettre départ=arrivée
                            if C.Goal(grille) and (C.deplacement[1,0] != 0 or C.deplacement[0,0] != 0):
                                last_turn = len(chemin)
                                chemin.append(C) #Si le chemin aboutit, on la rajoute aux chemins possibles
                                chemins_possibles.append(Configuration(C, copiesanseffetdebord(chemin), dist + np.linalg.norm(a + M.deplacement)))
                                chemin.pop()
                                #ATTENTION ON A UN EFFET DE BORD SUR CHEMIN
                                #IL FAUT DONC UTILISER UNE COPE LORSQU ON AJOUTE DANS CHEMIN POSSIBLE
                                #ON A DONC POUR CELA CREE LA FONCTION COPIESANSEFFETDEBORDCHEMIN=CHEMIN
                            else:
                                chemin.append(C)
                                Q.append(Configuration(C, copiesanseffetdebord(chemin), dist + np.linalg.norm(a + M.deplacement)))
                                chemin.pop()
    return chemins_possibles

def copiesanseffetdebord(chemin):
    cheminres = [chemin[k] for k in range(len(chemin))]
    return cheminres

def chemin_opt(grille, x_centre, y_centre):
    '''

    Parameters
    ----------
    grille : type Grille

    Returns
    -------
    chemins_opt : tous les chemins optimaux en comparant la distance en plus du temps

    '''
    chemins_possibles = BFS(grille, x_centre, y_centre)
    chemins_opt = []
    dist_min = chemins_possibles[0].distance
    for config in chemins_possibles:
        dist = config.distance
        if dist < dist_min:
            dist_min = dist
    for config in chemins_possibles:
        if config.distance == dist_min:
            chemins_opt.append(config)
    return chemins_opt

def trouve_cercle_circonscrit(pas, tailleX, tailleY):
    #on prend ces trois grandeurs en entrée, car ça ne sert à rien de prendre un cercke plus précis que ce qu'on va
    #se permettre par la suite

    drawing = svg2rlg("circuit_non_rond.svg")
    renderPM.drawToFile(drawing, "file_non_rond.png", fmt="PNG")
    image = open("file_non_rond.png")

    #on garde les bons rapports hauteur/largeur, il faut cependant prendre cela en compte
    alphaX = image.width/tailleX
    alphaY = image.height/tailleY

    nbX = int(tailleX/pas)
    nbY = int(tailleY/pas)

    print("taille de la grille:", nbX, nbY)


    #1/ On trouve le centre
    #A AMELIORER!!
    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2


    liste_bon_points = []
    matrice = np.zeros((nbX + 1, nbY + 1))
    for i in range(nbX):
        for j in range(nbY):
            (rouge, vert, bleu) = image.getpixel((pas * i * alphaX, pas * j * alphaY))
            # par défaut, on enlève, on ne garde que le vert
            matrice[i, j] = 0
            if vert != 0 and rouge == 0 and bleu == 0:
                matrice[i, j] = 1



    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if x == 0:
                liste_bon_points.append([i,j])

    x_max = max([x[0] for x in liste_bon_points])
    x_min = min([x[0] for x in liste_bon_points])
    y_max = max([x[1] for x in liste_bon_points])
    y_min = min([x[1] for x in liste_bon_points])

    rayon = (max(x_max-x_centre, x_centre-x_min, y_max-y_centre, y_centre-y_min)-1)/2
    print(rayon)



    s = svg.SVG()
    s.create(tailleX, tailleY)

    #on trace le cercle qu'on s'autorise
    s.circle("#000000", 2*rayon*pas, "#ff7f00", 1, x_centre*pas, y_centre*pas)
    #on trace les points qui existent
    for (x,y) in liste_bon_points:
        s.circle("#000000", 1, "#ff7f00", 5, x*pas, y*pas)
    s.finalize()
    try:
        s.save("points possibles.svg")
    except IOError as ioe:
        print(ioe)


    #maintenant on va renvoyer la matrice qui donne la grille facile de laquelle on va trouver la sol opt
    matrice = np.zeros((nbX + 1, nbY + 1))
    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if (i-x_centre)**2+(j-y_centre)**2 > rayon**2 and i != 0 and j != 0 and i != nbX and j != nbY:
                matrice[i,j] = 1

    #on trace les points qui existent
    s2 = svg.SVG()
    s2.create(tailleX, tailleY)
    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if x == 1:
                s2.circle("#000000", 1, "#ff7f00", 5, i*pas, j*pas)
    s2.finalize()
    try:
        s2.save("points possibles2.svg")
    except IOError as ioe:
        print(ioe)

    return matrice

def main_nouvelle_version(pas):
    #1/ on trouve une solution autour de l'enveloppe convexe la plus simple à trouver
    # on définit les coordonnées du centre du plan
    print("################   Avec le pas:", pas)

    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    matrice = trouve_cercle_circonscrit(pas, 212, 300)

    #On crée maintenant la grille sur laquelle l'aglorithme BFS va s'appliquer
    #On définit les coordonnées de départ I0<nbX, J0<nbY, et celles de la ligne d arrivee:
    I0= int(nbX * 2/10)
    J0= int(nbY * 5/10)-1
    print("coordonnées du départ:", I0, J0)
    ArriveeA=np.array([[0],[int(nbY * 1/2)]])
    ArriveeB=np.array([[int(nbX * 1/2)],[int(nbY * 1/2)]])
    G = Grille(nbX, nbY, np.array([[I0],[J0]]), (ArriveeA, ArriveeB))
    G.grid = matrice


    #On trouve un trajet opt
    bfs = BFS_arbre(G, [I0,J0], x_centre, y_centre)
    #config = bfs[0]
    print("nombre de solutions différentes optimales trouvées:", len(bfs))
    print("en voici une (ouvrir circuit_plus_grille_plus_itin.png):")
    for i, config in enumerate(bfs):


        #On crée le .svg de l'itinéraire optimal EN AJOUTANT LA LIGNE D ARRIVEE
        draw_itin(config, pas, 212, 300, pas*I0, pas*J0, ArriveeA, ArriveeB, matrice)
        #create new SVG figure
        fig = sg.SVGFigure(212, 300)
        #on crée la grille qui va s'appeller grille.svg
        draw_grille(pas, 212, 300)

        fig1 = sg.fromfile('circuit.svg')
        fig2 = sg.fromfile('grille.svg')
        fig3 = sg.fromfile('itinerary.svg')

        plot1 = fig1.getroot()
        plot2 = fig2.getroot()
        plot3 = fig3.getroot()

        fig.append([plot1, plot2, plot3])
        fig.save("circuit_plus_grille_plus_itin.svg")
        drawing = svg2rlg("circuit_plus_grille_plus_itin.svg")

        #fichier png qui montre le circuit avec le cadrillage par dessus
        renderPM.drawToFile(drawing, "circuit_plus_grille_plus_itin_"+str(pas)+" option numero"+str(i)+".png", fmt="PNG")

    #2/ on applique l'algo d'hélène




    return

def BFS_arbre(grille, depart, x_centre, y_centre):
    tree = Arbre(depart)
    indice_stockage = 0
    distance_min = 0


    delta = np.array([[0,0] for k in range(9)])
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            delta[(i+1)*3+j+1][0] = i
            delta[(i+1)*3+j+1][1] = j
    Q = deque()
    chemins_possibles = []


    Q.append(0)

    last_turn = -1
    while (len(Q) != 0):
        indice_queue = Q.popleft()

        sommet = tree.get_sommet(indice_queue)
        if (last_turn != -1 and sommet.get_depth() >= last_turn):
            vrais_chemins_possibles = []
            print("nombre de sommets crées:", tree.get_indice_act())
            for indice in chemins_possibles:
                vrais_chemins_possibles.append(tree.ret_chemin(indice))
            return vrais_chemins_possibles

        for a in delta:
            position2 = [sommet.get_position()[0] + sommet.get_speed()[0] + a[0], sommet.get_position()[1] + sommet.get_speed()[1] + a[1]]
            vitesse2 = [sommet.get_speed()[0] + a[0], sommet.get_speed()[1] + a[1]]
            distance2 = sommet.get_distance() + (vitesse2[0]**2 + vitesse2[1]**2)
            profondeur2 = sommet.get_depth() + 1

            if vitesse2[0] != 0 or vitesse2[1] != 0:
                if vitesse2[1] * (position2[0] - x_centre) >= vitesse2[0] * (position2[1] - y_centre):

                    if vitesse2[0]**2 <= 2+2*x_centre - position2[0] and vitesse2[1]**2 <= 2+2*y_centre-position2[1]:

                        if estValable(position2, vitesse2, grille):
                            #indice_stockage += tree.create_sommet_unique_bis(position2, vitesse2, distance2, profondeur2, indice_queue)


                            indice_stockage += 1
                            tree.create_sommet(position2, vitesse2, distance2, profondeur2, indice_queue)

                            if Goal(position2, vitesse2, grille):
                                last_turn = profondeur2
                                if distance_min == 0:
                                    distance_min = distance2
                                if distance2 == distance_min:
                                    chemins_possibles.append(indice_stockage)
                                if distance2 < distance_min:
                                    distance_min = distance2
                                    chemins_possibles = [indice_stockage]
                            else:
                                # ATTENTION A BIEN VERIFIER QUE LE SOMMET N EXISTE PAS ENCORE
                                #CA PEUT ETRE UNE PROCHAINE OPTI: ON NE VEUT QU'UNE SOL OPT DONC SI DEUX IT RENVOIENT AU MEME POINT, ON LES FUSIONNE
                                #EN GARDANT LE MEILLEUR ITINERAIRE DES DEUX (TEMPS PUIS DISTANCE)
                                #POUR L'INSTANT ON VA JUSTE METTRE A JOUR UNE LISTE RANDOM QUI GARDE EN MEMOIR LE GRAPH
                                Q.append(indice_stockage)
    return chemins_possibles

def intersect(A2, B2, position, vitesse):
    '''
    Parameters
    ----------
    A : point, de coordonnées (x_A, y_A)
    B : point, de coordonnées (x_b, y_b)

    Returns
    -------
    None.

    '''

    M_pos, M_dep = position, vitesse #ON NE DEVRAIT PAS AVOIR BESOIN DE TOCOORD CETTE FOIS
    x, y = M_pos[0], M_pos[1]
    u, v = M_dep[0], M_dep[1]

    x_A, y_A = A2[0], A2[1]
    x_B, y_B = B2[0], B2[1]

    # on le met au début car c'est le cas le plus probable donc on diminue la complexité en moyenne
    if (x_A == x and y_A == y) or (x_B == x and y_B == y):
        return True

    if (x_A == x_B):  # s'il s'agit d'un segment vertical
        if u == 0:
            if min(y_A, y_B) <= y and y <= max(y_A, y_B) and x == x_A:
                return True
            return False
        elif ((x_A <= x and x - u < x_A) or (x_A >= x and x - u > x_A)) and (
                min(y_A, y_B) <= y - v * (x - x_A) / u and y - v * (x - x_A) / u <= max(y_A, y_B)):
            # ATTENTION: LES CONDITIONS SERVENT A INDIQUER SI LE DEPLACEMENT FAIT TRAVERSER LA FRONTIERE, C EST DONC DES INEGALITES INVERSEES
            # IL FAUT DONC COMPARER LA POSITION INITIALE X-U QUI DOIT DONC ETRE DE L AUTRE COTE DE XA
            return True
        return False

    # il s'agit donc d'un segment horizontal
    if v == 0:
        if min(x_A, x_B) <= x and x <= max(x_A, x_B) and y == y_B:
            return True
        return False
    elif ((y_A <= y and y - v < y_A) or (y_A >= y and y - v > y_A)) and (
            min(x_A, x_B) <= (x - u * (y - y_A) / v) and (x - u * (y - y_A) / v) <= max(x_A, x_B)):
        return True
    return False

    return

#il y a surement plein d'erreurs de typage
def estValable(M_pos, M_dep, grille):

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


        ###M_pos, M_dep = toCoord(self.position), toCoord(self.deplacement)
        x, y = M_pos[0], M_pos[1]
        a, b = M_dep[0], M_dep[1]
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
                        if intersect(np.array([i,j]), np.array([i+sign_a,j]), M_pos, M_dep):
                            return False
                        return True
                    if grille.grid[i][j+sign_b] == 0:
                        if intersect(np.array([i,j]), np.array([i,j+sign_b]), M_pos, M_dep):
                            return False
                        return True
        return True

def Goal(position, vitesse, grille):
    return intersect(grille.arrivee[0], grille.arrivee[1], position, vitesse)

def main_arbre(pas):
    # on définit les coordonnées du centre du plan
    print("################   Avec le pas:", pas)

    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    print("le centre:"+ str(x_centre)+", "+str(y_centre))
    matrice = create_matrice_des_possibles(pas, 212, 300)

    #On crée maintenant la grille sur laquelle l'aglorithme BFS va s'appliquer
    #On définit les coordonnées de départ I0<nbX, J0<nbY, et celles de la ligne d arrivee:
    I0= int(nbX * 2/10)
    J0= int(nbY * 5/10)-1
    print("coordonnées du départ:", I0, J0)
    ArriveeA=[0, int(nbY * 1/2)]
    ArriveeB=[int(nbX * 1/2), int(nbY * 1/2)]
    G = Grille(nbX, nbY, [I0, J0], (ArriveeA, ArriveeB))
    G.grid = matrice

    #On utilise BFS, si on veux avoir toutes les solutions de BFS avant de toruver les optimales
    #for config in BFS(G):
        #config.affiche()
        #print("on a une config")

    #bfs = BFS(G)

    #bfs = chemin_opt(G, x_centre, y_centre)
    bfs = BFS_arbre(G, [I0, J0], x_centre, y_centre)
    #config = bfs[0]
    print("nombre de solutions différentes optimales trouvées:", len(bfs))
    print("en voici une (ouvrir circuit_plus_grille_plus_itin.png):")
    for i, chemin in enumerate(bfs):
        #On crée le .svg de l'itinéraire optimal EN AJOUTANT LA LIGNE D ARRIVEE
        draw_itin_arbre(chemin, pas, 212, 300, pas*I0, pas*J0, ArriveeA, ArriveeB, matrice)
        #create new SVG figure
        fig = sg.SVGFigure(212, 300)
        #on crée la grille qui va s'appeller grille.svg
        draw_grille(pas, 212, 300)

        fig1 = sg.fromfile('circuit.svg')
        fig2 = sg.fromfile('grille.svg')
        fig3 = sg.fromfile('itinerary.svg')

        plot1 = fig1.getroot()
        plot2 = fig2.getroot()
        plot3 = fig3.getroot()

        fig.append([plot1, plot2, plot3])
        fig.save("circuit_plus_grille_plus_itin.svg")
        drawing = svg2rlg("circuit_plus_grille_plus_itin.svg")

        #fichier png qui montre le circuit avec le cadrillage par dessus
        renderPM.drawToFile(drawing, "circuit_plus_grille_plus_itin_"+str(pas)+" option numero"+str(i)+"B non convexe.png", fmt="PNG")

    print("end")


    return

def draw_itin_arbre(chemin, pas, tailleX, tailleY, x0, y0, arriveeA, arriveeB, matrice):
    s = svg.SVG()
    s.create(tailleX, tailleY)
    n = len(chemin)
    #x0 = chemin[0][0]*pas
    #y0 = chemin[0][1]*pas

    #on ajoute la ligne d'arrivée
    s.line("#ffff00", 5, arriveeA[0]*pas, arriveeA[1]*pas, arriveeB[0]*pas, arriveeB[1]*pas)

    s.circle("#0000ff", 1, "#0000ff", 5, x0, y0)
    for k in range(0, n):
        pos = chemin[k]
        valeurpoint = matrice[int(x0/pas)][int(y0/pas)]
        if valeurpoint == 0:
            #ce cas n'est que pour signaler si le bfs nous fait passer par un point interdit
            #On a pas le droit d'être sur cette case pourtant on y est..., on indique ceci sur la graphe en mettant une pastille orange
            s.circle("#0000ff", 1, "#ff7f00", 5, x0, y0)
        else:
            s.circle("#0000ff", 1, "#0000ff", 5, x0, y0)
        s.line("#0000ff", 5, x0, y0, pos[0]*pas, pos[1]*pas)
        x0 = pos[0]*pas
        y0 = pos[1]*pas
    s.circle("#ffff00", 1, "#ffff00", 5, x0, y0)
    s.finalize()
    try:
        s.save("itinerary.svg")
    except IOError as ioe:
        print(ioe)

#le BFS unique ne fonctionne pas si bien que ça... la recherche d'antécédent similaires est trop couteuse en temps de
#calcul pour compenser le gain en complexité spatiale
def BFS_unique(grille, depart, x_centre, y_centre):
    tree = Arbre(depart)
    indice_stockage = 0
    distance_min = 0


    delta = np.array([[0,0] for k in range(9)])
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            delta[(i+1)*3+j+1][0] = i
            delta[(i+1)*3+j+1][1] = j
    Q = deque()
    chemins_possibles = []


    Q.append(0)

    last_turn = -1
    while (len(Q) != 0):
        indice_queue = Q.popleft()

        sommet = tree.get_sommet(indice_queue)
        #print("nombre de sommets crées:", tree.get_indice_act())

        if (last_turn != -1 and sommet.get_depth() >= last_turn):
            vrais_chemins_possibles = []
            print("nombre de sommets crées:", tree.get_indice_act())
            for indice in chemins_possibles:
                vrais_chemins_possibles.append(tree.ret_chemin(indice))
            return vrais_chemins_possibles

        for a in delta:
            position2 = [sommet.get_position()[0] + sommet.get_speed()[0] + a[0], sommet.get_position()[1] + sommet.get_speed()[1] + a[1]]
            vitesse2 = [sommet.get_speed()[0] + a[0], sommet.get_speed()[1] + a[1]]
            distance2 = sommet.get_distance() + (vitesse2[0]**2 + vitesse2[1]**2)
            profondeur2 = sommet.get_depth() + 1

            if vitesse2[0] != 0 or vitesse2[1] != 0:
                if vitesse2[1] * (position2[0] - x_centre) >= vitesse2[0] * (position2[1] - y_centre):

                    if vitesse2[0]**2 <= 2+2*x_centre - position2[0] and vitesse2[1]**2 <= 2+2*y_centre-position2[1]:

                        if estValable(position2, vitesse2, grille):
                            #indice_stockage += tree.create_sommet_unique_bis(position2, vitesse2, distance2, profondeur2, indice_queue)

                            indice_stockage += 1
                            tree.create_sommet_unique_bis(position2, vitesse2, distance2, profondeur2, indice_queue)

                            if Goal(position2, vitesse2, grille):
                                last_turn = profondeur2
                                if distance_min == 0:
                                    distance_min = distance2
                                if distance2 == distance_min:
                                    chemins_possibles.append(indice_stockage)
                                if distance2 < distance_min:
                                    distance_min = distance2
                                    chemins_possibles = [indice_stockage]
                            else:
                                # ATTENTION A BIEN VERIFIER QUE LE SOMMET N EXISTE PAS ENCORE
                                #CA PEUT ETRE UNE PROCHAINE OPTI: ON NE VEUT QU'UNE SOL OPT DONC SI DEUX IT RENVOIENT AU MEME POINT, ON LES FUSIONNE
                                #EN GARDANT LE MEILLEUR ITINERAIRE DES DEUX (TEMPS PUIS DISTANCE)
                                #POUR L'INSTANT ON VA JUSTE METTRE A JOUR UNE LISTE RANDOM QUI GARDE EN MEMOIR LE GRAPH
                                Q.append(indice_stockage)
    return chemins_possibles

#UTILISATION DE LA PROPRIETE DE L ENVELOPPE CONVEXE POUR DE MEILLEURS RESULTATS
#résultats expérimentaux: une distance de 2 max par rapport à l'enveloppe convexe de l'obstacle
def main_convexe(pas):
    print("################   Avec le pas:", pas)

    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    print("le centre:"+ str(x_centre)+", "+str(y_centre))
    matrice, R = create_matrice_des_possibles_convexe(pas, 212, 300)

    I0= int(nbX * 2/10)
    J0= int(nbY * 5/10)-1
    print("coordonnées du départ:", I0, J0)
    ArriveeA=[0, int(nbY * 1/2)]
    ArriveeB=[int(nbX * 1/2 - 3), int(nbY * 1/2)]
    G = Grille(nbX, nbY, [I0, J0], (ArriveeA, ArriveeB))
    G.grid = matrice

    bfs = BFS_arbre_convexe(G, [I0, J0], x_centre, y_centre, R)

    print("nombre de solutions différentes optimales trouvées:", len(bfs))
    print("en voici une (ouvrir circuit_plus_grille_plus_itin.png):")
    for i, chemin in enumerate(bfs):
        draw_itin_arbre(chemin, pas, 212, 300, pas*I0, pas*J0, ArriveeA, ArriveeB, matrice)
        fig = sg.SVGFigure(212, 300)
        draw_grille(pas, 212, 300)

        fig1 = sg.fromfile('circuit.svg')
        fig2 = sg.fromfile('grille.svg')
        fig3 = sg.fromfile('itinerary.svg')

        plot1 = fig1.getroot()
        plot2 = fig2.getroot()
        plot3 = fig3.getroot()

        fig.append([plot1, plot2, plot3])
        fig.save("circuit_plus_grille_plus_itin.svg")
        drawing = svg2rlg("circuit_plus_grille_plus_itin.svg")

        renderPM.drawToFile(drawing, "circuit_plus_grille_plus_itin_"+str(pas)+" option numero"+str(i)+"CONVEXE.png", fmt="PNG")

    print("end")


    return

def main_non_convexe(pas):
    print("################   Avec le pas:", pas)

    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    print("le centre:"+ str(x_centre)+", "+str(y_centre))
    matrice, R = create_matrice_des_possibles_non_convexe(pas, 212, 300)

    I0= int(nbX * 2/10)
    J0= int(nbY * 5/10)-1
    print("coordonnées du départ:", I0, J0)
    ArriveeA=[0, int(nbY * 1/2)]
    ArriveeB=[int(nbX * 1/2), int(nbY * 1/2)]
    G = Grille(nbX, nbY, [I0, J0], (ArriveeA, ArriveeB))
    G.grid = matrice

    bfs = BFS_arbre_convexe(G, [I0, J0], x_centre, y_centre, R)

    print("nombre de solutions différentes optimales trouvées:", len(bfs))
    print("en voici une (ouvrir circuit_plus_grille_plus_itin.png):")
    for i, chemin in enumerate(bfs):
        draw_itin_arbre(chemin, pas, 212, 300, pas*I0, pas*J0, ArriveeA, ArriveeB, matrice)
        fig = sg.SVGFigure(212, 300)
        draw_grille(pas, 212, 300)

        fig1 = sg.fromfile('circuit.svg')
        fig2 = sg.fromfile('grille.svg')
        fig3 = sg.fromfile('itinerary.svg')

        plot1 = fig1.getroot()
        plot2 = fig2.getroot()
        plot3 = fig3.getroot()

        fig.append([plot1, plot2, plot3])
        fig.save("circuit_plus_grille_plus_itin.svg")
        drawing = svg2rlg("circuit_plus_grille_plus_itin.svg")

        renderPM.drawToFile(drawing, "circuit_plus_grille_plus_itin_"+str(pas)+" option numero"+str(i)+" NON CONVEXE.png", fmt="PNG")

    print("end")


    return

def create_matrice_des_possibles_convexe(pas, tailleX, tailleY):

    #On est obligé de convertir notre svg en png pour accéder aux couleurs des pixels
    #Attention: on ouvre le circuit SANS la grille pour ne pas fausser l'analyse de couleur des pixels
    drawing = svg2rlg("circuit.svg")
    renderPM.drawToFile(drawing, "file.png", fmt="PNG")

    image = open("file.png")

    #on garde les bons rapports hauteur/largeur, il faut cependant prendre cela en compte
    alphaX = image.width/tailleX
    alphaY = image.height/tailleY

    nbX = int(tailleX/pas)
    nbY = int(tailleY/pas)

    print("taille de la grille:", nbX, nbY)

    #1/ On trouve le centre
    #A AMELIORER!!
    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2

    liste_bon_points = []
    matrice = np.zeros((nbX+1, nbY+1))
    for i in range(nbX):
        for j in range(nbY):
            (rouge, vert, bleu) = image.getpixel((pas*i*alphaX, pas*j*alphaY))
            #par défaut, on enlève, on ne garde que le vert
            matrice[i, j] = 0
            if vert != 0 and rouge == 0 and bleu == 0:
                matrice[i, j] = 1

    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if x == 0:
                liste_bon_points.append([i,j])

    x_max = max([x[0] for x in liste_bon_points])
    x_min = min([x[0] for x in liste_bon_points])
    y_max = max([x[1] for x in liste_bon_points])
    y_min = min([x[1] for x in liste_bon_points])

    rayon = (max(x_max-x_centre, x_centre-x_min, y_max-y_centre, y_centre-y_min)-1)/2
    R=(rayon)*2
    print("rayon du carcle circonscrit", rayon)
    rayon = (rayon+2)**2 #on définit une borne max

    #puis on modifie notre matrice en mettant 0 à tous les points trop loin du centre
    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if x == 1 and (i-x_centre)**2+(j-y_centre)**2 > rayon:
                matrice[i, j] = 0

    return matrice,R

def create_matrice_des_possibles_non_convexe(pas, tailleX, tailleY):
    #On est obligé de convertir notre svg en png pour accéder aux couleurs des pixels
    #Attention: on ouvre le circuit SANS la grille pour ne pas fausser l'analyse de couleur des pixels
    drawing = svg2rlg("circuit.svg")
    renderPM.drawToFile(drawing, "file.png", fmt="PNG")

    image = open("file.png")

    #on garde les bons rapports hauteur/largeur, il faut cependant prendre cela en compte
    alphaX = image.width/tailleX
    alphaY = image.height/tailleY

    nbX = int(tailleX/pas)
    nbY = int(tailleY/pas)

    print("taille de la grille:", nbX, nbY)

    #1/ On trouve le centre
    #A AMELIORER!!
    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2

    liste_bon_points = []
    matrice = np.zeros((nbX+1, nbY+1))
    for i in range(nbX):
        for j in range(nbY):
            (rouge, vert, bleu) = image.getpixel((pas*i*alphaX, pas*j*alphaY))
            #par défaut, on enlève, on ne garde que le vert
            matrice[i, j] = 0
            if vert != 0 and rouge == 0 and bleu == 0:
                matrice[i, j] = 1

    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if x == 0:
                liste_bon_points.append([i,j])

    x_max = max([x[0] for x in liste_bon_points])
    x_min = min([x[0] for x in liste_bon_points])
    y_max = max([x[1] for x in liste_bon_points])
    y_min = min([x[1] for x in liste_bon_points])

    rayon = (max(x_max-x_centre, x_centre-x_min, y_max-y_centre, y_centre-y_min)-1)/2
    R=(rayon)*2
    print("rayon du carcle circonscrit", rayon)
    rayon = (rayon+2)**2 #on définit une borne max

    #puis on modifie notre matrice en mettant 0 à tous les points trop loin du centre
    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if x == 1 and (i-x_centre)**2+(j-y_centre)**2 > rayon:
                #matrice[i, j] = 0
                1+1
    return matrice,R

def BFS_arbre_convexe(grille, depart, x_centre, y_centre, R): #rayon de l'enveloppe convexe
    tree = Arbre(depart)
    indice_stockage = 0
    distance_min = 0


    delta = np.array([[0,0] for k in range(9)])
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            delta[(i+1)*3+j+1][0] = i
            delta[(i+1)*3+j+1][1] = j
    Q = deque()
    chemins_possibles = []


    Q.append(0)

    last_turn = -1
    while (len(Q) != 0):
        indice_queue = Q.popleft()

        sommet = tree.get_sommet(indice_queue)
        if (last_turn != -1 and sommet.get_depth() >= last_turn):
            vrais_chemins_possibles = []
            print("nombre de sommets crées:", tree.get_indice_act())
            for indice in chemins_possibles:
                vrais_chemins_possibles.append(tree.ret_chemin(indice))
            return vrais_chemins_possibles

        for a in delta:
            position2 = [sommet.get_position()[0] + sommet.get_speed()[0] + a[0], sommet.get_position()[1] + sommet.get_speed()[1] + a[1]]
            vitesse2 = [sommet.get_speed()[0] + a[0], sommet.get_speed()[1] + a[1]]
            distance2 = sommet.get_distance() + (vitesse2[0]**2 + vitesse2[1]**2)
            profondeur2 = sommet.get_depth() + 1

            if vitesse2[0] != 0 or vitesse2[1] != 0:
                if vitesse2[1] * (position2[0] - x_centre) >= vitesse2[0] * (position2[1] - y_centre):

                    #if vitesse2[0]**2 <= 2+2*x_centre - position2[0] and vitesse2[1]**2 <= 2+2*y_centre-position2[1]:

                    #on gère mieux que ça les conditions de vitesse
                    truevitesseX = False
                    if vitesse2[0] >= 0:
                        if vitesse2[0]**2 <= 2*(x_centre + R - position2[0]):
                            truevitesseX = True
                    else:
                        if vitesse2[0]**2 <= 2*(position2[0] - x_centre + R):
                            truevitesseX = True

                    truevitesseY = False
                    if vitesse2[1] >= 0:
                        if vitesse2[1]**2 <= 2*(y_centre + R - position2[1]):
                            truevitesseY= True
                    else:
                        if vitesse2[1]**2 <= 2*(position2[1] - y_centre + R):
                            truevitesseY = True
                    #if True:
                    if truevitesseX == True and truevitesseY == True:
                        if estValable(position2, vitesse2, grille):

                            indice_stockage += 1
                            tree.create_sommet(position2, vitesse2, distance2, profondeur2, indice_queue)

                            if Goal(position2, vitesse2, grille):
                                last_turn = profondeur2
                                if distance_min == 0:
                                    distance_min = distance2
                                if distance2 == distance_min:
                                    chemins_possibles.append(indice_stockage)
                                if distance2 < distance_min:
                                    distance_min = distance2
                                    chemins_possibles = [indice_stockage]
                            else:
                                # ATTENTION A BIEN VERIFIER QUE LE SOMMET N EXISTE PAS ENCORE
                                #CA PEUT ETRE UNE PROCHAINE OPTI: ON NE VEUT QU'UNE SOL OPT DONC SI DEUX IT RENVOIENT AU MEME POINT, ON LES FUSIONNE
                                #EN GARDANT LE MEILLEUR ITINERAIRE DES DEUX (TEMPS PUIS DISTANCE)
                                #POUR L'INSTANT ON VA JUSTE METTRE A JOUR UNE LISTE RANDOM QUI GARDE EN MEMOIR LE GRAPH
                                Q.append(indice_stockage)
    return chemins_possibles

def create_matrice_des_possibles_convexe_int(pas, tailleX, tailleY):
    drawing = svg2rlg("circuit.svg")
    renderPM.drawToFile(drawing, "file.png", fmt="PNG")
    image = open("file.png")
    alphaX = image.width/tailleX
    alphaY = image.height/tailleY
    nbX = int(tailleX/pas)
    nbY = int(tailleY/pas)
    print("taille de la grille:", nbX, nbY)
    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    liste_bon_points = []
    matrice = np.zeros((nbX+1, nbY+1))
    for i in range(nbX):
        for j in range(nbY):
            (rouge, vert, bleu) = image.getpixel((pas*i*alphaX, pas*j*alphaY))
            matrice[i, j] = 0
            if vert != 0 and rouge == 0 and bleu == 0:
                matrice[i, j] = 1
    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if x == 0:
                liste_bon_points.append([i,j])
    x_max = max([x[0] for x in liste_bon_points])
    x_min = min([x[0] for x in liste_bon_points])
    y_max = max([x[1] for x in liste_bon_points])
    y_min = min([x[1] for x in liste_bon_points])
    rayon = (max(x_max-x_centre, x_centre-x_min, y_max-y_centre, y_centre-y_min)-1)/2
    R=(rayon)*2
    print("rayon du carcle circonscrit", rayon)
    rayon = (rayon+2)**2
    for i,L in enumerate(matrice):
        for j,x in enumerate(L):
            if x == 1 and (i-x_centre)**2+(j-y_centre)**2 > rayon:
                matrice[i, j] = 0

    #ON TROUVE MAINTENANT L ENVELOPPE CONVEXE, ET ON L AFFICHE

    #1. on récupère les points de l'obstacle: ceux qui valent 0 au milieu
    points_obstacle = []
    for i,L in enumerate(matrice):
        for j,p in enumerate(L):
            if p == 0 and (i-x_centre)**2+(j-y_centre)**2 <= rayon and i!=0 and i!=nbX:
                points_obstacle.append([i,j])

    for x in points_obstacle:
        matrice[x[0], x[1]] = 0


    #on lance Graham sur cette liste
    enveloppe = parcours_Graham(points_obstacle)

    for point in enveloppe:
        matrice[point[0], point[1]] = 0

    #il faut compléter l'intérieur si ce n'est pas des cases interdites
    enveloppe.sort() #on trie
    list_done = [False for x in enveloppe]
    last_gauche = enveloppe[0]
    last_droite = enveloppe[0]
    for i in range(1, len(enveloppe)):
        if enveloppe[i][0] == enveloppe[i-1][0]:
            list_done[i] = True
            list_done[i-1] = True
            for j in range(enveloppe[i-1][1]+1, enveloppe[i][1]):
                matrice[enveloppe[i][0],j] = 0
            last_gauche = enveloppe[i-1]
            last_droite = enveloppe[i]
        else:
            if list_done[i] == False:
                #s'il est à gauche:
                if enveloppe[i][1] < last_gauche[1]:
                    for j in range(enveloppe[i][1] + 1, last_droite[1]+1):
                        matrice[enveloppe[i][0], j] = 0
                else: #il est à droite:
                    for j in range(last_gauche[1], enveloppe[i][1]):
                        matrice[enveloppe[i][0], j] = 0

    return matrice,R

def parcours_Graham(L): #sert à trouver une enveloppe convexe à notre obstacle
    #1. on trouve le pivot et on le place en 0
    xmin,ymin = L[0][0], L[0][1]
    indicemin = 0
    for i,p in enumerate(L):
        if p[1] < ymin or (p[1]==ymin and p[0]<xmin):
            xmin = p[0]
            ymin = p[1]
            indicemin = i
    L[0], L[indicemin] = L[indicemin], L[0]
    x,y = L[0][0], L[0][1]
    #2. on trie le tableau par angle croissant par rapport au pivot
    for i in range(1, len(L)):
        if L[i][0] < x:
            L[i] = [np.pi/2 + np.tan((x - L[i][0]) / (L[i][1] - y)), L[i][0], L[i][1]]
        else:
            if L[i][0] == x:
                L[i] = [np.pi/2, L[i][0], L[i][1]]
            else:
                L[i] = [np.tan((L[i][1]-y)/(L[i][0]-x)), L[i][0], L[i][1]]
    L[1:].sort()
    for i in range(1, len(L)):
        L[i] = [L[i][1], L[i][2]]

    #3.
    pile=[]
    pile.append(L[0])
    pile.append(L[1])
    indice_pile = 1
    for i in range(2, len(L)):
        while (indice_pile >= 1 and produit_vectoriel(pile[indice_pile-1], pile[indice_pile], L[i])<=0) or pile[indice_pile]==L[i]:
            pile.pop()
            indice_pile -= 1
        pile.append(L[i])
        indice_pile +=1
    return pile

def produit_vectoriel(A, B, C):
    return (B[0]-A[0])*(C[1]-A[1]) - (C[0]-A[0])*(B[1]-A[1])

def main_convexe_int(pas):
    print("################   Avec le pas:", pas)

    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    print("le centre:"+ str(x_centre)+", "+str(y_centre))
    matrice, R = create_matrice_des_possibles_convexe_int(pas, 212, 300)
    print(matrice)
    I0= int(nbX * 2/10)
    J0= int(nbY * 5/10)-1
    print("coordonnées du départ:", I0, J0)
    ArriveeA=[0, int(nbY * 1/2)]
    ArriveeB=[int(nbX * 1/2), int(nbY * 1/2)]
    G = Grille(nbX, nbY, [I0, J0], (ArriveeA, ArriveeB))
    G.grid = matrice

    bfs = BFS_arbre_convexe(G, [I0, J0], x_centre, y_centre, R)

    print("nombre de solutions différentes optimales trouvées:", len(bfs))
    print("en voici une (ouvrir circuit_plus_grille_plus_itin.png):")
    for i, chemin in enumerate(bfs):
        draw_itin_arbre(chemin, pas, 212, 300, pas*I0, pas*J0, ArriveeA, ArriveeB, matrice)
        fig = sg.SVGFigure(212, 300)
        draw_grille(pas, 212, 300)

        fig1 = sg.fromfile('circuit.svg')
        fig2 = sg.fromfile('grille.svg')
        fig3 = sg.fromfile('itinerary.svg')

        plot1 = fig1.getroot()
        plot2 = fig2.getroot()
        plot3 = fig3.getroot()

        fig.append([plot1, plot2, plot3])
        fig.save("circuit_plus_grille_plus_itin.svg")
        drawing = svg2rlg("circuit_plus_grille_plus_itin.svg")

        renderPM.drawToFile(drawing, "circuit_plus_grille_plus_itin_"+str(pas)+" option numero"+str(i)+"CONVEXE INT.png", fmt="PNG")

    print("end")

    return

def main_arbre_obstacle_retour_arriere(pas):
    # on définit les coordonnées du centre du plan
    print("################   Avec le pas:", pas)

    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    print("le centre:"+ str(x_centre)+", "+str(y_centre))
    matrice = create_matrice_des_possibles(pas, 212, 300)

    #On crée maintenant la grille sur laquelle l'aglorithme BFS va s'appliquer
    #On définit les coordonnées de départ I0<nbX, J0<nbY, et celles de la ligne d arrivee:
    I0= 1
    J0= int(nbY * 4.5/10)
    print("coordonnées du départ:", I0, J0)
    ArriveeA=[int(nbX*1/2), int(nbY * 9/10)]
    ArriveeB=[int(nbX)-1, int(nbY * 9/10)]
    G = Grille(nbX, nbY, [I0, J0], (ArriveeA, ArriveeB))
    G.grid = matrice

    #On utilise BFS, si on veux avoir toutes les solutions de BFS avant de toruver les optimales
    #for config in BFS(G):
        #config.affiche()
        #print("on a une config")

    #bfs = BFS(G)

    #bfs = chemin_opt(G, x_centre, y_centre)
    bfs = BFS_arbre(G, [I0, J0], x_centre, y_centre)
    #config = bfs[0]
    print("nombre de solutions différentes optimales trouvées:", len(bfs))
    print("en voici une (ouvrir circuit_plus_grille_plus_itin.png):")
    for i, chemin in enumerate(bfs):
        #On crée le .svg de l'itinéraire optimal EN AJOUTANT LA LIGNE D ARRIVEE
        draw_itin_arbre(chemin, pas, 212, 300, pas*I0, pas*J0, ArriveeA, ArriveeB, matrice)
        #create new SVG figure
        fig = sg.SVGFigure(212, 300)
        #on crée la grille qui va s'appeller grille.svg
        draw_grille(pas, 212, 300)

        fig1 = sg.fromfile('circuit.svg')
        fig2 = sg.fromfile('grille.svg')
        fig3 = sg.fromfile('itinerary.svg')

        plot1 = fig1.getroot()
        plot2 = fig2.getroot()
        plot3 = fig3.getroot()

        fig.append([plot1, plot2, plot3])
        fig.save("circuit_plus_grille_plus_itin.svg")
        drawing = svg2rlg("circuit_plus_grille_plus_itin.svg")

        #fichier png qui montre le circuit avec le cadrillage par dessus
        renderPM.drawToFile(drawing, "circuit_plus_grille_plus_itin_"+str(pas)+" option numero"+str(i)+"B non convexe.png", fmt="PNG")

    print("end")


    return


def main_dijkstra(pas):
    print("ON UTILISE DIJKSTRA")
    print("################   Avec le pas:", pas)
    nbX = 212 // pas
    nbY = 300 // pas
    x_centre = nbX // 2
    y_centre = nbY // 2
    print("le centre:"+ str(x_centre)+", "+str(y_centre))
    matrice = create_matrice_des_possibles(pas, 212, 300)
    I0= int(nbX * 2/10)
    J0= int(nbY * 5/10)-1
    print("coordonnées du départ:", I0, J0)
    ArriveeA=[0, int(nbY * 1/2)]
    ArriveeB=[int(nbX * 1/2), int(nbY * 1/2)]


    delta = [[0,0], [1,0], [-1, 0], [0,-1], [1,-1], [-1,-1], [0,1], [1,1], [1,-1]]


    #il faut d'abord définir la matrice comprenant toutes les positions possibles
    l1 = int(np.sqrt(2*nbX))
    l2 = int(np.sqrt(2*nbY))
    grande_matrice = np.full((nbX, nbY, l1, l2), np.Infinity) #toutes les distances valent par défaut l'infini
    for i in range(nbX):
        for j in range(nbY):
            if matrice[i][j] == 0: #on a pas le droit d'être là
                for ia in range(l1):
                    for ib in range(l2):
                        grande_matrice[i,j,ia,ib] = False  #on rempli donc interdit sur ces cases-là

    grande_matrice[I0, J0, 0, 0] = 0 #on définit la position de départ


    predecesseur = grande_matrice.copy()
    Q = grande_matrice.copy()



    def isnull():
        for i in range(nbX):
            for j in range(nbY):
                for ia in range(l1):
                    for ib in range(l2):
                        if Q[i,j,ia,ib] == False:
                            return False
        return True

    def voisinde(s1):
        s1 = [i,j,ia,ib]
        return [[i+ia+d1, i+ib+d2, ia+d1, ib+d2] for [d1,d2] in delta]

    def maj_distances(s1, s2):
        if Q[s2] > Q[s1] + 1: #poids(s1,s2) = 1 car il n'y a qu'un itération en temps qui les sépare
            Q[s2] = Q[s1] + 1
            predecesseur[s2] = s1
        return

    def trouve_min(): #un peu lourd mais on a pas trop le choix
        min = np.Infinity
        indice = [0,0,0,0]
        for i in range(nbX):
            for j in range(nbY):
                for ia in range(l1):
                    for ib in range(l2):
                        if Q[i,j,ia,ib] != False:
                            if Q[i,j,ia,ib] < min:
                                min, indice = Q[i,j,ia,ib], [i,j,ia,ib]
        return indice

    while isnull(Q, nbX, nbY, l1, l2) == False:
        s1 = trouve_min()
        s1 = [i,j,ia,ib]
        Q[i,j,ia,ib] = False #on enlève ce sommet de ce qui nous interesse
        for s2 in voisinde(s1):
            maj_distances(s1, s2)


    return

debut = time.time()
main_dijkstra(25)
print("temps d'exécution:", time.time() - debut, "s")