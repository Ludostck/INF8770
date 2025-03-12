import numpy as np
import matplotlib.pyplot as py
from numpy import linalg as LA, strings


image_path = "./data/kodim01.png"

def KlRGB(imageName, quantificator, quantificatorID):
    imagelue = py.imread('./data/' + imageName + ".png")
    image=imagelue.astype('double')

    # Calcul des valeurs moyennes des RGB. Nécessaire pour calculer la matrice de covariance des RGB
    sommeR = 0.0
    sommeG = 0.0
    sommeB = 0.0
    for i in range(len(image)):
        for j in range(len(image[0])):
            sommeR=sommeR+image[i][j][0]
            sommeG=sommeG+image[i][j][1]
            sommeB=sommeB+image[i][j][2]
            
    nbPixels = len(image)*len(image[0])        
    MoyR= sommeR / nbPixels
    MoyG= sommeG / nbPixels
    MoyB= sommeB / nbPixels

    print("Moyenne des RGB")
    print(MoyR)
    print(MoyG)
    print(MoyB)

    # Calcul de la matrice de covariance des RGB
    print("Matrice de covariance")
    covRGB = np.zeros((3,3), dtype = "double")
    for i in range(len(image)):
        for j in range(len(image[0])):
            vecTemp=[[image[i][j][0] - MoyR], [image[i][j][1]] - MoyG, [image[i][j][2] - MoyB]]
            vecProdTemp = np.dot(vecTemp,np.transpose(vecTemp))
            covRGB = np.add(covRGB,vecProdTemp)

    covRGB = covRGB / nbPixels        
    print(covRGB)

    # Calcul des vecteurs propres et valeurs propres
    eigval, eigvec = LA.eig(covRGB)
    print("Valeur propre")
    print(eigval)
    print()
    print("Vecteur propre")
    print(eigvec)

    eigvec = np.transpose(eigvec)
    eigvecsansAxe0 = np.copy(eigvec)
    eigvecsansAxe0[0,:] = [0.0,0.0,0.0]
    eigvecsansAxe1 = np.copy(eigvec)
    eigvecsansAxe1[1,:] = [0.0,0.0,0.0]
    eigvecsansAxe2 = np.copy(eigvec)
    eigvecsansAxe2[2,:] = [0.0,0.0,0.0]

    imageKLsansAxe0 = np.copy(image)

    vecMoy =[[MoyR], [MoyG], [MoyB]] 
    for i in range(len(image)):
        for j in range(len(image[0])):
            vecTemp=[[image[i][j][0]], [image[i][j][1]], [image[i][j][2]]]
            imageKLsansAxe0[i][j][:] = np.reshape(np.dot(eigvecsansAxe0,np.subtract(vecTemp,vecMoy)),(3))

    imageKLsansAxe0 = np.copy(image)
    vecMoy =[MoyR, MoyG, MoyB] 
    imageRGBsansAxe0 = np.copy(image)

    for i in range(len(image)):
        for j in range(len(image[0])):
                imageRGBsansAxe0[i][j] = np.dot(eigvec, (imageRGBsansAxe0[i][j] - vecMoy))

    # On applique les quantificateur 
    for i in range(len(image)):
        for j in range(len(image[0])):
            imageRGBsansAxe0[i][j][0] = np.uint8(imageRGBsansAxe0[i][j][0] * quantificator[0])
            imageRGBsansAxe0[i][j][1] = np.uint8(imageRGBsansAxe0[i][j][1] * quantificator[1])
            imageRGBsansAxe0[i][j][2] = np.uint8(imageRGBsansAxe0[i][j][2] * quantificator[2])

    print(imageRGBsansAxe0)
    py.figure(figsize = (10,10))
    imageout = np.clip(imageRGBsansAxe0,0, 255)
    imageout= imageout.astype('uint8')
    py.imsave("data_transform/" + imageName + "_" + str(quantificatorID) + ".png", imageout)

if __name__ == "__main__":
    quantificator  = [[255, 255, 255],[255, 255, 16],[255, 255, 0],[255, 16, 16]]

    for i in range(len(quantificator)):
        KlRGB("kodim01", quantificator[i], i)

    for i in range(len(quantificator)):
        KlRGB("kodim02", quantificator[i], i)

    for i in range(len(quantificator)):
        KlRGB("kodim05", quantificator[i], i)

    for i in range(len(quantificator)):
        KlRGB("kodim13", quantificator[i], i)

    for i in range(len(quantificator)):
        KlRGB("kodim23", quantificator[i], i)
