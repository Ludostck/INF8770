import matplotlib
import numpy as np
import matplotlib.pyplot as py
from numpy import linalg as LA, strings
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve


# CHAT gpt generated
def ssim(image1, image2, window_size=11, C1=1e-4, C2=9e-4):
    """Calcule le SSIM entre deux images, canal par canal"""
    
    # Initialiser le SSIM total pour l'image
    ssim_total = 0
    num_channels = image1.shape[2]  # Nombre de canaux (3 pour RGB)
    
    # Calcul des moyennes locales (fenêtres) et du SSIM pour chaque canal
    for channel in range(num_channels):
        # Extraire le canal correspondant
        img1_channel = image1[:, :, channel]
        img2_channel = image2[:, :, channel]
        
        # Calcul des moyennes locales (fenêtres)
        window = np.ones((window_size, window_size)) / (window_size ** 2)  # Fenêtre moyenne
        mu1 = convolve(img1_channel, window, mode='nearest')
        mu2 = convolve(img2_channel, window, mode='nearest')
        
        # Calcul des variances locales et des covariances
        sigma1_sq = convolve(img1_channel ** 2, window, mode='nearest') - mu1 ** 2
        sigma2_sq = convolve(img2_channel ** 2, window, mode='nearest') - mu2 ** 2
        sigma12 = convolve(img1_channel * img2_channel, window, mode='nearest') - mu1 * mu2
        
        # Calcul du SSIM pour ce canal
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        
        # Moyenne du SSIM pour ce canal
        ssim_total += np.mean(ssim_map)
    
    # Moyenne du SSIM pour l'image entière
    ssim_index = ssim_total / num_channels
    
    return ssim_index

# Est basé sur le code cours ici:https: github.com/gabilodeau/INF8770/blob/master/Transformee%20KL%20sur%20image.ipynb
def Kl(image):

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

    covRGB = np.zeros((3,3), dtype = "double")
    for i in range(len(image)):
        for j in range(len(image[0])):
            vecTemp=[[image[i][j][0] - MoyR], [image[i][j][1]] - MoyG, [image[i][j][2] - MoyB]]
            vecProdTemp = np.dot(vecTemp,np.transpose(vecTemp))
            covRGB = np.add(covRGB,vecProdTemp)

    covRGB = covRGB / nbPixels        
    eigval, eigvec = LA.eig(covRGB)


    eigvec = np.transpose(eigvec)
    eigvecsansAxe2 = np.copy(eigvec)
    eigvecsansAxe2[2,:] = [0.0,0.0,0.0]

    imageKLsansAxe2 = np.copy(image)

    vecMoy =[[MoyR], [MoyG], [MoyB]] 

    for i in range(len(image)):
        for j in range(len(image[0])):
            vecTemp=[[image[i][j][0]], [image[i][j][1]], [image[i][j][2]]]
            imageKLsansAxe2[i][j][:] = np.reshape(np.dot(eigvecsansAxe2,np.subtract(vecTemp,vecMoy)),(3))
                                          

    invEigvecsansAxe2 = LA.pinv(eigvecsansAxe2);

    vecMoy =[MoyR, MoyG, MoyB] 
    imageRGBsansAxe2 = np.copy(image)

    for i in range(len(image)):
        for j in range(len(image[0])):
            #b=inv(M)a
            vecTempsansAxe2=[[imageKLsansAxe2[i][j][0]], [imageKLsansAxe2[i][j][1]], [imageKLsansAxe2[i][j][2]]]     
            imageRGBsansAxe2[i][j][:] = np.add(np.reshape(np.dot(invEigvecsansAxe2,vecTempsansAxe2),(3)),vecMoy)


    return imageRGBsansAxe2

def ReadImage(imageName):
    imagelue = py.imread('./data/' + imageName + ".png")
    return imagelue.astype('double')

def ReadTransformedImage(imageName, quantificatorID):
    imagelue = py.imread("data_transform/" + imageName + "_" + str(quantificatorID) + ".png")
    return imagelue.astype('double')

def SaveImage(image, imageName, quantificatorID):
    py.figure(figsize = (10,10))
    imageout = np.clip(image,0, 1)
    imageout= imageout.astype('double')
    py.imsave("data_transform/" + imageName + "_" + str(quantificatorID) + ".png", imageout)

def ApplyQuantificator(image, quantificator):
    # On applique les quantificateur 
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j][0] = (image[i][j][0] * quantificator[0]/255)
            image [i][j][1] = (image[i][j][1] * quantificator[1]/255)
            image[i][j][2] = (image[i][j][2] * quantificator[2]/255)


def RGB2YUV(image):
    imageYUV = image

    for i in range(len(image)):
        for j in range(len(image[0])):
            rgbVec = image[i][j]
            Y = rgbVec[0] * 0.299 + rgbVec[1] * 0.587 + rgbVec[2] * 0.114 
            U = 0.492 * (rgbVec[2] - Y)
            V = 0.877 * (rgbVec[0] - Y)
            imageYUV[i][j][0] = Y
            imageYUV[i][j][1] = U
            imageYUV[i][j][2] = V
    return imageYUV

# Généré avec une IA générative
def YUV2RGB(imageYUV):
    imageRGB = imageYUV

    for i in range(len(imageYUV)):
        for j in range(len(imageYUV[0])):
            yuvVec = imageYUV[i][j]
            Y = yuvVec[0]
            U = yuvVec[1]
            V = yuvVec[2]

            R = Y + 1.140 * V
            G = Y - 0.395 * U - 0.581 * V
            B = Y + 2.032 * U

            R = max(0, min(255, R))
            G = max(0, min(255, G))
            B = max(0, min(255, B))

            imageRGB[i][j][0] = R
            imageRGB[i][j][1] = G
            imageRGB[i][j][2] = B

    return imageRGB

def QualityCalculator(image, transformImage):
    psnrImage = peak_signal_noise_ratio(image, transformImage)
    ssimImage = ssim(image, transformImage)
    return psnrImage, ssimImage

def ComputeImage(imageName):

    quantificator  = [[255, 255, 255],[255, 255, 15],[255, 255, 0],[255, 15, 15]]
    imageNameVec = []
    # tauxDeCompressionVec = []
    psnrVec = []
    ssimVec = []

    imageRGB = ReadImage(imageName)
    transformImageRGBSansQuantificateur = Kl(imageRGB)

    for i in range(len(quantificator)):
        imageAvecQuantificateur = np.copy(transformImageRGBSansQuantificateur)
        ApplyQuantificator(imageAvecQuantificateur, quantificator[i])

        # Sauvegarde de l'image
        imageNameSave = imageName + "RGB"
        imageNameVec.append(imageNameSave[5:] + "_" + str(i))
        SaveImage(imageAvecQuantificateur, imageNameSave, i)

        # Calcul de la qualité de la transformation
        psnr, ssimImage = QualityCalculator(imageRGB, imageAvecQuantificateur)
        psnrVec.append(psnr)
        ssimVec.append(ssimImage)

        # Calcul de compressions
        # imageTransformed = ReadTransformedImage(imageNameSave, i)
        # ogImageSize = sys.getsizeof(imageRGB)
        # transforedImageSize = sys.getsizeof(imageTransformed)
        # tauxDeCompression = 1 - (transforedImageSize/ogImageSize)
        # tauxDeCompressionVec.append(tauxDeCompression)
        # print(imageNameSave + " Quantificateur " + str(i))
        # print(f'OG:{ogImageSize}')
        # print(f'Transformed:{transforedImageSize}')
        # print(f'Taux de compresssion:{tauxDeCompression}')

    # Transforme l'image RGB en YUV
    imageYUV = RGB2YUV(imageRGB)

    # Applique la transformation KL
    transformImageYUVSansQuantificateur = Kl(imageYUV)

    for i in range(len(quantificator)):

        imageYUVSansQuantificateurCopy = np.copy(transformImageYUVSansQuantificateur)
        # Applique les quantificateurs
        ApplyQuantificator(imageYUVSansQuantificateurCopy, quantificator[i])
        imageAvecQuantificateur = YUV2RGB(imageYUVSansQuantificateurCopy)

        # Calcul SSIM et PSNR
        QualityCalculator(imageRGB, imageAvecQuantificateur)
        
        # Save image
        imageNameSave = imageName + "YUV"
        imageNameVec.append(imageNameSave[5:] + "_" + str(i))
        SaveImage(imageAvecQuantificateur, imageNameSave, i)

        # Calcul de la qualité de la transformation
        psnr, ssimImage = QualityCalculator(imageRGB, imageAvecQuantificateur)
        psnrVec.append(psnr)
        ssimVec.append(ssimImage)
        # Calcul de compressions
        # imageTransformed = ReadTransformedImage(imageNameSave, i)
        # ogImageSize = sys.getsizeof(imageRGB)
        # transforedImageSize = sys.getsizeof(imageTransformed)
        # tauxDeCompression = 1 - (transforedImageSize/ogImageSize)
        # tauxDeCompressionVec.append(tauxDeCompression)
        # print(imageNameSave + " Quantificateur " + str(i))
        # print(f'OG:{ogImageSize}')
        # print(f'Transformed:{transforedImageSize}')
        # print(f'Taux de compresssion:{tauxDeCompression}')
    GraphSSIM(imageNameVec, ssimVec, imageName)
    GraphPSNR(imageNameVec, psnrVec, imageName)
    return

def CalculeImage(): 
    ComputeImage("kodim01")
    ComputeImage("kodim02")
    ComputeImage("kodim05")
    ComputeImage("kodim13")
    ComputeImage("kodim23")
    return 

def GraphSSIM(imageNameVec, dataVec, imageName):
    py.bar(imageNameVec, dataVec)
    py.title("SSIM")
    py.xlabel("Image")
    py.ylabel("ssim")
    py.savefig("./graph/" + imageName + "_ssim")
    return

def GraphPSNR(imageNameVec, dataVec, imageName):
    py.bar(imageNameVec, dataVec)
    py.title("PSNR")
    py.xlabel("Image")
    py.ylabel("psnr")
    py.savefig("./graph/" + imageName + "_psnr")
    return

if __name__ == "__main__":
    CalculeImage()
