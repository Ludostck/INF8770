import numpy as np
import matplotlib.pyplot as py
from numpy import linalg as LA, strings
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve

image_path = "./data/kodim01.png"

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

def SaveImage(image, imageName, quantificatorID):
    py.figure(figsize = (10,10))
    imageout = np.clip(image,0, 255)
    imageout= imageout.astype('uint8')
    py.imsave("data_transform/" + imageName + "_" + str(quantificatorID) + ".png", imageout)

def DoubleToUINT8ImageWithQuantifier(image, quantificator):
    # On applique les quantificateur 
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j][0] = np.uint8(image[i][j][0] * quantificator[0])
            image [i][j][1] = np.uint8(image[i][j][1] * quantificator[1])
            image[i][j][2] = np.uint8(image[i][j][2] * quantificator[2])


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

def RGB2YUV_GPT(image):
    transformation_matrix = np.array([
        [ 0.299,  0.587,  0.114 ],
        [-0.1687, -0.3313,  0.5 ],
        [ 0.5, -0.4187, -0.0813 ]
    ])
    offset = np.array([0, 128, 128])

    ycbcr_image = np.dot(image, transformation_matrix.T) + offset
    return ycbcr_image

def QualityCalculator(image, transformImage):
    print("PSNR")
    print(peak_signal_noise_ratio(image, transformImage))
    print("SSIM")
    print(ssim(image, transformImage))
    return

def ComputeImage(imageName):

    quantificator  = [[255, 255, 255],[255, 255, 15],[255, 255, 0],[255, 15, 15]]
    for i in range(len(quantificator)):
        image = ReadImage(imageName)
        imageNameSave = imageName + "RGB"
        transformImage = Kl(image)
        print(imageNameSave)
        QualityCalculator(image, transformImage)
        DoubleToUINT8ImageWithQuantifier(transformImage, quantificator[i])
        SaveImage(transformImage, imageNameSave, i)

    for i in range(len(quantificator)):
        image = ReadImage(imageName)
        image = RGB2YUV(image)
        imageNameSave = imageName + "YUV"
        transformImage = Kl(image)
        print(imageName)
        QualityCalculator(image, transformImage)
        DoubleToUINT8ImageWithQuantifier(transformImage, quantificator[i])
        SaveImage(transformImage, imageNameSave, i)

if __name__ == "__main__":
    ComputeImage("kodim01")
    ComputeImage("kodim02")
    ComputeImage("kodim05")
    ComputeImage("kodim13")
    ComputeImage("kodim23")















