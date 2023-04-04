import matplotlib.pyplot as plt
import cv2

def histshow(inimage, compositedImage):
    '''
    Takes in two 2d array, plot the histogram of them. 
    
    Args:
        inimage(float): input image (BRG)
        compositedImage(float): composited image (BGR)
    
    Returns:
        plot two histograms in one figure.
    '''
    # convert BGR to YUV
    inimage1= cv2.cvtColor(inimage, cv2.COLOR_BGR2YUV) 
    compositedImage1=cv2.cvtColor(compositedImage, cv2.COLOR_BGR2YUV)
    
    # calculate histogram only using Y channel 
    hist_in = cv2.calcHist([inimage1],[0],None,[256],[0,256])
    hist_co = cv2.calcHist([compositedImage1],[0],None,[256],[0,256]) 
    
    # plot the above computed histogram
    plt.figure(1)    
    plt.subplot(1, 2, 1)
    plt.plot(hist_in, color='b')
    plt.title('Histogram of input image ')

    plt.figure(1)    
    plt.subplot(1, 2, 2)
    plt.plot(hist_co, color='b')
    plt.title('Histogram of composited Image')
    plt.show()

inimage = cv2.imread("C:/Users/aduttagu/Downloads/IMG_9091.jpg")
compositedImage = cv2.imread("C:/Users/aduttagu/Downloads/IMG_9093.jpg")

hist = histshow(inimage, compositedImage)