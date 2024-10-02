import numpy as np # for math operations
import cv2 # for image reading and conversions
import matplotlib.pyplot as plt # for image display
import os # for file handling
import io # for handling Bytes data
from PIL import Image # for image data handling
from urllib.request import urlopen # to read data from a url

URL = 'https://raw.githubusercontent.com/RajkumarGalaxy/dataset/master/Images/002.jpg'
req = urlopen(URL)
file = io.BytesIO(req.read())
# convert it to grayscale
im = Image.open(file).convert('L')
# transform it into an array
im = np.asarray(im)
# display the image
plt.imshow(im, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image', c='r')
plt.show()