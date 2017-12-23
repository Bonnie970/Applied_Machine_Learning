import numpy as np
from PIL import Image
import os
	
trainImgfolder = 'trainImages'
if not os.path.exists(trainImgfolder):
    os.mkdir(trainImgfolder)
x = np.loadtxt("train_x.csv", delimiter=",", dtype = np.uint8)
x = x.reshape(-1, 64, 64)

y = np.loadtxt("train_y.csv", delimiter=",", dtype = np.uint8)
os.chdir(trainImgfolder)

for index,img in enumerate(x):
    result = Image.fromarray(img,'L')
    filename = str(index) + ".png"
    label = y[index]
    result.save(str(label) + '/' + filename)
os.chdir("..")

testImgfolder = 'testImages'
if not os.path.exists(testImgfolder):
    os.mkdir(testImgfolder)
x = np.loadtxt("test_x.csv", delimiter=",", dtype = np.uint8) # load from text
x = x.reshape(-1, 64, 64)
os.chdir(testImgfolder)
for index,img in enumerate(x):
    result = Image.fromarray(img,'L')
    filename = str(index) + ".png"
    result.save(filename)
