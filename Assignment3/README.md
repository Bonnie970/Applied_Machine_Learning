# image classification: modified mnist  

All of the code was implemented on Python 2.7

-----------------------------------------------------------------------------------------------------------------
Run Convolutional Neural Net

0. GPU is highly recommended to use to run our CNN code --> Please check GPU Setup on Goolge Compute Engine below 

1. Make sure all following Python packages installed: 

		tensorflow, keras, scipy, tensorflow-gpu, h5py, numpy, pillow 
		
2. download text_x.csv,train_x.csv,train_y.csv to your directory 

3. convert train_x and test_x from csv to png image files:
		
		python csv2png.py 
	           
4. prepare data augmentation, split train and validation into subfolders
		
		python prepare_augmentation.py
		
5. run the code of CNN to generate model file and make predictions
		
		python cnn.py
		
6. do fine tuning with the model generated from last step
		
		in cnn.py: enable_fine_tune = True
		python cnn.py  
		
7. the name of the CNN model file is xxx.h5 and final prediction is saved in XXX.csv


-----------------------------------------------------------------------------------------------------------------
GPU Setup in Google Compute Engine

check file GPUsetup.txt



-----------------------------------------------------------------------------------------------------------------
Run Neural Net

Python package required: numpy

python finalNN.py

In order to properly run the function “finalNN.py” make sure that you have the following files in the same folder:

- “train_x.csv” - a file consisting of 50000 rows each containing 4096 0s or 1s separated by commas. Every line is a training example picture after preprocessing has been done.

- “train_y.csv” - a file consisting of 50000 numbers, each on their line and each are the class that corresponds to the training example in “train_x.csv”

- “test_x.csv” - a file the same format as the first one. 

The program is after the run, creating a file with the name “result.csv”, which has the same format as “train_y.csv” but this is all the predictions of 
the pictures in “test_x.csv”. Make sure there is no file with the name “result.csv” in the same folder. 

The program trains a neural network with one hidden layer, and 12 hidden units in that layer. 


-----------------------------------------------------------------------------------------------------------------
Run Logistic Regression 

Python package required: sklearn, numpy

This code takes more than 3 hours to run

python logistic_regression.py





