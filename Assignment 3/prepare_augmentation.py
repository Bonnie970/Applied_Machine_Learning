import os,random
import numpy as np

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36,
               40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

def create_subdir():
    #create subdirectories
    for cls in classes:
        if not os.path.exists('%d'%(cls)):
            os.makedirs('%d'%(cls))

trainfolder = "trainImages"
os.chdir(trainfolder)
create_subdir()
os.chdir('../')

valfolder = 'valImages'
if not os.path.exists(valfolder):
    os.mkdir(valfolder)
os.chdir(valfolder)
create_subdir()
os.chdir('../')

#move images based on their classes
train_fileY = 'train_y.csv'
y_train = np.loadtxt(train_fileY, delimiter=",")


l = os.listdir("%s"%trainfolder)
for file in l:
    if file.endswith(".png"):
        index = int(file.split('.')[0])
        os.rename('%s/%s'%(trainfolder,file), '%s/%d/%s'%(trainfolder,y_train[index],file))
for cls in classes:
	file_list = os.listdir('%s/%d'%(trainfolder,cls))
	total = float(len(file_list))
	nval = int((total/50000)*2500)
	for file in file_list[:nval]:       
		os.rename('%s/%d/%s'%(trainfolder,cls,file), '%s/%d/%s' % (valfolder,cls, file))



