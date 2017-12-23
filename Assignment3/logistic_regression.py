from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression(class_weight='balanced',multi_class='ovr')

#read train x and y
x = np.loadtxt('old/10k_processed_train_x_chunk0', delimiter=",")
x /= 255
y = np.loadtxt('old/10k_processed_train_y_chunk0', delimiter=",")
#split to validation and train
nval = 2500
print('training ...')
model.fit(x[nval:],y[nval:])
print('evaluating ...')
print(model.score(x[:nval],y[:nval]))

testX = np.loadtxt('data/test_x.csv', delimiter=",")
testY = model.predict(testX)

trial = 'logistic0'
print('Saving prediction results to test_trial_{}.csv'.format(trial))
with open('test_trial_{}.csv'.format(trial), 'w') as f:
    f.write('Id,Label\n')
    id = 1
    for y in testY:
        f.write('{},{}\n'.format(id, y))
        id += 1
f.close()