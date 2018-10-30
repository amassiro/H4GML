from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout

import ROOT as root
import numpy as np

np.random.seed(1234)



def getDataFromFile(fileinfo, branchlist) :

  tfil = root.TFile(fileinfo)
  tree = tfil.Get('H4GSel')
  datasize = tree.GetEntriesFast()
  
  print "Reading NN inputs from " + fileinfo
  print "    total = ", datasize
  
  data = np.empty([datasize, len(branchlist)])
  counter = 0
  for entry in tree :
    index_variable = 0
    if not (counter%1000) : print " counter = ", counter
    for branch in branchlist :
      data[counter][index_variable] = getattr(entry, branch)
      index_variable = index_variable + 1

    counter = counter + 1
  
  tfil.Close()

  return data




def getDataFromFileWithCut(fileinfo, branchlist, cut_variable, cut_value) :

  tfil = root.TFile(fileinfo)
  tree = tfil.Get('H4GSel')
  
  datasize = tree.GetEntries(cut_variable + " == " +  str(cut_value))
  
  print "Reading NN inputs from " + fileinfo
  data = np.empty([datasize, len(branchlist)])
  counter = 0
  for entry in tree :
    index_variable = 0
    
    # check cut
    if getattr(entry, cut_variable) == cut_value :
    
      for branch in branchlist :
        data[counter][index_variable] = getattr(entry, branch)
          
        index_variable = index_variable + 1
     
      counter = counter + 1
  
  tfil.Close()

  print " counter = ", counter 
  
  return data






def getDataFromFileWithCutGreater(fileinfo, branchlist, cut_variable, cut_value) :

  tfil = root.TFile(fileinfo)
  tree = tfil.Get('H4GSel')

  datasize = tree.GetEntries(cut_variable + " >= " +  str(cut_value))
 
  print "Reading NN inputs from " + fileinfo
  data = np.empty([datasize, len(branchlist)])
  counter = 0
  for entry in tree :
    index_variable = 0
    
    # check cut
    if getattr(entry, cut_variable) >= cut_value :
    
      for branch in branchlist :
        data[counter][index_variable] = getattr(entry, branch)
        index_variable = index_variable + 1
     
      counter = counter + 1
  
  tfil.Close()

  print " counter = ", counter 
  
  return data




#######################################
#
# Start here
#

#sigfile = 'data/signal_skim_m_10.root.train.root'
sigfile = 'all.root'
 
brlist  = ['p1_pt', 'p2_pt', 'p3_pt', 'p4_pt',
           'p1_eta', 'p2_eta', 'p3_eta', 'p4_eta',
           'p1_phi', 'p2_phi', 'p3_phi', 'p4_phi',
           #
           'p1_r9', 'p2_r9', 'p3_r9', 'p4_r9',
           #
           #
           'p12_mass', 'p13_mass', 'p14_mass', 'p23_mass', 'p24_mass', 'p34_mass',
           'p12_dr', 'p13_dr', 'p14_dr', 'p23_dr', 'p24_dr', 'p34_dr',
           #
           #
           'best_combination'
           #
           #
           ]

numvars = len(brlist)
ntrain  =  0.50   # 80% for train
ntest   =  1 - ntrain

data = getDataFromFile (sigfile, brlist)

print " sig data.size = " , data.size , " ---> ", data.size/numvars

datasize = data.size/numvars


print "data =", data

print "Splitting data between " + str(int(len(data)*ntrain)) + " training and " + str(int(len(data)*ntest)) + " testing samples sig ..."

temp_data_train = data[ : int(len(data)*ntrain)]
temp_data_test  = data[ : int(len(data)*ntrain)]

#
# s[i:j:k]  slice of s from i to j with step k
#
# 
#  c = [a[j] [0:3:1] for j in range (len(a)) ]
#
#

# brlist = varialbe + flag (target)
# in the next lines the long list of list is divided such that the "flag" becomes "labels" list

list_data_train = [ temp_data_train[j] [0:(numvars-1):1] for j in range (len(temp_data_train)) ]
list_data_test  = [ temp_data_test [j] [0:(numvars-1):1] for j in range (len(temp_data_test )) ]

list_label_data_train = [ temp_data_train[j] [(numvars-1):numvars:1] for j in range (len(temp_data_train)) ]
list_label_data_test  = [ temp_data_test [j] [(numvars-1):numvars:1] for j in range (len(temp_data_test )) ]



data_train = np.vstack( [list_data_train] )
data_test  = np.vstack( [list_data_test ] )

label_data_train = np.vstack( [list_label_data_train] )
label_data_test  = np.vstack( [list_label_data_test ]  )





print " data_train   = ", data_train
print " label_data_train = ", label_data_train


## define model

model = Sequential()
model.add(Dense(50, input_dim = numvars-1, kernel_initializer='normal', activation='relu'))
model.add(Dense(30, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()

# train model
history= model.fit(    data_train, label_data_train, 
                       batch_size=len(data_train)/8,
                       epochs=20,
                       #epochs=200,
                       shuffle=True, 
                       validation_data = (data_test, label_data_test) 
                   )

# save model
model.save("h4g_NN_category.h5")




#########################################################
## plot

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D as plt3d

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white' 


#########################################################
## validation plot

fig_validation = plt.figure(1, figsize=(12,12))

epochs = range(1, len(history.history["loss"])+1)
plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
plt.plot(epochs, history.history["val_loss"], lw=3, label="Validation loss")
plt.xlabel("Epoch"), plt.ylabel("Cross-entropy loss");



#########################################################
## plot structure of NN

from keras.utils import plot_model, print_summary
print_summary(model)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)




##########################################################

labels_predicted_keras = model.predict(data_test).ravel()


labels_difference = [labels_predicted_keras[j] - label_data_test[j][0] for j in range(len(label_data_test)) ]

print " labels_predicted_keras = ", labels_predicted_keras

#print " labels_difference = ", labels_difference

plt.figure(2)
plt.hist(labels_difference, normed=True, bins=30)
plt.ylabel('reg-true');



labels_difference_0 = [labels_predicted_keras[j] - label_data_test[j][0]    for j in range(len(label_data_test))  if label_data_test[j][0] == 0    ]
labels_difference_1 = [labels_predicted_keras[j] - label_data_test[j][0]    for j in range(len(label_data_test))  if label_data_test[j][0] == 1    ]
labels_difference_2 = [labels_predicted_keras[j] - label_data_test[j][0]    for j in range(len(label_data_test))  if label_data_test[j][0] == 2    ]



plt.figure(3)

plt.subplot(1,3,1)
plt.hist(labels_difference_0, normed=True, bins=30)
plt.xlabel('reg-true A');

plt.subplot(1,3,2)
plt.hist(labels_difference_1, normed=True, bins=30)
plt.xlabel('reg-true B');

plt.subplot(1,3,3)
plt.hist(labels_difference_2, normed=True, bins=30)
plt.xlabel('reg-true C');




plt.figure(4)

labels_distribution = [ label_data_test[j][0]    for j in range(len(label_data_test))  ]

plt.hist(labels_distribution, normed=True, bins=30)
plt.xlabel('true best combination');



#########################################################
## plots only at the end

plt.show()

