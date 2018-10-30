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
  data = np.empty([datasize, len(branchlist)])
  counter = 0
  for entry in tree :
    index_variable = 0
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

sigfile = 'data/signal_skim_m_10.root.train.root'
 
brlist  = ['p1_pt', 'p2_pt', 'p3_pt',
           'p1_eta', 'p2_eta', 'p3_eta',
           'p1_phi', 'p2_phi', 'p3_phi',
           #
           #
           'p12_mass', 'p13_mass', 'p14_mass', 'p23_mass', 'p24_mass', 'p34_mass',
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
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# train model
history= model.fit(    data_train, label_data_train, 
                       batch_size=len(data_train)/8,
                       epochs=100,
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
## plot samples


#range_ = ( (0, 10), (0, 10) )
#fig = plt.figure(0, figsize=(12,12))

#plt.subplot(2,2,1)
#plt.title("Signal")
#plt.xlabel("layer 0")
#plt.ylabel("layer 1")
#plt.hist2d(np.array(data_sig_train)[:,0], np.array(data_sig_train)[:,1], range=range_, bins=20, cmap=cm.coolwarm)

#plt.subplot(2,2,2)
#plt.title("Background")
#plt.hist2d(np.array(data_bkg_train)[:,0], np.array(data_bkg_train)[:,1], range=range_, bins=20, cmap=cm.coolwarm)
#plt.xlabel("layer 0")
#plt.ylabel("layer 1")

##plt.subplot(2,2,3)
##plt.title("Signal")
##plt.xlabel("layer 2")
##plt.ylabel("layer 3")
##plt.hist2d(np.array(data_sig_train)[:,2], np.array(data_sig_train)[:,3], range=range_, bins=20, cmap=cm.coolwarm)

##plt.subplot(2,2,4)
##plt.title("Background")
##plt.hist2d(np.array(data_bkg_train)[:,2], np.array(data_bkg_train)[:,3], range=range_, bins=20, cmap=cm.coolwarm)
##plt.xlabel("layer 2")
##plt.ylabel("layer 3")



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
#### ROC curve

#from sklearn.metrics import roc_curve

#labels_predicted_keras = model.predict(data_test).ravel()
#false_positive_rates_keras, true_positive_rates_keras, thresholds_keras = roc_curve(label_data_test, labels_predicted_keras)

##labels_predicted_keras = model.predict(data_train).ravel()
##false_positive_rates_keras, true_positive_rates_keras, thresholds_keras = roc_curve(label_data_train, labels_predicted_keras)

#from sklearn.metrics import auc
#auc_keras = auc(false_positive_rates_keras, true_positive_rates_keras)

#plt.figure(3)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(false_positive_rates_keras, true_positive_rates_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='best')

## Zoom in view of the upper left corner.
#plt.figure(4)
#plt.xlim(0, 0.2)
#plt.ylim(0.8, 1)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(false_positive_rates_keras, true_positive_rates_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve (zoomed in at top left)')
#plt.legend(loc='best')



## HEP glossary
#plt.figure(5)
#plt.plot([0, 1], [1, 0], 'k--')
#false_positive_one_minus_rates_keras = [ (1-value) for value in false_positive_rates_keras ]
#plt.plot(true_positive_rates_keras, false_positive_one_minus_rates_keras)
#plt.xlabel('Signal efficiency')
#plt.ylabel('Background rejection')
#plt.title('ROC curve')
#plt.legend(loc='best')



#########################################################
## plots only at the end

plt.show()


