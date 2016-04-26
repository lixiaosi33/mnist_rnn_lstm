import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1:]
else:
    print "Require a file containing data to generate the plot"
    exit(0)


def draw_plot(attribute_list, ylabel):

    x = range(1, len(attribute_list)+1, 1)
    plt.plot(x, attribute_list)
    plt.xlabel('number of epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    


# train set accuracy
plt.figure(facecolor='white')
for file_name in fname:
    data = pickle.load(open(file_name, 'rb')) 
    history = data['history'] 

    # list of accuracy values over the epochs
    accuracy = history['acc']  
    draw_plot(accuracy, 'train set accuracy')
plt.show()



# train set loss
plt.figure(facecolor='white')
for file_name in fname:
    data = pickle.load(open(file_name, 'rb')) 
    history = data['history'] 

    # list of loss values over the epochs
    loss = history['loss']
    draw_plot(loss, 'train set loss')
plt.show()



# test set accuracy
plt.figure(facecolor='white')
for file_name in fname:
    data = pickle.load(open(file_name, 'rb')) 
    history = data['history'] 

    # list of validation accuracy over the epochs
    val_acc = history['val_acc']
    draw_plot(val_acc, 'test set accuracy')
plt.show()



# test set loss
plt.figure(facecolor='white')
for file_name in fname:
    data = pickle.load(open(file_name, 'rb')) 
    history = data['history'] 

    # list of validation loss values over the epochs
    val_loss = history['val_loss']
    draw_plot(val_loss, 'test set loss')
plt.show()






