import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file containing data to generate the plot"
    exit(0)
 
def draw_plot(attribute_list):

    x = range(1, len(attribute_list)+1, 1)
    plt.figure()
    plt.plot(x, attribute_list)
    plt.show()

data = pickle.load(open(fname, 'rb')) 
test_score = data['test_score']
test_accuracy = data['test_accuracy']
history = data['history'] 
execution_time = data['execution_time']    

# list of accuracy values over the epochs
accuracy = history['acc']  
draw_plot(accuracy)

# list of loss values over the epochs
loss = history['loss']
draw_plot(loss)

# list of validation accuracy over the epochs
val_acc = history['val_acc']
draw_plot(val_acc)

# list of validation loss values over the epochs
val_loss = history['val_loss']
draw_plot(val_loss)
