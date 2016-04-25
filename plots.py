import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file containing data to generate the plot"
    exit(0)
    
data = pickle.load(open(fname, 'rb')) 
test_score = data['test_score']
test_accuracy = data['test_accuracy']
hist = data['hist']    
execution_time = data['execution_time']    
    
print (hist)    
