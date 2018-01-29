
# coding: utf-8

# In[ ]:


'''
PURPOSE: To build a classifier using Linear Regression to distinguish images of two actors.
'''


# In[582]:


import numpy as np 
import matplotlib.pyplot as plt
from random import randint


# In[675]:


#load data sets containing only 'Alec Baldwin' and 'Steve Carel'
x_train = np.load("x_train0.npy") 
y_train = np.load("y_train0.npy")

x_val = np.load("x_val0.npy")
y_val = np.load("y_val0.npy")

x_test = np.load("x_test0.npy")
y_test = np.load("y_test0.npy")


# In[264]:


def replace_labels(y,labels):
    y_relabeled = np.copy(y)
    for label in labels:
        for index in np.where(y == label[0]):
            np.put(y_relabeled, index, label[1])
    return y_relabeled.astype(int)

#change output labels to 0 and 1


# In[265]:


def flatten_set(x):
    #returned ndarray should have shape (N, M), where N = # pixels and M = # images
    for i in range(x.shape[-1]):
        flattened_image = x[...,i].flatten() 
        if i == 0:
            x_flattened = flattened_image
        else:
            x_flattened = np.vstack((x_flattened, flattened_image))
            
    return x_flattened.T


# In[397]:


def cost(x,y,theta):
    #quadratic cost function
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return np.sum( (y - np.dot(theta.T,x)) ** 2)
    


# In[301]:


def dcost_dtheta(x,y,theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return -2*np.sum((y-np.dot(theta.T, x))*x, 1)



# In[474]:


def grad_descent(cost, dcost_dtheta, x, y, init_theta, alpha,max_iter):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_theta-10*EPS
    t = init_theta.copy()
    iter  = 1
 
    while np.linalg.norm(t - prev_t) >  EPS and iter < max_iter:
#        print(np.linalg.norm(t - prev_t), max_iter)
        prev_t = t.copy()
        t -= alpha*dcost_dtheta(x, y, t)
#         if iter % int(max_iter/10) == 0:
#             print "Iter", iter
#             print "x = (%.2f, %.2f, %.2f), cost(x) = %.2f" % (t[0], t[1], t[2], cost(x, y, t)) 
#             print "Gradient: ", dcost_dtheta(x, y, t), "\n"
        iter += 1
    return t


# In[758]:


def pred_y(x,theta):

    x = np.vstack((np.ones((1, x.shape[1])), x ))    
    h_all = np.dot(theta.T,x)
    
    y_pred = np.ones(h_all.shape[0])
    
    for i in range(h_all.shape[0]):
        h=h_all[i]
        if h > 0.5:
            y_pred[i] = 1
        elif h < 0.5:
            y_pred[i] = 0
        else:
            y_pred[i]=randint(0,1)
    return y_pred


# In[757]:


def performance(y_pred, y):
    boolean = y_pred == y_test
    num_correct = len(np.where(boolean==True)[0])
    return num_correct / float(y.shape[0]) * 100.0


# In[676]:


y_train = replace_labels(y_train, [("Alec Baldwin",1), ("Steve Carell",0)])
y_val = replace_labels(y_val, [("Alec Baldwin",1), ("Steve Carell",0)])
y_test = replace_labels(y_test, [("Alec Baldwin",1), ("Steve Carell",0)])


# In[677]:


x_train = flatten_set(x_train) / 255.0
x_val = flatten_set(x_val) / 255.0
x_test = flatten_set(x_test) / 255.0


# In[761]:


#initialize from normal distribution
pixel_inten_mean = np.mean(x_train)
pixel_inten_std  = np.std(x_train)
theta0 = np.random.normal( 0, 1, x_train.shape[0]+1) #of dimension (1025,)


# In[765]:


for i in range(0,20):
    theta = grad_descent(cost, dcost_dtheta, x_train, y_train, theta0, 0.000001,i)
    if i%10 == 0:
        y_pred = pred_y(x_val,theta)
        print(performance(y_val,theta))

