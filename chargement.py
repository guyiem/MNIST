#coding: utf-8

#import torch
import numpy as npy
from torchvision import datasets, transforms

if len(sys.argv)<=1:
    # if no argument given my default repertory for stocking MNIST dataset 
    rep = '../../donnees/MNIST'
else:    
    rep = sys.argv[1]

train_dataset0 = datasets.MNIST(rep,download=False)
train_dataset1 = datasets.MNIST(rep, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
# two calls at datasets.MNIST with two ways, in the second case we use the transform with "to tensor" and "normalize"


print(' \n ---- STUDY OF THE FIRST DATASET ----')
print('--- what is a dataset ----')
print("type : ",type(train_dataset0)) # well ok... docs tells us that dataset as a length
print("length of dataset : ",len(train_dataset0)) # 6000, wich corresponds to the length of the training set MNIST.
print('--- end of what is a dataset ----')
print('--- what is IN a dataset ----') # we see in doc there is a __get_item__ method, so let's use it
el0 = train_dataset0[0]
print("type :",type(el0)) # we see here it's a tuple  so we check the length
print("length :",len(el0)) # we see there is two elements for both so we check there types
print(" type of the two elements for train_dataset0 : ",type(el0[0]),type(el0[1])) # we see there is a PIL image (https://en.wikipedia.org/wiki/Python_Imaging_Library) and a tensor. We print tensor shape
print( el0[1].size() ) # It seems there is only one element in the tensor, so probably the value of the number printed on the image. So let'plot the image and check tensor value.
#el0[0].show()
print(el0[1]) # so we have a collection of digit image with a label giving the corresponding value.
print('--- end of what is IN a dataset ----')
print(' ---- END STUDY OF THE FIRST DATASET ---- \n')

print(' \n ---- STUDY OF A SECOND DATASET ---- ')
train_dataset1 = datasets.MNIST(rep, download=False,transform=transforms.ToTensor()) # we just try ToTensor(), and doing same code as before, we see we now have a tensor instead of an image
print('--- what is a dataset ----')
print("type : ",type(train_dataset1)) # well ok... docs tells us that dataset as a length
print("length of dataset : ",len(train_dataset1)) # 6000, wich corresponds to the length of the training set MNIST.
print('--- end of what is a dataset ----')
print('--- what is IN a dataset ----') # we see in doc there is a __get_item__ method, so let's use it
el1 = train_dataset1[0]
print("type :",type(el1)) # we see here it's a tuple  so we check the length
print("length :",len(el1)) # we see there is two elements for both so we check there types
print(" type of the two elements for train_dataset0 : ",type(el1[0]),type(el1[1])) # two tensors this time -> image was converted to a tensor
print('--- end of what is IN a dataset ----')
images_flatten = npy.array([elem[0].numpy().flatten()  for elem in train_dataset1])
print(' means and std :',npy.mean(images_flatten),npy.std(images_flatten))
print(' ---- END STUDY OF THE SECOND DATASET ---- \n')



print('--- what is a dataset ----')
print("type : ",type(train_dataset0),type(train_dataset1)) # Both have same type, seems natural ! we see on the following link that this object as a length, so we test it (https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset )
print("length of dataset : ",len(train_dataset0),len(train_dataset1)) # both have same length, seems natural.
print('--- end of what is a dataset ----')

print('--- what is IN a dataset ----') # we see in doc there is a __get_item__ method, so let's use it
el0 = train_dataset0[0]
el1 = train_dataset1[0]
print("type :",type(el0),type(el1)) # we see here it's a tuple for both way,  so we check the length
print("length :",len(el0),len(el1)) # we see there is two elements for both, which seems quite natural. so we check there types, first for train_dataset0
print(" type of the two elements for train_dataset0 : ",type(el0[0]),type(el0[1])) # we see there is a PIL image (https://en.wikipedia.org/wiki/Python_Imaging_Library) and a tensor. We print tensor shape
print( el0[1].size() ) # It seems there is only one element in the tensor, so probably the value of the number printed on the image. So let'plot the image and check tensor value.
el0[0].show()
print(el0[1]) # so we have a collection of digit image with a label giving the corresponding value.
print(" type of the two elements for train_dataset1 : ",type(el1[0]),type(el1[1])) # here we have two tensors. So transforms.ToTensor() transforms the PIL image to a tensor.
print('--- end of what is IN a dataset ----')

print(' --- computation of the mean and the std of the original dataset --- ')

print('--- visual comparaison of the normalisation effect ----')
el1 = train_dataset1[0]
print(type(el1[1]))
print('--- visual comparaison of the normalisation effect ----')
