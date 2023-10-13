import torch
import pandas as pd
import numpy as np
import matplotlib_inline as plt


sclar = torch.tensor(7)
print(sclar)
print(sclar.ndim)

vector = torch.tensor([7,7])
print(vector)
print(vector.ndim)

zeros = torch.zeros([3, 3])
print(zeros)
print(zeros.ndim)
print(zeros.shape)

random = torch.rand(3, 4)

print("random number", random.shape, random.ndim)

random_img = torch.rand(size=(244, 244, 3))

print("random number img", random_img.shape, random_img.ndim)


range = torch.arange(0,10)

range = torch.arange(start=0, end=1000, step=100)

print(range)

print("##"*100)

#  Multiplaction

tensor = torch.tensor([1, 3, 6])

print("Tensor", tensor)

print("Final value", tensor * tensor)

print("##"*100)

# Matrix Multiplation
final = torch.matmul(tensor, tensor)

print("Matrix Multiplation Value:", final)
