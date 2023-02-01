import tensorflow as tf
from tensorflow.python.client import device_lib
import torch

# tf
print('\n\n************** TF CHECK **************\n\n')
print("tf.test.is_gpu_available()", tf.test.is_gpu_available())
print("tf.test.is_built_with_cuda()", tf.test.is_built_with_cuda())
print("tf device", device_lib.list_local_devices())

# torch
print('\n\n************** Pytorch CHECK **************\n\n')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])