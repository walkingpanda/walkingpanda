import numpy as np
import torch
from torch import nn

def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0

def fizz_buzz_decode(i, label):
    return [str(i), 'fizz', 'buzz', 'fizzbuzz'][label]


def helper(i):
    print(fizz_buzz_decode(i, fizz_buzz_encode(i)))


for i in range(1, 16):
    helper(i)

NUM_DIGITS = 10
def binary_encode(i, NUM_DIGITS): # 将一个十进制数转换为二进制
    return np.array([i >> d & 1 for d in range(NUM_DIGITS)][::-1])

#print(binary_encode(15, NUM_DIGITS))