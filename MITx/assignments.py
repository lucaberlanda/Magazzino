import numpy as np


def scalar_function(x, y):
    # Your code here
    if x <= y:
        return x * y
    else:
        return x / y


def vector_function(x, y):
    vfunc = np.vectorize(scalar_function)
    return vfunc(x, y)


x = np.array([4, 6])
y = np.array([6, 3])
print(vector_function(x, y))
quit()

inputs = np.array([3, 4])
weights = np.array([0.3, 0.7])
out = np.tanh(np.matmul(np.transpose(inputs), weights))
out2 = np.tanh(np.matmul(inputs, weights))
print(out, out2)
quit()
h = 3
w = 4
A = np.random.random([h, w])
B = np.random.random([h, w])
s = A + B
print(A.shape, B.shape, s.shape)
print(A, B, s)

s = np.linalg.norm(A + B)

print(s)

matmul
