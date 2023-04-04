import torch

# create tensor1, tensor2, tensor3
tensor1 = torch.randint(0, 2, (2, 7))
tensor2 = torch.randint(0, 2, (2, 10))
tensor3 = torch.randint(0, 2, (2, 10))

# compute the Cartesian product of the indices of the three tensors
idx = torch.cartesian_prod(torch.arange(tensor1.size(1)),
                           torch.arange(tensor2.size(1)),
                           torch.arange(tensor3.size(1)))

# initialize the result tensor with zeros
result = torch.zeros(2, idx.shape[0])

# compute the product of the corresponding elements of the tensors
for i in range(idx.shape[0]):
    result[:, i] = tensor1[:, idx[i, 0]] * tensor2[:, idx[i, 1]] * tensor3[:, idx[i, 2]]

print(result)
