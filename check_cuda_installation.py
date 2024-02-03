import torch

# Generate a random tensor of shape (5, 3)
random_tensor = torch.rand(5, 3)
print("Random Tensor:")
print(random_tensor)

# Print the version of the torch library
torch_version = torch.__version__
print("Torch Version:", torch_version)

# Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)