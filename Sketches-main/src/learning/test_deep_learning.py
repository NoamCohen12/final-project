import kagglehub

# Download latest version
path = kagglehub.dataset_download("luischuquimarca/banana-ripeness-images-datasets")

print("Path to dataset files:", path)