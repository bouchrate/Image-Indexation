import numpy as np
from skimage import io, color, feature
import os
def compute_hog(image):
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)
    
    # Compute HOG features
    hog_features, hog_image = feature.hog(gray_image, visualize=True)
    
    return hog_features

def save_hog_vectors(dataset_path, save_path):
    # Create an empty dictionary to store HOG vectors with IDs and class labels
    hog_dict = {}

    # Loop through each class in the dataset
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            # Loop through each image in the class
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # Extract the ID from the filename (adjust as needed based on your naming convention)
                    image_id = os.path.splitext(filename)[0]

                    # Load the image
                    image_path = os.path.join(class_path, filename)
                    image = io.imread(image_path)

                    # Compute HOG vector for the image
                    hog_vector = compute_hog(image)
                    print(hog_vector)

                    # Get the class label from the folder name
                    class_label = class_folder

                    # Add the HOG vector to the dictionary with the image ID as the key
                    # The value is a tuple containing the HOG vector, class label, and image ID
                    hog_dict[image_id] = (hog_vector, class_label)

    # Save the HOG vectors dictionary to a file
    np.save(save_path, hog_dict)

    

# Set the path to your dataset and where you want to save the HOG vectors
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
dataset_path = "C:/Users/TERMASS BOUCHRA/Desktop/Project/Flower-classification/flower"
save_path = "C:/Users/TERMASS BOUCHRA/Desktop/Project/Flower-classification/hog_vectors_by_class.npy"

# Call the function to compute and save HOG vectors organized by class
save_hog_vectors(dataset_path, save_path)

# Load the HOG vectors dictionary from the file
loaded_hog_dict = np.load(save_path, allow_pickle=True).item()

print("Available Image IDs:", list(loaded_hog_dict.keys()))
print("Available Image IDs n:", list(loaded_hog_dict.values()))

