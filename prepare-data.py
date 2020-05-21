# =============================================================================
# Import all packages
# =============================================================================
import cv2
import numpy as np
import glob
import argparse

# =============================================================================
# Instantiate command line argument parser
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--cat_embeddings", help="Path to ImageEm vector for cat", default="npy/cat.npy")
parser.add_argument("--dog_embeddings", help="Path to ImageEm vector for dog", default="npy/dog.npy")
parser.add_argument("--cat_path", help="Path to training set of cat images", default="../demos/cat-and-dog/dataset/training_set/cats/")
parser.add_argument("--dog_path", help="Path to training set of dog images", default="../demos/cat-and-dog/dataset/training_set/dogs/")
parser.add_argument("--cat-file", help="Path to where cat dataset should be saved", default="cat_data.csv")
parser.add_argument("--dog-file", help="Path to where dog dataset should be saved", default="dog_data.csv")
args = parser.parse_args()

# =============================================================================
# Import cat and dog embeddings
# =============================================================================
cat_embeddings = np.load(args.cat_embeddings)
dog_embeddings = np.load(args.dog_embeddings)

# =============================================================================
# Define the dot product function
# =============================================================================
def dot_product_image(im_path, embedding, file, classnum):
    im = cv2.imread(im_path)
    im = cv2.resize(im, (64, 64))
    im = np.ravel(im)
    dots = []
    for i in range(100, im.shape[0], 100):
        chunk = im[i - 100:i]
        dots.append(np.dot(chunk, embedding))
    with open(file, 'a+') as file:
        for dot in dots:
            file.write(str(dot))
            file.write(",")
        file.write(classnum)
        file.write("\n")

# =============================================================================
# Load cat and dog image data
# =============================================================================
cats_training_path = args.cat_path
dogs_training_Path = args.dog_path

cat_pics = glob.glob(cats_training_path + "*.jpg")
print("Number of cat pics = ", len(cat_pics))

dog_pics = glob.glob(dogs_training_Path + "*.jpg")
print("Number of dog pics = ", len(dog_pics))

# =============================================================================
# Prepare data using pre-trained ImageEm vectors
# =============================================================================
for i, cat_pic in enumerate(cat_pics):
    dot_product_image(cat_pic, cat_embeddings, args.cat_file, "0")

for i, dog_pic in enumerate(dog_pics):
    dot_product_image(dog_pic, dog_embeddings, args.dog_file, "1")
