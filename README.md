# Nearest_Neighbor
Nearest Neighbor handwritten digits classifier (using the MNIST dataset)
The MNIST dataset consists of images of handwritten digits, 
along with their labels. Each image has 28×28 pixels, where each pixel is in grayscale
scale, and can get an integer value from 0 to 255. Each label is a digit between 0 and 9. The
dataset has 70,000 images. Althought each image is square, we treat it as a vector of size 28×28 = 784.

### Nearest Neighbor Algorithm:
Accepts as input:
- a set of images
- a vector of labels corresponding to the images 
- a query image
- a number k 
returns a prediction of the query image, given the label set of images.
The algorithm uses the k nearest neighbors, using the Euclidean
L2 metric. In case of a tie between the k labels of neighbors, it will choose an arbitrary
option.
