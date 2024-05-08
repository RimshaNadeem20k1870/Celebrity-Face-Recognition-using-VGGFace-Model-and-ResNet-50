# Celebrity-Face-Recognition-using-VGGFace-Model
We using a pre-trained model trained on Face recognition to recognize similar faces. Here, we are particularly interested in recognizing whether two given faces are of the same celebrity or not.
// pretrained models- models that are trained on large datsets and can be integrated using tensorfloe,keras, etc

## 1.Importing dataset from kaggle 
Link has been given below:
https://www.kaggle.com/datasets/hereisburak/pins-face-recognition
also included in code.
## 2. Import necessay libraries
//google collab is recommend as most libraries will be already installed.

## 3.Load metadata
size=17534

## 4.Defining load image fumction for reading image 
    OpenCV loads images with color channels in BGR order. So we need to reverse them

## 5.We have given the predefined model for VGG face
  VGGFace network architecture VGGFace model uses size. Fig 2 shows the dimensions of the layers (height, width, depth). Convolutional layers are accompanied by ReLU rectification units. The last 3 layers are fully connected (FC). The output dimension of the first two FC layers and is 4096 each, of the last FC layer ( 2622.
  //as we were not able to integrate due to libraries error we have predefined it.
  ![VGGFace-network-architecture-VGGFace-model-uses-size-Fig-2-shows-the-dimensions-of-the (1)](https://github.com/RimshaNadeem20k1870/Celebrity-Face-Recognition-using-VGGFace-Model/assets/145101419/72d23e6e-bd9c-4071-89ff-8a0b27021853)
  
## 6.Load the model defined and Then load the given weight file named "vgg_face_weights.h5" 

## 7. Get vgg face descriptor
   It is serving as a new model using second last layer for feature extraction.

## 8. Loading images and embeddings
   We will be loading each image from datset one and one and will embed it.
   Embedding images is the process of representing images as fixed-length numerical vectors in a high-dimensional space. Embedding images-embedding images allows us to represent them in a compact and meaningful format that facilitates various image-related tasks, including feature extraction etc.
   We are using 1500 images.It will take aroung 4 hours.
## 9. Plot images and got distance between two pairs
    The distance between the embeddings of the two images is calculated using the distance function. This function computes the distance between two vectors, likely representing the embeddings of the images.
     The distance provides a measure of dissimilarity or similarity between the images in the embedding space. Smaller distances typically indicate greater similarity, while larger distances indicate greater dissimilarity.

## 10. Create train and test data
    every 9th example goes in test data and rest go in train data - This helps ensure that the model is evaluated on diverse examples

## 11.  Encode Labels
    converting categorical labels (e.g., class names or categories) into numerical representations that can be understood by machine learning algorithms.
## 12. Standardize the feature values
      
Standardizing feature values refers to the process of scaling the values of features in a dataset to have a mean of 0 and a standard deviation of 1. This process is also known as feature scaling or normalization and is a common preprocessing step in machine learning pipelines. Standardizing feature values ensures that all features have a similar scale,

## 13. Reduce dimensions using PCA
    In datasets with a large number of features, the curse of dimensionality can lead to increased computational complexity, overfitting, and difficulty in visualization. PCA reduces the number of features while preserving most of the information, making the dataset more manageable and improving model performance.
## 14.  Build a Machine Learning Classifier SVM classifier as your model of choice. SVM is a popular choice for image classification tasks due to its effectiveness in high-dimensional spaces and ability to handle non-linear decision boundaries through the use of kernel functions.

## 15.  We are taking example indexes and loading images from data and identifying the celebrity.

## 16.Accuracy Score: 96.455%         
