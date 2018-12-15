# Flower classification resnet50 and inceptionV3

State-of-the-art CNN architectures for multi-class flower classification using deep learning framework Keras.


The notebook can be run on Google colab https://colab.research.google.com/notebooks/welcome.ipynb with GPU/TPU. 

# Dataset:



The dataset is extracted from Kaggle that contains 4242 labelled images of flower from five different classes.
https://www.kaggle.com/alxmamaev/flowers-recognition/kernels

To get the dataset directly from Kaggle to the current session in colab:


	!pip install -q kaggle
	
	!mkdir -p ~/.kaggle
	
	!cp kaggle.json ~/.kaggle/
	
	!chmod 600 ~/.kaggle/kaggle.json
	
	!kaggle datasets download -d alxmamaev/flowers-recognition
	!unzip flowers-recognition.zip

# ResidualNetork(50 layers):

A powerful Resnet architecture mostly used for image classification.


Stage 1:

Zero-padding used to pad the input with a pad of (3,3)
A 2D Convolution with 64 filters of shape (7,7) and uses a stride of (2,2)
BatchNormalization applied followed by MaxPooling using a (3,3) window and a (2,2) stride.


Stage 2:

The convolutional block uses three set of filters of size [64,64,256]
The 2 identity blocks use three set of filters of size [64,64,256].


Stage 3:


The convolutional block uses three set of filters of size [128,128,512]
The 3 identity blocks use three set of filters of size [128,128,512]


Stage 4:


The convolutional block uses three set of filters of size [256, 256, 1024]
The 5 identity blocks use three set of filters of size [256, 256, 1024]


Stage 5:


The convolutional block uses three set of filters of size [512, 512, 2048]
The 2 identity blocks use three set of filters of size [512, 512, 2048]
The 2D Average Pooling uses a window of shape (2,2) 

The flatten reduces to 1D

A Fully Connected Dense layer of size 64 using relu activation

A Fully Connected Dense layer to reduce its input to the number of classes using a softmax activation.

 ## Accuracy:
 
 Resnet50:(random_initialization)
 optimiser ='stocastic gradient descent' 
 loss = categorical cross entropy
 epoch =10
 batch_size = 32
 train_acc = .8006
 val_acc  = .5032
 
 # Transfer learning:

Using Pre-trained "Imagenet" weights on Resnet50 and InceptionV3 achitecture and fine tuning the classifier.

## Accuracy:                       							
Resnet50:(trained on Imagenet)+Dense128+Dense64
optimiser =stochastic gradient descent 
loss = categorical cross entropy
epoch =10
batch_size = 32
train_acc = .9916
val_acc  = .8834

InceptionV3:(trained on Imagenet)+Dense128+Dropout+Dense64
optimiser = stochastic gradient descent  
loss = categorical cross entropy
epoch =10
batch_size = 32
train_acc = .9046
val_acc  = .7956

Note: More data and tweaking hyperparameters would result in better accuracy.

Reference:
[[Deep Residual Learning for Image Recognition|https://arxiv.org/abs/1512.03385]]

[[Rethinking the Inception Architecture for Computer Vision|https://arxiv.org/abs/1512.00567]]
