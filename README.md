# Data_Augmentation_GANs

As of now, all of the data augmentation methods that we are aware of focus on ways to create new, synthetic data out of the existing data. 
Data augmentations are used nearly universally in computer vision applications to obtain more training data and improve model generalisation. 


The primary techniques employed are:

* rotation, 
* cropping, 
* noise injection, 
* flipping, 
* zooming, and many other techniques. 

These changes are made in real time in computer vision utilising data generators. A batch of data is randomly altered (augmented) when it is fed into your neural network. Before training, there is nothing you need to prepare.

The model is going to be taught to assess various approaches like as cropping, flipping, zooming, and so on. It will get training to undo these augmentations, and 
to your neural network it is randomly transformed (augmented). You don’t need to prepare anything before training.

The model will be trained to analyse the cropping, flipping, zooming and etc.. techniques. It will be trained to reverse these augmentations done and perceive the picture.

However, what if we added to the current dataset by selecting characteristics from other pictures and producing fresh ones for training? The models will have fresh characteristics—more precisely, new picture details—to work with and hone.

Our goal with this repository is to generate new images (augment dataset) using GANs (generative adversarial networks) from the current dataset.

Seems intriguing? We'll see!


