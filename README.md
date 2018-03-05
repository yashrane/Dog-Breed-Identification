# Dog-Breed-Identification

### The Problem
Kaggle hosted a [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identificationhttps://www.kaggle.com/c/dog-breed-identification) challenge, where the goal is to build a model that can identify the breed of a dog when given an image. There are 120 breeds, and a relatively small number of training images per class, which makes the problem harder than it orginally seems.

### Methodology
The obvious answer to an image recognition problem would be to use a convolutional neural network. Unfortunately, the small number of training examples makes this difficult, as any CNN trained soley on the given training images would be severely overfit. To alleviate this, I used transfer learning with Resnet18 to give my model a warm start and drastically cut down on training difficulties.

I chose Resnet18 as my starting model due to its relatively deep structure and robustness against the vanishing gradient problem. The deep structure allowed my model to be complex enough to accurately identify the dogs. Resnet34, or other deepr models, may be able to achieve greater accuracy, but due to limitations on the computation power I have available, I used Resnet18. Additionally, Resnet uses residual blocks to mitigate vanishing gradients, which allows every training example to have a greater impact on the optimization of the model and helps the model converge on an optimal solution easier than it would otherwise on such a limited training set.

### Results
The model performed reasonably well, achieving ~70% accuracy on the validation set. A few sample predictions can be seen below.

![alt text](https://github.com/yashrane/Dog-Breed-Identification/blob/master/predictions/prediction%231.jpg)

![alt text](https://github.com/yashrane/Dog-Breed-Identification/blob/master/predictions/prediction%234.jpg)

### Future Improvements
If I wanted to improve the accuracy of this model, I would have to do one of two things: use a deeper model or use a larger training set. If using a deeper model improves the accuracy, that would mean that my current model is underfitting the data. Using a larger training set, such as the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), would reduce overfitting, is almost guaranteed to improve accuracy. However, it will also take much more computation power.
Going forward, it would also be a good idea to do error analysis and plot a confusion matrix to see exactly what kinds of images it is getting wrong. It may turn out that it is incorrectly classifying breeds that are very similar to begin with.
