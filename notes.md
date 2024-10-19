# Classifiers

First standardize the data to prevent the absolute values of a given feature to have undue influence
- To do this, first find the mean and divide by standard deviation

## Nearest Neighbors Classifiers
Biggest problem is that you have to calculate the distance for all the vectors in relation to each other so the complexity grows linearly for each feature vector

## Evaluation


# Support Vector Machines

Said earlier that 0-1 loss function was all the same

but now, some losses will be weighted differently

Previously had training and test set

Now, we will have training, test, and validation set

## SVM Problem Formulation
Assuming binary classification

### Linear SVM uses Linear Boundary
SVM uses hyperplane as its decision boundary for binary classification
- which in 2D is just a line so y=mx+b
- uses d in the notation to annotate the decision boundary, rather than say an i, j, or k

- above the line expression is positive, below is negative
- points further from the boundary will have larger magnitude

#### Classifcation function (or prediction function)
SVM assigns a label to a feature vector by plugging the vector into the decision boundary function
- Then a positive feature gets +1, negative feature gets -1
- In the case of 0, it gets +1 value

Written as sign(a^T*x_i + b) **Literally sign(function)**

### What about no clean boundary?
You need a numerical way to compare boundaries
Develop loss functions that define which is better

#### Loss Function 1
- Zero loss if classified correctly
- Positive loss if x_i misclassified with larger loss for a larger magnitude of distance from boundary

Loss function 1 meets reqs:
- max(0, -y_i*(a^T * x_i + b)) because positive y gets no loss while negative number for misclassification gets loss because of the max function
- and because you multiply the loss times it's magnitude, the max of (0, loss*magnitude) gets that loss magnitude for measuring loss

Training Error Cost
-S(a,b) = 1/N*sum(Insert eq here from lecture)

Boundaries that have roughly similar average magnitudes of the data away from the decision boundary provides a larger margin for better generalization when classifying run-time data.
- maximize those magnitudes

#### Loss Function 2
Impose a small positive loss for correctly classified instances close to the boundary

Hinge Loss
- max(0, 1 - y_i*(a^T * x_i + b))
- prove with some values

Problems
- Favors decision boundaries with large absolute magnitudes because increasing magnitudes can artificially zero out a loss function for a near-border instance essentially making it negligible
- You can arbitrarily boost the coefficient of the decision boundary to make that happen
- also makes the classification function extremely sensitive to small changes in features which is bad for robustness to run-time data

So it turns out small magnitudes are better

##### Fixing Hinge Loss
add a penalty on the square magnitude
- a^2 = a^T*a
- add lambda((a^T*a/2)) to the training error cost function
Lambda is known as the regularization parameter and we need to find what value we need to given, but typically between 0.001 and 0.1

###### Calc review
Find the minimum of a function by taking the derivate and setting it to zero

With multivariable, take the partial derivative of each value and with respect to that value and set it equal to 0

For the gradient, you just put it all in a vector

If you can't solve the equations, use iterative solutions, ex: Newton's method.

### Training Procedure
1. Training error cost S(a,b) is function of decision boundary parameters
2. Fix lambda and set initial values of (a,b)
3. search iteratively for (a,b) that minimizes S(a,b) on the training set
4. Repeat

### Iterative minimization by gradient descent
Gradient gives us the steepest way up the hill so we make it negative to give us down the hill
- We take the gradient of the max inside the sum funtion and we take the gradient of the a^T*a
- Not good for training to scale linearly with size of training set, that's bad

To fix that, we take one random item and conduct the gradient of that item, and we do that every step.

This is called SGD or Stochastic Gradient Descent

### Stochastic Gradient Descent
An epoch is a certain number of steps in the descent (usually set to be approximately the size N of the training set)

In the eth epoch, it is common to set the steplength n = m/(e + n) where m and n are constants chose by the experiment
- As you progress in the epochs, make your step sizes smaller by changing m and n

n is the size of the step you're taking. Larger in the beginning, but smaller toward the end because if your step size is too big, you might skip the minimum.

#### Validation and Testing
- split training 70%, val 10%, test set 20%
- for each choice of lambda, run SGD to find best parameters on training set
- choose best lambda based on accuracy on validation set after training the model with training set.
- evaluate SVM accuracy on test set
- This process avoids overfitting(a,b) and lambda

#### Extension to multiclass
All vs all
- (10 choose 2) to train separate binary classifier for each pair of classifiers=45 with (10 choose 2)
- complexity scales quadratically
- To classify, run all classifiers and see which class label is chose most

One vs all
- run a vs all others, then b vs all others, etc.
- So there's only 10 classifiers for each which scales linearly

This is called ensemble methods where you use many classifiers

#### Linear Classification
Uses one or more hyperplanes as decision boundaries

Insert equations here

The a^T vector controls the orientation of the boundary which makes sense since it basically combines a ton of lines to make the hyperplane

- SVMs are linear by design

Naive Bayes Classifiers are also linear

Insert equation here

- at prediction time log() terms are fixed a and b in a^T * x + b = 0
- values of x are indicator values in Bernoulli Native Bayes and counts in Multinomial

K-Nearest Neighbors are nonlinear
- but the decision boundary can be approximated as locally linear in specific locations
- Adding neighbors becomes that degree polynomial drawing that hyperplane I believe, confirm that

Definitely insert that picture here from slides

Noise attack on convolutional neural networks
- Take image not in training set, create an image of noise and scale it (multiply it) to .007 or 1/150. 
- In the range of 0-255 of the pixels we have basically added 1-2, but it destroys the classifier and not it thinks the panda is a reddish, light brown gibbon

Insert CAPTCHA defense example because it's a super cool use case.

#### Why adversarial linear classification?

Basically gives us intuitino to understand attacking and defending nonlinear classifiers

## Attacks
feature vectos x_i are 
- not training data
- are attack candidates that can be pushed to x_i + delta_x_i

Attack vectors must be constrained because it's not an attack if you literally change the data

### Attack Goals
1. Targeted attack is to defeat just one point for any given reason like a SPAM email
2. Reliability attacks just destroy the classifier

Optimal policy on the attack depends on the steepness of the line, and which way to move depends on the a-vector

Noise attack with absolute sum constraint results in the same attack vectors as the patch attack

Noise attack with max constraint (l_inf)
- you get a square around the point instead of a diamond using: max(abs(delta_x_i^1), (delta_x_i^2)) <= C

Noise attack with Euclidean constraint (l_2), basically pythagorean theorem *at least in 2D for sure, idk about 3D
- creates a circle
- attack vector must be perpendicular to the boundary so delta_x_i is + or - k * a for some value k since it's in the direction of the a-vector

### Defense
You retrain by using original images and noisy images using the correct labels actually hardens the classifier. Could do this multiple times, but usually only one works. People can commonly only train the noise rather than the original image which is a bad move because then you could be attacked by just going back to the original.



# Random Forest Classifier


## Decision Trees
Suck but starting here leads to random forests

Looks like separating via boundaries and creating quadrants and the like is similar to my thoughts about creating partial hyperplanes or changing hyperplanes dynamically based on ruling out specific classes

Uncertainty is the number of bits (number of binary questions) to get from the top to the bottom

First question should start with the highest probability of execution 
- 1 Spade, 2 Hearts, 4 Clubs, 1 Diamond
- Ask Club first, Then Heart, then ask either are you a Spade or are you a Diamond

Entropy = Uncertainty for a general distribution
**But the entropy of a dataset H(D) = sum(P(i)*log_2(1/P(i))) from i=1 to c  the weighted mean across all c classes**
- In the above cards, it would be 1.75 bits

Information gain of a Split is the amount it reduces entropy on average

I = H(D) - (H(D_1)*Split_data/total_data + Other_side_of_split/total_data * H(D_2))

### Training

### Classifying

### Choosing dimension and split

## Random Forest Classifiers
Usually the first you would try on many classification problems

## How to Choose a Classifier

# Neural Networks

## Neurons
One neuron implements the calculation: o = F(w^T * x + b) same thing as a ^ T
- x is input vector
- w is weights vector and b is bias parameter
- F is nonlinear activation function
- o is output number

Usual choice for F is ReLU (rectified linear unit): F(u) = max(0, u)

### Binary Classifier of 2 Neurons
Insert picture from lecture

- Insert the x-vector into neuron activiation function F()  with weight_1 then do the same with weight_2

Show decision space picture

### Extend from Binary to Multiclass Classifier
The equation goes up to subscript k for k-number of neurons

Replace max of neuron outputs by **softmax** function s(o) to get probabilistic decisions instead of definitive classifications
- Insert softmax activiation function picture

O_k becomes a probability vector which can then be classified according to the highest probability

#### Minibatch Stochastic Gradient Descent
- Cost function is S(theta, x; lambda) = cross-entropy loss L + regularization summation across weights
    - where theta is the parameter vector that includes all {(w_i, b_i)}
    - Inside cross-entropy loss is y_i^T is the dot product with the log(1/minibatch) vector
    - weights * weights transposed times (lambda/2)

S(theta, x; lambda) is minimized by SGD in which each step is calculated with reference to a minibatch of the data set, not just one item
- Most common to use a minibatch of value from the power of 2 such as (2,4,8,16,32,64, etc.)


As with SVMs and other classifiers, we still have to properly prepocess the data to get meaningful features

## Stacking Layers to get a neural network

Add an input layer of neurons to generate the features that go to the original output layer. Any number of intermediate layers can be in between the input and output layers

### Forward pass and backpropagation
- Given parameters theta, a forward pass through the network represents prediction of s from x
- Turns out, we can understand setting the parameters theta as training backpropagation

S = comparison between y_i and s(o(u (x, theta_1, theta_2))) read as s is a function of o which is a function of u and theta_2 and u is a function of (x, theta_1)
- With S, we want to take a step in the negative gradient direction to change the value of theta_2 to make a better value of theta_2, and the same thing with x and theta_1.
    - Chain Rule 
    - Jacobian is a matrix of partial derivatives
- Recursion for gradients is backpropagation

#### Training Trick 1: Dropout to avoid overfitting
- During training randomly drop neurons with fixed probability
- Dropout prevents higher layers becoming too dependent on a small set of weights
- No dropout during prediction because they've already been removed.

Insert Picture here

#### Training Trick 2: Gradient Scaling
Taking steps down a gradient is not generally smooth
**Momentum** smooths the trajectory  by taking a moving average of recent steps
- Other gradient scaling techniques include Adagrad, RMSprop, and Adam

#### When not to use a fully connected input layer
- In image classification a fully connected input layer is pixel-oriented, but patterns aren't found in individual pixels

#### Image Convolution: Pattern detection by image convolution
- Take a grayscale image I with pixel value I_u,v at location (u,v)
- And small "kernel" iage M with pixel value M_u,v

Construct a new image N=conv(I,M). Then multiply I * M and sum all of the products

Insert picture


Stride and padding

### Stacking Convolutional Layers
The input convolutional layer is a simple pattern detector and next layer detects patterns of patterns and so on
- The **receptive field** of a location in a block is set of pixels in the original image that influence that location
    - aka what pixel in original field drives this decision. 1st layer shoud be very small receptive field whereas whole image should be receptive field as end.

To shrink blocks, use a stride > 1, but you can also use pooling. Insert picture and is simple

## Taxonomy of Adversarial Attacks
Security violation
- Integrity = False Negatives, Availability = False Positive

Noise attack with max constraint (l_inf)
Add or subtract C from either dimension, and the information to add or subtract is found in the sign(a) vector minute 26-30 ish

## Extend Ideas of constrained noise attack from linear case to Neural Networks
Again, since pixels are basically 0-255, that's basically adding or subtracting 1 for adding noise

### Problem 1: Nonlinearity
NNs have nonlinear components: Ex: Softmax or activation functions like RELU
- But RELU is piecewise linear and other functions are approximately linear locally
- NN decision space is locally linear, making noise attacks possible

### Problem 2: Finding direction to attack
Linear boundary is a^T * x + b = 0, then direction of attack is:
- + or - a for Euclidean constraint (l_2)
- + or - sign(a) for max constraint(l_inf)

In NN given x_i, we will examine gradients loss using .... minute 38

#### Revising backpropagation:
- Recall training NN by fixing training data set {(x_i,y_i)}
- Finding parameters theta that minizes loss function (ex: cross-entropy loss)

#### Attacking trained model with backpropagation
Now, model parameters theta are fixed
- We move the data point x_i in order to **maximize** the loss function

Open Box attack because we have access to the model

epison value is the magnitude of the attack. Unlike + or - 32 in HW, epsilon is between 0 and 1

- If an item is already misclassified by the classifier, we ignore it because why mess with it?

FGSM attack code, takes in image (Tensor object kinda with metadat), target is target label
- nll is negative_log_likelihood. Just a random loss function
- initialize gradients as 0
- do backpropagation with .backward()
- take gradient, compute sign
- then create modified image
- Ensure image is still valid image with torch.clamp(x,0,1) to stay within 0 to 1

#### Iterative method
Take many steps by iteration the following update to x_i: x_i = clip of set (x_i + alpha * sign(gradient))
- The clip function keeps each feature of x_i within the set of its original value, between min and max allowable values
- Can outperform FGSM where loss function is nonconvex

#### Targeted iterative method
Suppose we want to move x_i so that is specifically has label y_target
- Must minimize the loss funtion of L_i(theta, x_i, y_target)
- Clip becomes x_i = clip of set (x_i - alpha * sign(gradient))

### From Open to Closed box attacks
Naturally open because backprop requires information about model structure
- Experiments show that adversarial examples are **transferable**
    - Examples found in one model often work against another model even if models are trained on different subsets of the same training data
    - Why? Today's models approximate linear modles locally so the decision boundaries get quite similar

Reverse Engineering attack to build a replica
- We can do the same thing we did to the spam filter to a public neural network classifier to turn open box into closed box

Defenses against evasion attacks
- Iterative retraining
    - Train the model
    - Create many adversarial examples (benefits from FGSM being fast)
    - Add them to training set and retrain
    - Iterate if desired
- Incorporate FGSM into the loss function for training
    - Cool little thought project to multiply original loss function by weight alpha, then multiply the loss function of perturbed examples with (1-alpha)


# Regression Problem
Given set of feature vectors x_i where each has numerical label y_i, we want to train a model that can map unlabeled vectors to numerical values
- Can think of regression as fitting a line to data

Regression is like classification except that prediction target is number and not class label which apparently changes everything

Feature vectors are now **explanatory variables** and y (labels) are the **dependent variables**

## Linear model
Begin by modeling y as a linear function of x^(j) plus randomness
- This is the normal y = x * Beta + Error that you're familar with
    - Error is a zero-mean random variable tha represents model error
- Vector notation basically the exact same except its x.T * Beta_vector + Error rather than a.T * x_vector + bias

Since Y is not a vector basically like the label vectors except this is numbers you get
||Error||^2 = E.T * E =  (y_vector- x * Beta_vector).T * (y-X * Beta)
- And you choose Beta to minimize the errors
Differentiate wrt Beta and set to zero
- and you get this X.T * X * Beta_vector - X.T * y_vector = 0
If X.T * X is invertible,
- you get Beta^hat = (X.T * X)^(-1) * X.T * y_vector

### Constant offset
Mathematically, we can add Beta_0 as a constant to offset for when all x = 0 because Error doesn't act as that apparently

So we but Beta_0 at the top of the Beta_vector and we add a 1 at the beginning of the x_vector like [1 x(1) x(2)]
so that you pretty much multiply Beta_0 * 1 so you get Beta_0
- Basically adding a column of 1's on the on the far left column of X

### Evaluating models using R-squared
Variance of all of the components (y, x.T*Beta, and Error)
so we get R^2 = var(x.T*Beta)/var(y)

## Transforming variables to find a linear fitting
Could take natural log, cube, do whatever you want to the variables

Problems: 
1. Could overfit because it's not obvious how to transform the explanatory variables
2. Linear regression model parameters are very sensitive to outliers

## Avoid Overfitting
Methods
1. Validation: Use validation set to choose transformed explanatory variables but # of combinations is exponential in # of variables
2. Regularization: 

### Regularization
In OLS, cost function was ||e||^2

In regularized least squares, add complexity penalty weighted by lambda, known as **ridge regression**

e.T * e + lambda * ||Beta||^2 = (y - X*Beta) + lambda * Beta.T * Beta

### Training with RLS
Differentiate cost function wrt Beta and set to zero gives (X.T * X + lambda * I) *  Beta - X.T * y_vector = 0
    - (X.T * X + lambda * I) Always invertible







