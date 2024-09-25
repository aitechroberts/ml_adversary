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