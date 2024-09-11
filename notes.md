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