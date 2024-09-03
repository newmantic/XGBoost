# XGBoost


XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm used for classification and regression tasks. It is an implementation of gradient boosting designed for efficiency, speed, and scalability. XGBoost builds an ensemble of decision trees in a sequential manner, where each tree is trained to correct the errors made by the previous ones.


Gradient Boosting is an ensemble technique that combines the predictions of several weak learners (typically decision trees) to produce a strong learner. The idea is to sequentially add trees that minimize the errors (residuals) of the ensemble so far.
The general form of a boosted model is:
F(x) = sum(f_k(x) for k in 1 to K)
where each f_k(x) is a decision tree.

The objective function to be minimized in XGBoost consists of two parts:
Obj(Theta) = sum(L(y_i, F(x_i))) + sum(Omega(f_k))
L(y_i, F(x_i)): The loss function that measures how well the model F(x) predicts the true labels y_i. For regression, this might be mean squared error; for classification, it might be logistic loss.
Omega(f_k): A regularization term that penalizes the complexity of the model (e.g., the depth of the trees) to prevent overfitting.


Additive Learning:
XGBoost builds the model in stages by adding one tree at a time. At each stage, the goal is to add a tree that reduces the objective function. If F_m(x) is the model after m trees, the next tree f_m+1(x) is added such that:
F_m+1(x) = F_m(x) + f_m+1(x)

Gradient Descent:
The new tree f_m+1(x) is chosen to minimize the objective function. This is done using gradient descent, where the gradient of the loss function with respect to the model's predictions is used to update the model:
g_i = partial derivative of L(y_i, F_m(x_i)) with respect to F_m(x_i)
h_i = second partial derivative of L(y_i, F_m(x_i)) with respect to F_m(x_i)
Here, g_i is the gradient (how much to adjust the prediction), and h_i is the Hessian (second derivative, which helps in adjusting the step size).

Regularization:
Regularization in XGBoost is crucial to control overfitting. The regularization term typically includes:
L2 regularization on the weights of the leaves in the trees:
Omega(f) = gamma * T + 0.5 * lambda * sum(w_j^2)
where T is the number of leaves, w_j are the weights of the leaves, gamma is the penalty for adding a new leaf, and lambda controls the regularization strength.

Tree Structure:
The trees in XGBoost are binary trees, where each leaf contains a weight w_j. The prediction for a data point x_i is the sum of the weights of the leaves that the data point falls into across all trees:
F(x_i) = sum(w_j for each tree)

Learning Rate:
The learning rate eta controls the contribution of each tree to the final model. After each tree is added, the model is updated as:
F_m+1(x) = F_m(x) + eta * f_m+1(x)
A smaller eta requires more trees to reach the same model complexity but generally leads to better generalization.


Initialize the model with a constant value (e.g., the mean of the labels).
For m from 1 to M (the total number of trees):
Compute the gradients and Hessians for all data points.
Fit a new tree to these gradients and Hessians.
Update the model by adding the contribution of the new tree.
Output the final model, which is the sum of all trees.
