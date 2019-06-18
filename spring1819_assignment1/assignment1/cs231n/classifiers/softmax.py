from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        y_pred = X[i].dot(W)
        y_pred_norm = y_pred - np.mean(y_pred) 
        # To prevent overflow errors, we multiply numerator and denominator by C, where C is max(y_pred).
        # This means C*exp(y) = exp(y+logC)
        y_pred_norm_exp = np.exp(y_pred_norm)
        loss += -np.log(y_pred_norm_exp[y[i]]/np.sum(y_pred_norm_exp))
        dW_grad_y = np.empty(shape=[1,num_classes])
        for y_i in range(num_classes):
            if y_i == y[i]:
                dW_grad_y[0][y_i] = y_pred_norm_exp[y_i]/np.sum(y_pred_norm_exp) - 1
            else:
                dW_grad_y[0][y_i] = y_pred_norm_exp[y_i]/np.sum(y_pred_norm_exp)
        dW += np.matmul(np.expand_dims(X[i], axis=1), dW_grad_y)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(np.square(W))
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    y_pred = np.matmul(X, W)
    y_pred_norm = y_pred - np.mean(y_pred, axis=1, keepdims=True)
    # To prevent overflow errors, we multiply numerator and denominator by C, where C is max(y_pred).
    # This means C*exp(y) = exp(y+logC)
    y_pred_norm_exp = np.exp(y_pred_norm)
    loss = -np.sum(np.log(y_pred_norm_exp[np.arange(num_train), y] / np.sum(y_pred_norm_exp, axis=1)))
    dW_grad_y = y_pred_norm_exp / np.sum(y_pred_norm_exp, axis=1, keepdims=True)
    dW_grad_y[np.arange(num_train), y] -= 1
    dW = np.matmul(np.transpose(X), dW_grad_y)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(np.square(W))
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
