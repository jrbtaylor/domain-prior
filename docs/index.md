---
title: Uncertain about Uncertainty in Deep Learning
tagline: 
description: This post shows how a top paper in deep learning fails a simple 
baseline
---

## Outline
0. This post discusses uncertainty in deep learning, comparing the popular
   epistemic uncertainty paper by Kendall and Gal with some new PixelCNN 
   approaches I wanted to try
1. Why we need uncertainty
    - No model is perfect; uncertainty allows us to account for examples that
    lie outside the training distribution - e.g. due to outliers, a lack of 
    diversity in the training set or adversarial examples
    - recent literature on adversarial examples highlights the sensitivity of 
    deep neural nets to data outside the training distribution
2. Epistemic uncertainty
    - epistemic vs aleatoric uncertainty
3. Adversarial examples
4. Out-of-domain examples


# Uncertain about Uncertainty in Deep Learning

This post discusses uncertainty in deep learning: why we need it and how the
current best approach fails a simple baseline.



## Why we need uncertainty

Regardless of the fact that deep learning has produced results in computer 
vision that were nearly unimaginable a decade ago, neural nets should 
not be taken blindly as always being correct.
No model is perfect &mdash; the real world is noisy and will always find 
outliers not captured in the training distribution.
Uncertainty allows us to account for possible errors in our models and weigh
the risks when making decisions with their outputs.

Imagine a neural network trained to detect and segment tumors in medical images.
If the model is applied in another clinic where the images have different 
distributions of pixel intensities and/or noise, we'd want the model to work the
same, but failing that, we'd want some indication from the model that it was 
less confident in its results.

The recent explosion of literature on adversarial examples has highlighted the
sensitivity of deep neural nets to data away from the training distribution.
In some cases, adding imperceptibly-small noise to an image can produce 
seemingly nonsensical results from state of the art models &mdash; for example,
[the turtle that a Google Inception-V3 model classifies as a rifle](
https://www.youtube.com/watch?v=piYnd_wYlT8).

<div style="text-align:center"><img src ="https://github.com/jrbtaylor/conditional-pixelcnn/blob/master/docs/images/rifle_turtle.gif?raw=true"/></div>
*A 3D printed turtle from MIT that gets classified as a rifle*

So not only do these models lack a built-in mechanism for representing 
uncertainty (as opposed to a probabilistic model), they're also sensitive to 
small perturbations to their inputs and fail to extrapolate smoothly outside
of the training distribution.



## Epistemic Uncertainty

The need for uncertainty in deep learning has been stressed before and work
in this area can be broken down into two camps: 1) fundamentally changing the 
models to be probabilistic in nature; 2) continue to train networks with 
deterministic weights by gradient descent and exploit some hack afterward that 
vaguely mimics uncertainty.
For the former, we have Bayesian deep learning, which puts a prior distribution 
over the weights and requires calculating the expectation over the distributions
for all weights at inference.
This is currently not computationally feasible on a large scale.

The popular paper ["What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"](
https://arxiv.org/abs/1703.04977) by Kendall and Gal, which is the focus of this 
post, falls under the latter category.
Note that there's also a well-written and concise[blog post for the paper,](
https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/
)which I wish more authors would do. 
The post is a great intro to uncertainty.

The paper covers two commonly-used forms of uncertainty:
1. Aleatoric: uncertainty with respect to information missing from the data;
    e.g. occlusions in computer vision; usually task-specific and not depedent
    on the amount of training data.
2. Epistemic: model uncertainty that can be explained away given enough data;
    this is the focus of this post, as it allows us to say if an input matches 
    the training distribution, where we can be reasonably confident
    in our model.

A quick tangent: 
Aleatoric uncertainty is represented by adding a second set of outputs to 
the model. 
These uncertainty outputs down-weight the loss during training in exchange 
for a regularizing penalty. 
This allows the model to balance the trade-off of being completely certain, 
(taking the full loss for wrong predictions), 
with being uncertain (increasing the penalty) by learning which inputs tend to
have higher losses &mdash; e.g. a segmentation model might learn to be less
confident around unclear object boundaries.

The technique proposed for epistemic uncertainty estimates it as the prediction 
variance sampled over multiple dropout masks at test.
For the full derivation, the reader is encouraged to see the paper at the link above.
Instead, I'll offer what I hope is a more intuitive explanation. 

The approach can be understood as exploiting the same property of neural nets as 
adversarial examples: outside the training distribution, the network exhibits
highly non-linear and irregular behaviours.
For a network trained with dropout, perturbations to the activations introduced 
by dropout are effectively smoothed out throughout training for examples in the
training distribution.
For examples that differ significantly from the training examples (i.e. outliers
or adversarial examples), the perturbations added by dropout lead to higher
variance in the predictions.

