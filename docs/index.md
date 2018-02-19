---
title: Uncertain about Uncertainty: Top papers fail simple baselines
tagline: 
description: This post compares methods to evaluate model uncertainty for deep 
neural nets
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
3. PixelCNNs as an image prior and using their loss as a proxy for uncertainty
4. Adversarial examples
5. Out-of-domain examples


# Uncertain about Uncertainty: Top papers fail simple baselines

This post discusses uncertainty in deep learning: why we need it, where the
current best approach fails and some new ideas.



## Why we need uncertainty

Regardless of the fact that deep learning has produced results in computer 
vision that may have been unimaginable a decade ago, neural nets should 
not be taken blindly as always being correct.
No model is perfect &mdash; the real world is noisy and will always find 
outliers not captured in the training distribution.
Uncertainty allows us to account for possible errors in our models and weigh
the risks when making decisions with their outputs.

The recent explosion of literature on adversarial examples has highlighted the
sensitivity of deep neural nets to data away from the training distribution.
In some cases, adding imperceptibly-small noise to an image can produce 
seemingly nonsensical results from state of the art models &mdash; for example,
[the turtle that a Google Inception-V3 model classifies as a rifle](
https://www.youtube.com/watch?v=piYnd_wYlT8).

<div style="text-align:center"><img src ="https://github.com/jrbtaylor/conditional-pixelcnn/blob/master/docs/images/rifle_turtle.gif?raw=true"/></div>
*A 3D printed turtle from MIT that gets classified as a rifle*

So not only do these models have no inherent notion of uncertainty (as opposed
to a probabilistic model), they're also sensitive to small perturbations to 
their inputs.



## Epistemic Uncertainty

The need for uncertainty in deep learning has been stressed before and some work
exists on making neural nets more Bayesian.
Bayesian deep learning puts a prior distribution over the weights and requires
calculating the expectation over the distributions for all weights at inference,
which is currently not computationally feasible on a large scale.

The simpler approach is to continue to train deterministic networks with 
gradient descent and exploit some hacks afterward that vaguely mimic uncertainty.
The popular paper ["What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"](
https://arxiv.org/abs/1703.04977) by Kendall and Gal, which is the focus of this 
post, falls under this category.

Note that Alex Kendall also covers the paper in a well-written and concise[blog post,](
https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/
)which I wish more authors would do.
Before I critique the work itself, I have to give the author credit &mdash; 
the blog post is a great intro to uncertainty.

The paper covers two common forms of uncertainty:
1. Aleatoric: uncertainty with respect to information missing from the data;
    e.g. occlusions in computer vision; usually task-specific and not depedent
    on the amount of training data
2. Epistemic: model uncertainty that can be explained away given enough data;
    this is the focus of this post, as it allows us to say if an input matches 
    the training distribution, where we can be reasonably confident
    in our model.

Aleatoric uncertainty is represented by adding a second set of outputs to 
the model. 
These uncertainty outputs down-weight the loss during training in exchange 
for a regularizing penalty. 
This allows the model to balance the trade-off of being completely certain, 
(taking the full loss for wrong predictions), 
with being uncertain (increasing the penalty) by learning which inputs lead to
higher losses.

The technique proposed for epistemic uncertainty estimates it as the prediction 
variance sampled over multiple dropout masks at test.
For the full derivation, the reader is encouraged to see the paper at the link above.
Instead, I'll offer what I hope is a more intuitive explanation. 
The approach can be understood as exploiting the same property of neural nets as 
adversarial examples: outside the training distribution, the network exhibits
highly non-linear and irregular behaviours.
By applying dropout, we perturb the activations such that for inputs not captured
in the training set, the resulting activations may exhibit higher variance. 