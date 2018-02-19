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

The need for uncertainty in deep learning 