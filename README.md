# domain-prior
Demonstrates the use of a PixelCNN to detect out-of-domain examples as a pre-filtering for an application-specific CNN

## TO-DO:
- [ ] Train a convnet with a soft attention mechanism to classify MNIST digits
- [ ] Train PixelCNNs on MNIST with the 3 loss functions: average; weighted by normalized difference to left- and upper-neighbors; weighted by attention mechanism of classifier network
    - [ ] Save the validation losses as a numpy array for each for comparison to other datasets
    - [ ] Run the EMNIST (letters) dataset through the PixelCNNs and save their losses
- [ ] Write visualization code, s.t. for a given dataset we can visualize the examples with highest/lowest loss from the PixelCNN
    - [ ] Save collage of best/worst examples
    - [ ] Generate gif of collage that transitions between images and pixel-wise loss maps (holding each end of the GIF for a couple seconds, then interpolating between)
    - [ ] Plot histogram of losses on MNIST and EMNIST together
- [ ] Generate fast adversarial examples for the classifier network and calculate losses for each PixelCNN
