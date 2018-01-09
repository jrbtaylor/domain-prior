# domain-prior
Demonstrates the use of a PixelCNN to detect out-of-domain examples as a pre-filtering for an application-specific CNN

## TO-DO:
- [ ] Train a PixelCNN on MNIST
- [ ] Run the EMNIST (letters) dataset through the MNIST-trained PixelCNN and calculate prediction losses 3 ways:
    1. Uniform average over the image
    2. Weight by the normalized difference of each pixel between it's left and upper neighbors (to effectively down-weight large constant regions that are easy to predict and not domain-specific)
    3. Weight by the attention map of a classifier trained on the original dataset (i.e. the application-specific CNN)
- [ ] Generate fast adversarial examples for the classifier network and run through the PixelCNN, calculate losses the same 3 ways as above
