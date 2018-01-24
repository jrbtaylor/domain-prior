# domain-prior
Develop and test ideas using PixelCNNs to detect out-of-domain examples as a pre-filtering for an application-specific CNN.
The objective is to be able to selectively ignore the classifier output in cases where the input differs too much from the training distribution as modelled by the PixelCNN &mdash; i.e. produce a binary estimate of the data uncertainty (high or low).
This is compared with Monte-Carlo sampling of Dropout as an approximation of Bayesian "uncertainty" (cite paper...).

TODO: Re-run now with plots of epistemic uncertainty and compare. Re-write the rest of this.

Compares three different weightings of the PixelCNN cross entropy loss:
1. Average (mean) over the image
2. Weighted by the (normalized) difference between each pixel and its upper- and left-neighbours &mdash; i.e.  to ignore easily predictable large constant regions that tend to not be informative of the image domain
3. Weighted by the (normalized) pixel saliency found by backpropagating the classification CNN output to its input and taking the absolute value &mdash; i.e. similar to (2) but learned, to ignore PixelCNN loss in regions of the image that are not meaningful in the application domain

Findings:
- This idea doesn't seem to work at all for distinguishing between MNIST and EMNIST (letters). I had hoped that at least the saliency-weighted loss would pick out that the PixelCNN can't predict the new corners, but that doesn't appear to be the case.