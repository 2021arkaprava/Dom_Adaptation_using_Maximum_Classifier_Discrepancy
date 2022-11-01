# Dom_Adaptation_using_Maximum_Classifier_Discrepancy

Here we've experimented the following paper: https://arxiv.org/abs/1712.02560

We've experimented using SVHN to MNIST.

Results:
Trained on SVHN fully supervised, then test on SVHN: 91%
Trained on SVHN fully supervised, then test on MNIST: 65%
Trained using this method, using source as SVHN labeled and target MNIST unlabeled and tested on MNIST: 71%, with epochs = 50 and n = 2.
