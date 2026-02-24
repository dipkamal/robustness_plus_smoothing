# Training for Trustworthy Saliency Maps: Adversarial Training Meets Feature-Map Smoothing

Repository for paper "*Training for Trustworthy Saliency Maps: Adversarial Training Meets Feature-Map Smoothing"*.

**Overview:**
Input-gradient–based attribution methods, such as Vanilla Gradient (VG) and Integrated Gradients (IG), are widely used to explain image classifiers through saliency maps. Yet these maps are often noisy, unstable under perturbations, and difficult to trust. Prior work has largely focused on refining attribution techniques, leaving open how the training process itself shapes explanation quality. 

In this work, we establish a theoretical link between model sensitivity and explanation stability, and study how adversarial training influences saliency maps. While adversarial training improves sparsity, it also degrades output stability, producing explanations that fluctuate across small perturbations in logit-space. To address this trade-off, we propose augmenting adversarial training with lightweight feature-map smoothing block (e.g., Gaussian filter) inserted into intermediate layer. 

## Requirement
- kornia (for applying filters to feature-maps)
- cleverhans (for adversarial training)
- captum (for saliency maps)
- quantus (for computing several metrics for saliency maps)
- ImageNette dataset ([download link](https://github.com/fastai/imagenette). )

The directory consists of the following file and folders: 
- FMNIST: This folder consists of a notebook that demonstrates training of several FMNIST models.
- CIFAR-10: This folder consists of python files for training CIFAR-10 models.
- ImageNette: This folder consists of python files for training ImageNette models.
- common_metrics.py: This program file consists of python functions for evaluating saliency maps.
