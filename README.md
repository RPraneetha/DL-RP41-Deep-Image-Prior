# Reproducing Inpainting and Image Restoration from [Deep Image Prior](https://arxiv.org/pdf/1711.10925.pdf)

## Introduction
The goal of this project is to reproduce [Figure 7](https://arxiv.org/pdf/1711.10925.pdf#figure.caption.7) and [Table
 1](https://arxiv.org/pdf/1711.10925.pdf#table.caption.8) of the paper Deep Image Prior by Dmitry Ulyanov et. al. The
  goal of this paper was to show that for image generation and image restoration tasks, the structure of a network
   with randomly-initialized weights can function as a handcrafted prior, and the performance of this network on
    tasks like inpainting, image restoration, flash-no flash reconstruction, etc. is comparable to that of
     learning-based methods. The authors did this by they used untrained deep convolutional neural networks and
      instead of training it on a huge dataset, they fit a generator network to a single degraded image. They highly
       emphasize that the only information required to solve image restoration problems is contained in the single
        degraded input image they have given as input, and the handcrafted structure of the network.

## A Few Formulae 
Neural networks usually used in practice, specifically trained networks which first learn from date and then are applied to solve image tasks 
use $$x=f_\theta(z)$$, where $$x$$ is an image which is mapped to a random code vector $$z$$. The network described in this paper
 captures the information learnt solely on the basis of its architecture, without learning any of its parameters. This is done by
  considering the network itself as a parameterization of the input image.
    
Image restoration tasks can be formulated as minimizing the term $$x^* = \min_{x} E(x; x_0) + R(x)$$ where $$ E(x; x_0) $$ is a task-dependent
 data term, $$x_0$$ is the corrupted image and $$R(x)$$ is the regularizer. This regularizer is replaced with the implicit prior captured 
  by the network as $$\theta^* = argmin_{x} E(f_\theta(z); x_0)$$ consequently formulating the problem as $$x^* = f_\theta^*(z)$$

## Playing Architect
This paper uses a handcrafted encoder-decoder architecture with skip connections as shown in [Figure 1](#Skip-Architecture) 
 as given in the [supplementary material](https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM#pdfviewer).

![Skip-Architecture](/assets/skip_architecture.png)

The architecture is based on a U-net "hourglass",  and is a fully-convolutional encoder-decoder architecture, where the input of the network _z_ has 
 the same spatial resolution as the output of the image _f(z)_. LeakyRELU is used as a non-linearity, and the downsampling method technique used is strides, 
  implemented within the convoltuional modules. Upsampling operation is nearest neighbour upsampling for the inpainting application, and bilinear for image restoration.
   We have used reflective padding in all the convolutional layers. For the input $$z$$ we have generated noise randomly using Bernoulli sampling, and for the 
    optimization process, we have used the _ADAM_ optimizer. The optimal results can be obtained when the network is carefully 
     tuned to a particular task, and so clearly the parameters and the architecture will differ for each task. 
  
## Man the Parameters

The authors have given the hyperparameters they have used in the implementation of their code, but there is a discrepancy 
 between the parameters they have provided in the supplementary material and the ones they have used in the code. 
  (We have tested our code with both the sets of hyperparameters and did not find any significant difference.) The standard 
   architecture uses the following set of hyperparameters: <br /> <br />
   $$\boxed{
   z \in \mathbb{R}^{32xHxW} \sim U(0, \frac{1}{10}) \\
   n_u = n_d = [128, 128, 128, 128, 128] \\
   k_u = k_d = [3, 3, 3, 3, 3] \\
   n_s = [4, 4, 4, 4, 4] \\
   k_s = [1, 1, 1, 1, 1] \\
   \sigma_p = \frac{1}{30} \\
   num_iter = 2000 \\
   LR = 0.01 \\
   upsampling = bilinear
   }
   $$
  
## Glorified Copy-Paste

### Architecture

### Inpainting

### Image Restoration

## Results

## Conclusion
