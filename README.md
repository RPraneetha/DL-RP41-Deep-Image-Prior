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
        
## Architecture
This paper uses a handcrafted encoder-decoder architecture with skip connections as shown in [Figure 1](#Skip-Architecture) 
which was provided in the [supplementary material](https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM#pdfviewer).

![Skip-Architecture](/assets/skip_architecture.png)

We can see that the architecture is similar to a U-net "hourglass" architecture, where the input of the network _z_ has 
 the same spatial resolution as the output of the image _f(z)_. The optimal results can be obtained when the network is 
  carefully tuned to a particular task, hence the parameters and the architecture differs a little for each task. **Insert math about architecture**
  $m \in {0,1}^{H \times W}$
  
 ## Parameters
 
 The authors have given the hyperparameters they have used in the implementation of their code, but there is a discrepancy 
 between the parameters they have provided in the supplementary material and the ones they have used in the code. 
  (We have tested our code with both the sets of hyperparameters and did not find any significant difference.) **Insert hyperparameters**
  
## Replication
### Inpainting

### Image Restoration

## Results

## Conclusion
