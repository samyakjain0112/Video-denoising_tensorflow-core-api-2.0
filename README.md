# Video-denoising_tensorflow-core-api 2.0
#### Inspired from the Transformer Network used in the paper(attention is all you need by vaswani et al) 'https://arxiv.org/pdf/1907.01361.pdf'  and U-Net for video denoising used in the paper (Fast DVD-Net) 'https://arxiv.org/pdf/1907.01361.pdf'
#### Using the similiar architecture as in attention is all you need the model broadly consists of an ENCODER and a DECODER.
#### The encoder part of the model inputs five noisy video frames that is [t-2,t-1,t,t+1,t+2] where t is the nosiy frame which we aim to denoise
#### The decoder part of the model inputs two previous ground truth denoised reference frames that are[T-2,T-1] and the aim of the entire model is to predict the Tth frame. 
#### We also concatenate the encoder and decoder corresponding feature maps so that the model can generalise well.
#### The main difference in our proposed architecture and the current state of the art is the using of T-2,T-1 reference ground truth denoised frames in the model and the using of encoder and decoder type architecture.
#### The loss fuction is a weighted average of DSSIM and L2 loss.
_________________________________________________________________________________________________________________________________
