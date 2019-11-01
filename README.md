# Video-denoising_tensorflow-core-api 2.0
#### Inspired from the Transformer Network used in the paper(attention is all you need by vaswani et al) 'https://arxiv.org/pdf/1907.01361.pdf'  and U-Net for video denoising used in the paper (Fast DVD-Net) 'https://arxiv.org/pdf/1907.01361.pdf'
#### Using the similiar architecture as in attention is all you need the model broadly consists of an ENCODER and a DECODER.
#### The encoder part of the model inputs five noisy video frames that is [t-2,t-1,t,t+1,t+2] where t is the nosiy frame which we aim to denoise
#### The decoder part of the model inputs two previous ground truth denoised reference frames that are[T-2,T-1] and the aim of the entire model is to predict the Tth frame. 
#### We also concatenate the encoder and decoder corresponding feature maps so that the model can generalise well.
#### The main difference in our proposed architecture and the current state of the art is the using of T-2,T-1 reference ground truth denoised frames in the model and concatenating of encoder and decoder feature maps.
#### The loss fuction is L2 loss.
_________________________________________________________________________________________________________________________________

#MODEL
The input to the encoder part is 5 frames[t-2,t-1,t,t+1,t+2] where t is the noisy frame which we aim to denoise. The input to the decoder part is 2 previous decoised ground truth time frames [T-2,T-1] and we aim to predict the T present denoised frame. 
In the encoder part taking inspiration from different architecture networks we first do SPATIAL denoising by passing each frame to a 8 layers long convolutional network and the parameters are shared for each of the five noisy frames. Now the output of the five frames is concatinated to form a feature map of 15 layers (3*5) and this is the passed through a unet which first downsamples and then upsamples them. However the U-net used here is a bit different from the conventional ones as here instead of concatination addition operation is used. Now the U-net outputs 3 feature maps. While this was all happening on the encoder side on the decoder side the two reference ground truth frames[T-1,T-2]  
