# Style-Transfer
A style transfer demo

### Intuition

Noticed that the original image data is compressed at each layer of a CNN. By doing this, will lose some details but gain efficiency (e.g. pooling operation). Imaging that if we represent the original image with the feature maps output from one of the convolutional layer, we will get a blurry image without details.

|                           original                           |                        representation                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![elephant](https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/elephant.jpg) | ![content](https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/content.jpg) |

As a result, there are "room" to balance content and style, and any details didn't be filled in can be filled in with style.

### Loss function

<img src="https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/cap_2. Style Transfer Theory_00:10:21_05.jpg" alt="cap_2. Style Transfer Theory_00:10:21_05" style="zoom:50%;" />

There are three inputs: the content image, the style image, and the target image(image going to be optimized). The optimize goal is a weighted combination of the **content loss** and the **style loss**.

**Gram Matrix**

Gram Matrix $G$ is a statistical representation of style. It looks suspiciously like autocorrelation matrix.

- "Auto" (self) - correlation between a thing and itself
- "Correlation" - how related one thing is to another
- "Autocorrelation" - how related X is to itself

$$
G = \frac{1}{N} X X^T
$$

Be noticed that The Gram Matrix $G$ is a matrix, therefor 2D, but the output of convolution $X$ is 3D, so the first step is **flatten** $X$ along the spatial dimension, and the second step is **transpose** (if needed; different libraries use different ordering conventions) so that "color" / "feature maps" comes first. If `X.shape` is $(C, HW)$, then `G.shape` is $(C, C)$. By doing this, the spatial dimension disappears, this is make sense because when it comes to style, we don't care about *where* something occurred in the image.

<img src="https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/cap_2. Style Transfer Theory_00:08:29_03.jpg" alt="cap_2. Style Transfer Theory_00:08:29_03" style="zoom:50%;" />

**Content loss**

- Pass the content image and the target image through a same pre-trained CNN like VGG
- Calculate the Mean Squared Error (MSE) between these two outputs

**Style loss**

- Pass the style image through the same CNN

- Grab the output at five different locations and calculate the Gram Matrix
- Do the same thing for the target image
- Calculate the MSE between the Gram Matrix of the style image and target image
- Calculate the weighted sum of these MSEs

**Total loss**

Add these the content loss and the style loss together to get the total loss
$$
L = \alpha L_{content} + \beta L_{style}
$$

$$
X^* = argmin_{X} L
$$

### Results

|                         style_image                          |                        content_image                         |                            output                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/nahan.jpg" alt="nahan" style="zoom:50%;" /> | <img src="https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/DHU.jpg" alt="DHU" style="zoom:50%;" /> | <img src="https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/DHU_nahan.jpg" alt="DHU_nahan" style="zoom:50%;" /> |
| <img src="https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/lesdemoisellesdavignon.jpg" alt="lesdemoisellesdavignon" style="zoom:50%;" /> | <img src="https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/sydney.jpg" alt="sydney" style="zoom:50%;" /> | <img src="https://cdn.jsdelivr.net/gh/Shadowalker1995/images/2020/sydney_lesdemoisellesdavignon.jpg" alt="sydney_lesdemoisellesdavignon" style="zoom:50%;" /> |

**loss**

```
iter=0, loss=3710.9208984375
iter=1, loss=1113.28857421875
iter=2, loss=732.7880249023438
iter=3, loss=569.886474609375
iter=4, loss=478.1509704589844
iter=5, loss=423.4969787597656
iter=6, loss=385.49444580078125
iter=7, loss=358.8546447753906
iter=8, loss=337.9999084472656
iter=9, loss=321.8232116699219
duration: 0:00:50.642452
```

