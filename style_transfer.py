"""
FileName:	style_transfer.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-03-18 23:27:47
"""

from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.optimize import fmin_l_bfgs_b

import tensorflow as tf
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


def VGG16_AvgPool(shape):
    # we want to account for features across the entire image
    # so get gid of the maxpool which throws away information
    # and using avgpool
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)

    i = vgg.input
    x = i
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # replace it with AvgPooling2D
            x = AveragePooling2D()(x)
        else:
            x = layer(x)
    return Model(i, x)


def VGG16_AvgPool_CutOff(shape, num_convs):
    # there are 13 convolutions in total
    if num_convs < 1 or num_convs > 13:
        print("num_convs argument must be in the range [1, 13]")
        return None

    model = VGG16_AvgPool(shape)

    n = 0
    output = None
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        if n >= num_convs:
            output = layer.get_output_at(1)
            break
    return Model(model.input, output)


def gram_matrix(img):
    # input is (H, W, C) (C = # feature maps)
    # convert to (C, H*W)
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))

    # gram = XX^T / N
    G = K.dot(X, K.transpose(X) / img.get_shape().num_elements())
    return G


def style_loss(y, t):
    return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


def load_img_and_preprocess(path, shape=None):
    img = image.load_img(path, target_size=shape)

    # convert image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def minimize(fn, epochs, batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(
            func=fn,
            x0=x,
            maxfun=20)

        x = np.clip(x, -127, 127)
        print("iter={}, loss={}".format(i, l))
        losses.append(l)

    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]


def unpreprocess(img):
    mean = [103.939, 116.779, 123.68]
    img[..., 0] += mean[0]
    img[..., 1] += mean[1]
    img[..., 2] += mean[2]
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / (x.max() - x.min())
    return x


if __name__ == '__main__':
    # content_img_name = 'DHU'
    content_img_name = 'sydney'
    # content_img_name = 'elephant'
    content_img = load_img_and_preprocess(
        'content/{}.jpg'.format(content_img_name),
        # (225, 300),
        )

    # resize the style image since it doesn't matter
    h, w = content_img.shape[1:3]
    # style_imge_name = 'nahan'
    # style_imge_name = 'starrynight'
    style_imge_name = 'lesdemoisellesdavignon'
    # style_imge_name = 'flowercarrier'
    style_img = load_img_and_preprocess(
        'styles/{}.jpg'.format(style_imge_name),
        (h, w))

    batch_shape = content_img.shape
    shape = content_img.shape[1:]

    # see the image
    # plt.imshow(img)
    # plt.show()

    # make only 1 VGG here as the final model needs to have a common input
    vgg = VGG16_AvgPool(shape)

    # create the content model with single output
    # 1,2,4,5,7-9,11-13,15-17
    content_model = Model(vgg.input, vgg.layers[13].get_output_at(1))
    content_target = K.variable(content_model.predict(content_img))

    # create the style model with multiple ouputs
    symbolic_conv_outputs = [
        layer.get_output_at(1) for layer in vgg.layers \
            if layer.name.endswith('conv1')]
    style_model = Model(vgg.input, symbolic_conv_outputs)
    # calculate the targets that are output at each layer
    style_layers_targets = [K.variable(y) for y in style_model.predict(style_img)]

    # assume the weight of the content loss is 1
    # only weight the style losses
    style_weights = [0.2, 0.4, 0.3, 0.5, 0.2]
    # style_weights = [weight / 2 for weight in style_weights]

    # create the total loss which is the sum of content + style loss
    loss = K.mean(K.square(content_model.output - content_target))
    for w, symbolic, target in zip(style_weights, symbolic_conv_outputs, style_layers_targets):
        # gram_matrix() expects a (H, W, C) as input()
        loss += w * style_loss(symbolic[0], target[0])

    # once again, create the gradients and loss / grads function
    # note: it doesn't matter which model's input is used
    # they are both pointing to the same keras Input layer in memory
    grads = K.gradients(loss, vgg.input)

    # just like theano.function
    get_loss_and_grads = K.function(
        inputs=[vgg.input],
        outputs=[loss] + grads)


    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
    plt.imshow(scale_img(final_img))
    plt.show()
    image.save_img('results/{}_{}.jpg'.format(content_img_name, style_imge_name), scale_img(final_img))
