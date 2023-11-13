from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from tensorflow.keras.preprocessing.image import save_img
from tensorflow import io
from tensorflow import image

import numpy as np
import tensorflow as tf
import time
import argparse
import warnings

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.utils import get_file

from scipy.optimize import fmin_l_bfgs_b

# The following PIL import is fine if you need to use PIL directly
from PIL import Image



"""
Neural Style Transfer with Keras 2.0.5

Based on:
https://github.com/keras-team/keras-io/blob/master/examples/generative/neural_style_transfer.py

Contains few improvements suggested in the paper Improving the Neural Algorithm of Artistic Style
(http://arxiv.org/abs/1605.04603).

-----------------------------------------------------------------------------------------------------------------------
"""

parser = argparse.ArgumentParser(description='Neural style transfer.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')

parser.add_argument('style_image_paths', metavar='ref', nargs='+', type=str,
                    help='Path to the style reference image.')

parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')


parser.add_argument("--style_masks", type=str, default=None, nargs='+',
                    help='Masks for style images')

parser.add_argument("--content_mask", type=str, default=None,
                    help='Masks for the content image')

parser.add_argument("--color_mask", type=str, default=None,
                    help='Mask for color preservation')

parser.add_argument("--image_size", dest="img_size", default=400, type=int,
                    help='Minimum image size')

parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float,
                    help="Weight of content")

parser.add_argument("--style_weight", dest="style_weight", nargs='+', default=[1], type=float,
                    help="Weight of style, can be multiple for multiple styles")

parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float,
                    help="Scale the weighing of the style")

parser.add_argument("--total_variation_weight", dest="tv_weight", default=8.5e-5, type=float,
                    help="Total Variation weight")

parser.add_argument("--num_iter", dest="num_iter", default=10, type=int,
                    help="Number of iterations")

parser.add_argument("--model", default="vgg16", type=str,
                    help="Choices are 'vgg16' and 'vgg19'")

parser.add_argument("--content_loss_type", default=0, type=int,
                    help='Can be one of 0, 1 or 2. Readme contains the required information of each mode.')

parser.add_argument("--rescale_image", dest="rescale_image", default="False", type=str,
                    help="Rescale image after execution to original dimentions")

parser.add_argument("--rescale_method", dest="rescale_method", default="bilinear", type=str,
                    help="Rescale image algorithm")

parser.add_argument("--maintain_aspect_ratio", dest="maintain_aspect_ratio", default="True", type=str,
                    help="Maintain aspect ratio of loaded images")

parser.add_argument("--content_layer", dest="content_layer", default="conv5_2", type=str,
                    help="Content layer used for content loss.")

parser.add_argument("--init_image", dest="init_image", default="content", type=str,
                    help="Initial image used to generate the final image. Options are 'content', 'noise', or 'gray'")

parser.add_argument("--pool_type", dest="pool", default="max", type=str,
                    help='Pooling type. Can be "ave" for average pooling or "max" for max pooling')

parser.add_argument('--preserve_color', dest='color', default="False", type=str,
                    help='Preserve original color in image')

parser.add_argument('--min_improvement', default=0.0, type=float,
                    help='Defines minimum improvement required to continue script')


# Parse the arguments
args = parser.parse_args()

def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")
  
# Process the arguments
base_image_path = args.base_image_path
style_reference_image_paths = args.style_image_paths
result_prefix = args.result_prefix

style_image_paths = style_reference_image_paths  # This list is redundant, you can use style_reference_image_paths directly

style_masks_present = args.style_masks is not None
mask_paths = args.style_masks if style_masks_present else []

# Assert to ensure the correct number of style masks is provided
if style_masks_present:
    assert len(style_image_paths) == len(mask_paths), f"Wrong number of style masks provided. " \
                                                      f"Number of style images = {len(style_image_paths)}, " \
                                                      f"Number of style mask paths = {len(mask_paths)}."

content_mask_present = args.content_mask is not None
content_mask_path = args.content_mask

color_mask_present = args.color_mask is not None

# Parse the arguments
args = parser.parse_args()

# Now you can safely use args to load the model
model = VGG16(include_top=False) if args.model == "vgg16" else VGG19(include_top=False)

# Convert string arguments to boolean
rescale_image = str_to_bool(args.rescale_image)
maintain_aspect_ratio = str_to_bool(args.maintain_aspect_ratio)
preserve_color = str_to_bool(args.color)

# Assign weights from the arguments
content_weight = args.content_weight
total_variation_weight = args.tv_weight

# Compute style weights
if len(style_image_paths) != len(args.style_weight):
    print("Mismatch in number of style images provided and number of style weights provided. "
          "Found {} style images and {} style weights. "
          "Equally distributing weights to all other styles.".format(len(style_image_paths), len(args.style_weight)))

    weight_sum = sum(args.style_weight) * args.style_scale
    style_weights = [weight_sum / len(style_image_paths)] * len(style_image_paths)
else:
    style_weights = [style_weight * args.style_scale for style_weight in args.style_weight]

# Decide pooling function
pool_type = str(args.pool).lower()
assert pool_type in ["ave", "max"], 'Pooling argument is wrong. Needs to be either "ave" or "max".'

# In TensorFlow 2.x, you would typically define a pooling layer in the model itself rather than using a flag
# However, if you need to decide between using average or max pooling elsewhere in the code:
use_average_pooling = pool_type == "ave"

read_mode = "gray" if args.init_image == "gray" else "color"

# Dimensions of the generated picture will be set later
img_width = img_height = None
img_WIDTH = img_HEIGHT = None
aspect_ratio = None  # This will be calculated based on the image dimensions

assert args.content_loss_type in [0, 1, 2], "Content Loss Type must be one of 0, 1, or 2"

# Util function to open, resize, and format pictures into appropriate tensors will go here
# Ensure you import TensorFlow and related modules if you're working with tensors
import numpy as np
from PIL import Image

def preprocess_image(image_path, load_dims=False, read_mode="color"):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    # Read the image
    img = tf.io.read_file(image_path)
    # Decode the image in the correct color mode
    img = tf.image.decode_image(img, channels=3 if read_mode == "color" else 1)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if load_dims:
        # Use tf.shape to get image dimensions
        img_HEIGHT, img_WIDTH = tf.shape(img)[0], tf.shape(img)[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

        img_width = args.img_size
        img_height = int(img_width * aspect_ratio) if maintain_aspect_ratio else args.img_size

    # Resize the image as needed using TensorFlow
    img = tf.image.resize(img, [img_height, img_width], method=tf.image.ResizeMethod.LANCZOS5)

    # Convert RGB to BGR for VGG model
    if read_mode == "color":
        img = img[..., ::-1]

    # Subtract the mean values per channel (imagenet mean)
    img -= [103.939, 116.779, 123.68]

    # Add a batch dimension
    img = tf.expand_dims(img, axis=0)

    # If using 'channels_first', transpose the image
    if K.image_data_format() == "channels_first":
        img = tf.transpose(img, (0, 3, 1, 2))

    return img


import tensorflow as tf

def deprocess_image(x):
    # Check if the data format is 'channels_first' and transpose accordingly
    if K.image_data_format() == "channels_first":
        x = tf.transpose(x, (1, 2, 0))

    # Add imagenet mean per channel (this reverses the mean subtraction)
    mean = [103.939, 116.779, 123.68]
    if K.image_data_format() == "channels_first":
        mean = tf.reshape(mean, (1, 1, 1, 3))
    else:
        mean = tf.reshape(mean, (1, 1, 3))

    x += mean

    # BGR -> RGB
    x = x[..., ::-1]

    # Clip the values to 0-255 and convert to uint8
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)

    # Remove the batch dimension if there is one
    if x.shape[0] == 1:
        x = tf.squeeze(x, axis=0)

    return x
  
# util function to preserve image color
import tensorflow as tf

def original_color_transform(content, generated, mask=None):
    # Convert images to float32 tensors
    content = tf.convert_to_tensor(content, dtype=tf.float32)
    generated = tf.convert_to_tensor(generated, dtype=tf.float32)

    # Normalize the pixel values to [0, 1] (assuming input is uint8)
    content = content / 255.0
    generated = generated / 255.0

    # Convert RGB to YCbCr
    content_ycc = tf.image.rgb_to_yuv(content)
    generated_ycc = tf.image.rgb_to_yuv(generated)

    # Create mask if not provided
    if mask is None:
        mask = tf.ones_like(content_ycc[:, :, 0])

    # Broadcast the mask to have the same number of channels as the images
    mask = mask[:, :, tf.newaxis]

    # Replace the chrominance of the generated image with that of the content image
    generated_ycc = tf.concat([generated_ycc[:, :, 0:1], content_ycc[:, :, 1:] * mask + generated_ycc[:, :, 1:] * (1 - mask)], axis=-1)

    # Convert YCbCr back to RGB
    generated = tf.image.yuv_to_rgb(generated_ycc)

    # Denormalize the pixel values to [0, 255]
    generated = tf.clip_by_value(generated * 255.0, 0, 255)
    generated = tf.cast(generated, tf.uint8)

    return generated

def load_mask(mask_path, shape, return_mask_img=False):
    # Read the mask image
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    
    # Resize the mask to match the target shape
    if K.image_data_format() == "channels_first":
        _, channels, width, height = shape
        resize_shape = [width, height]
    else:
        _, width, height, channels = shape
        resize_shape = [height, width]
    mask = tf.image.resize(mask, resize_shape)
    
    # Perform binarization of mask
    mask = tf.where(mask <= 0.5, x=0.0, y=1.0)

    if return_mask_img:
        # Squeeze to remove the channel dimension since the mask is grayscale
        return tf.squeeze(mask, axis=-1)

    # Create a mask tensor with the same shape as the input image
    mask_tensor = tf.tile(mask, [1, 1, channels])

    if K.image_data_format() == "channels_first":
        mask_tensor = tf.transpose(mask_tensor, perm=[2, 0, 1])

    return mask_tensor

from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D

def pooling_func(x, pool_type='max'):
    # Define the pool size and strides
    pool_size = (2, 2)
    strides = (2, 2)
    
    # Apply Average Pooling if specified
    if pool_type == 'ave':
        return AveragePooling2D(pool_size, strides=strides)(x)
    # Apply Max Pooling by default
    else:
        return MaxPooling2D(pool_size, strides=strides)(x)

# get tensor representations of our images
# Process the base image
base_image = preprocess_image(base_image_path, load_dims=True, read_mode=read_mode)

# Process the style reference images
style_reference_images = [preprocess_image(style_path) for style_path in style_image_paths]

# For the generated image, we'll start with a tensor that has the same shape as the base image
combination_image = tf.Variable(tf.random.uniform(shape=base_image.shape, dtype=tf.float32))

# Combine the images into a single tensor
image_tensors = tf.stack([base_image] + style_reference_images + [combination_image])

# Create an input layer with the combined images tensor
input_layer = tf.keras.layers.Input(tensor=image_tensors)

# Load the VGG model with the input layer
if args.model == "vgg19":
    # Load VGG19
    vgg = VGG19(include_top=False, weights='imagenet', input_tensor=input_layer)
else:
    # Load VGG16 by default
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_layer)

# If you need to access the model's intermediate layers, you can do so as follows:
# x = vgg.get_layer('block5_conv3').output
# x = pooling_func(x)

# Create the model
model = Model(inputs=input_layer, outputs=vgg.output)

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model
import tensorflow as tf

# Set the correct data format
tf.keras.backend.set_image_data_format('channels_last')  # or 'channels_first'

# Load the model
if args.model == "vgg19":
    # Load pre-trained VGG19 model without the classifier layers
    model = VGG19(include_top=False, weights='imagenet')
else:
    # Load pre-trained VGG16 model without the classifier layers
    model = VGG16(include_top=False, weights='imagenet')

# Print a message indicating that the model was successfully loaded
print('Model loaded with pre-trained ImageNet weights.')

layer_output = outputs_dict['layer_name']
layer_shape = shape_dict['layer_name']

# compute the neural style loss
# first we need to define 4 util functions

# Improvement 1
# the gram matrix of an image tensor (feature-wise outer product) using shifted activations
import tensorflow as tf

def gram_matrix(x):
    assert tf.rank(x) == 3, "Input tensor must be 3-dimensional"
    # If the input tensor uses 'channels_first', we permute it to 'channels_last' for consistency
    if tf.keras.backend.image_data_format() == 'channels_first':
        x = tf.transpose(x, (1, 2, 0))
    # We make the input tensor 2-dimensional by flattening the height and width dimensions
    # into the features dimension, while preserving the channels dimension
    features = tf.reshape(x, (-1, tf.shape(x)[-1]))
    # Compute the Gram matrix by multiplying the matrix by its transpose
    # Note: 'features' matrix has shape [H*W, C]
    # The transpose will have shape [C, H*W]
    # The result will be a square matrix of shape [C, C] where C is the number of channels
    gram = tf.matmul(features, features, transpose_a=True)
    
    return gram

def style_loss(style, combination, mask_path=None, content_mask_path=None, nb_channels=None):
    assert tf.rank(style) == 3
    assert tf.rank(combination) == 3

    # If there is a content mask path, load and apply the mask
    if content_mask_path is not None:
        content_mask = load_mask(content_mask_path, (1, *style.shape), return_mask_img=True)
        combination = combination * tf.stop_gradient(content_mask)

    # If there is a mask path, load and apply the mask
    if mask_path is not None:
        style_mask = load_mask(mask_path, (1, *style.shape), return_mask_img=True)
        style = style * tf.stop_gradient(style_mask)
        # If no content mask, apply style mask to combination
        if content_mask_path is None:
            combination = combination * tf.stop_gradient(style_mask)

    # Compute the Gram matrices for the style and combination features
    S = gram_matrix(style)
    C = gram_matrix(combination)

    # Assume the number of channels is the last dimension of the style tensor
    channels = int(style.shape[-1])
    # Calculate the size as the product of the dimensions of the style feature map
    size = int(style.shape[0]) * int(style.shape[1])
    
    # Compute the style loss
    loss = tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
    return loss

import tensorflow as tf

def content_loss(base, combination):
    # Determine the dimension ordering and set channel_dim accordingly
    channel_dim = 0 if tf.keras.backend.image_data_format() == "channels_first" else -1

    # Get the number of channels from the base image tensor shape
    channels = base.shape[channel_dim]
    size = img_width * img_height

    # Calculate the scaling multiplier based on content_loss_type
    if args.content_loss_type == 1:
        multiplier = 1. / (2. * (channels ** 0.5) * (size ** 0.5))
    elif args.content_loss_type == 2:
        multiplier = 1. / (channels * size)
    else:
        multiplier = 1.

    # Compute the content loss as the scaled sum of squared differences
    loss = multiplier * tf.reduce_sum(tf.square(combination - base))
    return loss

import tensorflow as tf

def total_variation_loss(x):
    assert tf.rank(x) == 4, "Input tensor must be 4-dimensional"
    
    if tf.keras.backend.image_data_format() == "channels_first":
        # Calculate the difference for the height dimension
        a = tf.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, 1:, :img_height - 1])
        # Calculate the difference for the width dimension
        b = tf.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, :img_width - 1, 1:])
    else:
        # Calculate the difference for the width dimension
        a = tf.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
        # Calculate the difference for the height dimension
        b = tf.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])

    # Summing up the pixel-wise differences and raising to the power of 1.25
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Selection of feature layers to use for the style loss component
if args.model == "vgg19":
    feature_layers = [
        'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'
    ]
else:  # Default to VGG16 feature layers
    feature_layers = [
        'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
        'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'
    ]

# Assuming outputs_dict is a dictionary mapping layer names to their output tensors,
# args.content_layer is the name of the content layer, and content_loss is defined.

# Get the features of the base and combination images from the specified content layer
layer_features = outputs_dict[args.content_layer]
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[-1, :, :, :]

# Calculate the content loss
content_loss_value = content_loss(base_image_features, combination_features)
loss = tf.zeros(shape=())
loss += content_weight * content_loss_value

# Assuming 'feature_layers' is a list of layer names for which you want to compute style loss
nb_layers = len(feature_layers) - 1

# Prepare style masks
if style_masks_present:
    # If masks are provided, use them
    style_masks = mask_paths
else:
    # Otherwise, create a list of None values to indicate no masks
    style_masks = [None for _ in range(nb_style_images)]

# Determine the channel index based on the image data format
channel_index = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1

import tensorflow as tf

# Initialize the loss variable
loss = tf.zeros(shape=())

# Iterate through the layers to calculate the style loss
for i in range(len(feature_layers) - 1):
    # Get the features for the current layer
    layer_features = outputs_dict[feature_layers[i]]
    combination_features = layer_features[-1, :, :, :]
    style_reference_features = layer_features[1:-1, :, :, :]
    sl1 = [style_loss(style_feat, combination_features) for style_feat in style_reference_features]

    # Get the features for the next layer
    next_layer_features = outputs_dict[feature_layers[i + 1]]
    combination_features_next = next_layer_features[-1, :, :, :]
    style_reference_features_next = next_layer_features[1:-1, :, :, :]
    sl2 = [style_loss(style_feat, combination_features_next) for style_feat in style_reference_features_next]

    # Calculate the style loss for each style image and add it to the total loss
    for j in range(nb_style_images):
        # Difference between the style losses of layer i and layer i+1
        sl = sl1[j] - sl2[j]
        
        # Apply geometric weighting to the style loss
        weighted_sl = (style_weights[j] / (2 ** (nb_layers - (i + 1)))) * sl
        loss += weighted_sl

# Add total variation loss to the total loss
loss += total_variation_weight * total_variation_loss(combination_image)

# Assuming 'initial_value' is a preprocessed image tensor suitable for the model input

# Convert the initial image to a tf.Variable which will be trainable by the optimizer
combination_image = tf.Variable(initial_value, dtype=tf.float32)

# Set up the GradientTape context to watch the trainable tf.Variable
with tf.GradientTape() as tape:
    # Run the model and compute the current loss
    # It is assumed that the 'loss' computation has been defined elsewhere in the code
    current_loss = loss(combination_image)

# Compute the gradients of the loss with respect to the combination image
grads = tape.gradient(current_loss, combination_image)

# Apply the gradients to the combination image using an optimizer
# Assuming an optimizer has been defined, e.g., Adam
optimizer = tf.optimizers.Adam(learning_rate=5.0)
optimizer.apply_gradients([(grads, combination_image)])

def eval_loss_and_grads(combination_image):
    with tf.GradientTape() as tape:
        # Compute the loss
        current_loss = loss(combination_image)
        
    # Compute the gradients
    grads = tape.gradient(current_loss, combination_image)
    # Flatten the gradients to be compatible with scipy.optimize
    grad_values = grads.numpy().flatten().astype('float64')
    return current_loss.numpy().astype('float64'), grad_values



class Evaluator(tf.Module):
    def __init__(self, compute_loss):
        super(Evaluator, self).__init__()
        self.compute_loss = compute_loss
        self.loss_value = None
        self.grad_values = None

    def __call__(self, x):
        with tf.GradientTape() as tape:
            # Update the image
            x = tf.convert_to_tensor(x.reshape((1, img_width, img_height, 3)), dtype=tf.float32)
            # Watch the input image
            tape.watch(x)
            # Compute the loss
            loss_value = self.compute_loss(x)
        # Compute the gradients
        grad_values = tape.gradient(loss_value, x)
        # Store the values
        self.loss_value = loss_value.numpy().astype('float64')
        self.grad_values = grad_values.numpy().flatten().astype('float64')
        return self.loss_value

    @property
    def grads(self):
        return self.grad_values

# Assume 'compute_loss' is a function that computes the loss given an image
evaluator = Evaluator(compute_loss)

# Optimization loop
for i in range(num_iter):
    print(f"Starting iteration {i + 1} of {num_iter}")
    start_time = time.time()

    # Run the L-BFGS optimizer
    x, min_val, info = fmin_l_bfgs_b(evaluator, x.flatten(), fprime=evaluator.grads, maxfun=20)
