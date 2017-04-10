"""Dump features calculated from a network using Caffe.

Run 'python frame_features.py --help' for usage information.
"""

import argparse
import csv
import errno
import logging
import numpy as np
import os
import pickle

from os.path import basename, dirname, splitext

import caffe
import skimage

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('images',
                    help="""
                         CSV file containing as fields the input image path and
                         the path to output activations to. The output path can
                         be omitted if --output_directory is specified.""")
parser.add_argument('--output_directory',
                    default=None,
                    help="""
                         Output directory to dump features for all images to.
                         If specified, the second field of the images file is
                         ignored.""")
parser.add_argument('--batch_size',
                    default=500,
                    help="""
                         Batch size of images to pass to net at once.
                         Generally, larger is faster but uses more memory.""")

args = parser.parse_args()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                    datefmt='%H:%M:%S')


CAFFE_ROOT = "/home/achald/caffe/"
MODEL_DIR = '{caffe}/models/bvlc_reference_caffenet'.format(caffe=CAFFE_ROOT)
MODEL_PROTOTXT = '{model_dir}/deploy.prototxt'.format(model_dir=MODEL_DIR)
MODEL_CAFFEMODEL = '{model_dir}/bvlc_reference_caffenet.caffemodel'.format(
    model_dir=MODEL_DIR)
MEAN_FILE = '{caffe}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(
    caffe=CAFFE_ROOT)
LAYER_OUTPUT = 'fc7'  # Layer to output activations from.

def mkdir_p(path):
    """Make directory (and necessary parent directories) at path."""
    # Courtesy: http://stackoverflow.com/questions/600268/
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_float_image(filename):
    """Load image as float matrix.

    Args:
        filename (str): Path to image.

    Returns:
        np.float32 matrix in range [0, 1] of size
            (H x W x 3) in RGB or
            (H x W x 1) in grayscale
    """

    im = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
    if (len(im.shape) == 2):
        im = im[:, :, np.newaxis]
    if (im.shape[-1] == 1):
        im = np.concatenate((im, im, im), axis=2)
    return im


def load_net_transformer():
    """Loads a Caffenet and transformer."""
    net = caffe.Net(MODEL_PROTOTXT, MODEL_CAFFEMODEL, caffe.TEST)

    # Preprocess input. Modified from
    # https://github.com/BVLC/caffe/blob/79539180edc73d3716149fb28e07c67a45896be1/examples/00-classification.ipynb
    # and
    # https://github.com/BVLC/caffe/blob/79539180edc73d3716149fb28e07c67a45896be1/python/caffe/classifier.py
    input_layer = net.inputs[0]
    transformer = caffe.io.Transformer({input_layer: net.blobs[
        input_layer].data.shape})
    transformer.set_transpose(input_layer, (2, 0, 1))
    transformer.set_mean(input_layer, np.load(MEAN_FILE).mean(1).mean(1))
    # The reference model operates on images in [0,255] range (not [0,1]).
    transformer.set_raw_scale(input_layer, 255)
    # The reference model has channels in BGR order instead of RGB.
    transformer.set_channel_swap(input_layer, (2, 1, 0))

    return net, transformer

def compute_activations(net, transformer, images):
    """Calculate activations from LAYER_OUTPUT layer for a list of images.

    Args:
        net (caffe.Net)
        transformer (caffe.io.Transformer)
        images (list of nd.float32 arrays representing images)

    Returns:
        (len(images), OUTPUT_LAYER_DIMENSIONS) size nd.float32 array with
        activations for each image.
    """
    input_layer = net.inputs[0]

    # Set the batch size.
    net.blobs[input_layer].reshape(len(images), 3, 227, 227)

    batch_inputs = np.array([transformer.preprocess(input_layer, image)
                             for image in images])
    net.forward_all(**{input_layer: batch_inputs})
    return net.blobs[LAYER_OUTPUT].data

def read_batched_input_file(input_file_path, output_directory, batch_size):
    with open(input_file_path) as input_file:
        path_reader = csv.reader(input_file)
        batch_image_paths = []
        batch_output_paths = []
        for row in path_reader:
            image_path = row[0]
            if output_directory is None:
                if len(row) != 2:
                    raise ValueError(('Image list must contain ouput path if '
                                      '--output_directory is not specified.'))
                output_path = row[1]
            else:
                image_name = splitext(basename(image_path))[0]
                output_path = '{output}/{image_name}.npy'.format(
                    output=output_directory,
                    image_name=image_name)
            if os.path.exists(output_path):
                logging.info('Skipping {}'.format(output_path))
                continue
            batch_output_paths.append(output_path)
            batch_image_paths.append(image_path)
            # Yield once we have a batch ready, clear the batch after yielding.
            if len(batch_image_paths) == batch_size:
                yield (batch_image_paths, batch_output_paths)
                batch_image_paths = []
                batch_output_paths = []
        # Yield last batch.
        yield (batch_image_paths, batch_output_paths)

def main():
    batch_size = int(args.batch_size)
    output_directory = None
    if args.output_directory:
        output_directory = args.output_directory
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

    batched_input_output_paths = read_batched_input_file(
        args.images, output_directory, batch_size)

    caffe.set_mode_gpu()
    net, transformer = load_net_transformer()
    logging.info('Loaded network.')

    for batch_index, (batch_input_paths, batch_output_paths) in enumerate(
            batched_input_output_paths):
        batch_images = map(load_float_image, batch_input_paths)
        logging.info('Loaded batch {} images.'.format(batch_index))

        batch_activations = compute_activations(net, transformer, batch_images)
        logging.info('Computed activations for {}.'.format(batch_index))

        for i, output_path in enumerate(batch_output_paths):
            activations = batch_activations[i, :]
            mkdir_p(dirname(output_path))
            with open(output_path, 'wb') as f:
                pickle.dump(activations, f)
        # Usually python takes care of memory, but in this case,
        # batch_activations won't be cleared until after the next batch_images
        # are loaded, so del'ing it before hand helps reduce memory usage.
        del batch_activations
        logging.info('Deleted variables')


if __name__ == '__main__':
    main()
