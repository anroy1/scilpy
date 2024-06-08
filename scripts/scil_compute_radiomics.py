#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**Still a work in progress**

Compute a density map and binary from a streamlines file then extract radiomics features (GLRLM only)

This script correctly handles compressed streamlines.
"""
import argparse

import numpy as np
import nibabel as nib
import six

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from pyradiomics.radiomics import glrlm, imageoperations

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.')
    p.add_argument('in_image',
                   help='Path to an optional input image for radiomic feature calculation')
    p.add_argument('param_file',
                   help='Path to parameter file (.yml/.yaml or .json')
    p.add_argument('out_img',
                   help='path of the output image file.')

    p.add_argument('--binary', metavar='FIXED_VALUE', type=int,
                   nargs='?', const=1,
                   help='If set, will store the same value for all intersected'
                        ' voxels, creating a binary map.\n'
                        'When set without a value, 1 is used (and dtype uint8).\n'
                        'If a value is given, will be used as the stored value.')
    
    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle, optional=args.reference)
    assert_outputs_exist(parser, args, args.out_img)

    if args.in_bundle is None and args.in_image is None:
        parser.error("Either '--in_bundle' or 'in_image' must be provided")

    max_ = np.iinfo(np.int16).max
    if args.binary is not None and (args.binary <= 0 or args.binary > max_):
        parser.error('The value of --binary ({}) '
                     'must be greater than 0 and smaller or equal to {}'
                     .format(args.binary, max_))

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines
    transformation, dimensions, _, _ = sft.space_attributes

    streamline_count = compute_tract_counts_map(streamlines, dimensions)
    streamline_mask = streamline_count

    dtype_to_use = np.int32
    if args.binary is not None:
        if args.binary == 1:
            dtype_to_use = np.uint8
        streamline_mask[streamline_mask > 0] = args.binary

    # Crop image
    # bb is bounding box, upon which image and mask are cropped
    bb, correctedMask = imageoperations.checkMask(streamline_count, streamline_mask, label=1)
    if correctedMask is not None:
        streamline_mask = correctedMask
    croppedImage, croppedMask = imageoperations.cropToTumorMask(streamline_count, streamline_mask, bb)

    glrlmFeatures = glrlm.RadiomicsGLRLM(croppedImage, croppedMask, args.param_file)
    glrlmFeatures.enableAllFeatures()

    # Calculate features and print
    result = glrlmFeatures.execute()

    print('Calculated GLSZM features')
    for key, value in six.iteritems(result):
        print('\t', key, ':', value)

if __name__ == "__main__":
    main()
