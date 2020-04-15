import numpy as np
import os
import argparse
from parse import parse
import textwrap


def parse_cmdline_args(py_code):

    if py_code == 'subframes_maker':

        # subframes_maker settings
        parser = argparse.ArgumentParser(
            prog='PROG',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\

                 ---------------------------------------------------------                   
                                    SUB-FRAMES CREATOR 
                 ---------------------------------------------------------               
                 
                 Function that exports sub-frames, created on the basis of 
                 images loaded by a dataloader, and their associated new 
                 annotations.

                 ---------------------------------------------------------

                '''))

        required_args = parser.add_argument_group('Required arguments')

        required_args.add_argument('--img-path',
                                    required=True,
                                    help='Path of the directory with images to process. '
                                        'Must contain image files (any format).')

        required_args.add_argument('--ann-path',
                                    required=True,
                                    help='Path to the annotation file. '
                                        'File must be in JSON format.')

        required_args.add_argument('--out-dir',
                                    required=True,
                                    help='Directory where images will be saved.')

        required_args.add_argument('--ann-type',
                                    type=str,
                                    required=True,
                                    help='Type of annotation to save. '
                                        'Choose between \'bbox\', \'point\' or \'both\'.')

        required_args.add_argument('--size',
                                    type=str,
                                    required=True, 
                                    metavar='HxW',
                                    help='Size of the sub_frames '
                                        '(height x width).')

        args = parser.parse_args()

        args.height, args.width = parse('{}x{}', args.size)
        args.height, args.width = int(args.height), int(args.width)
                
    elif py_code == 'analyse':

        # COCO annotations analyse settings
        parser = argparse.ArgumentParser(
            prog='PROG',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\

                 -----------------------------------------------------------------------------
                                            COCO-ANNOTATIONS ANALYSE
                 -----------------------------------------------------------------------------                
                 
                 Function allowing the analysis of COCO-type object detection annotation file 
                 (JSON format).

                 To create and/or adapt your annotations to COCO-style, see: 
                 https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch 

                 -----------------------------------------------------------------------------

                '''))

        required_args = parser.add_argument_group('Required arguments')

        required_args.add_argument('--ann-path',
                                    required=True,
                                    metavar='PATH',
                                    help='Path to the annotation file. '
                                        'File must be in JSON format.')

        optional_args = parser._action_groups.pop()

        optional_args.add_argument('--out-path',
                                   metavar='PATH',
                                   type=str,
                                   default=None,
                                   help='Path to the folder where results will be saved.')

        optional_args.add_argument('--xa-lab',
                                   default=None,
                                   type=str,
                                   metavar='LABEL',
                                   help='Label of the annotation plot x axis.')

        optional_args.add_argument('--ya-lab',
                                   default=None,
                                   metavar='LABEL',
                                   help='Label of the annotation plot y axis.')

        optional_args.add_argument('--a-title',
                                   default=None,
                                   type=str,
                                   metavar='TITLE',
                                   help='Title of the annotation plot.')

        optional_args.add_argument('--xc-lab',
                                   default=None,
                                   type=str,
                                   metavar='LABEL',
                                   help='Label of the categories plot x axis.')

        optional_args.add_argument('--yc-lab',
                                   default=None,
                                   type=str,
                                   metavar='LABEL',
                                   help='Label of the categories plot y axis.')

        optional_args.add_argument('--c-title',
                                   default=None,
                                   type=str,
                                   metavar='TITLE',
                                   help='Title of the categories plot.')

        parser._action_groups.append(optional_args)
        args = parser.parse_args()

    return args
