import numpy as np
import os
import argparse
from parse import parse
import textwrap

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_cmdline_args(py_code):

    if py_code == 'subframes_maker':

        # subframes_maker settings
        parser = argparse.ArgumentParser(
            prog='Sub-frames Creator',
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
                                    metavar='PATH',
                                    help='Path of the directory with images to process. '
                                        'Must contain image files (any format).')

        required_args.add_argument('--ann-path',
                                    required=True,
                                    metavar='PATH',
                                    help='Path to the annotation file. '
                                        'File must be in JSON format.')

        required_args.add_argument('--out-dir',
                                    required=True,
                                    metavar='PATH',
                                    help='Directory where images will be saved.')

        required_args.add_argument('--ann-type',
                                    type=str,
                                    required=True,
                                    metavar='TYPE',
                                    help='Type of annotation to save. '
                                        'Choose between \'bbox\', \'point\' or \'both\'.')

        required_args.add_argument('--size',
                                    type=str,
                                    required=True, 
                                    metavar='HxW',
                                    help='Size of the sub_frames '
                                        '(height x width).')

        parser.add_argument('--overlap',
                            metavar='BOOL',
                            type=str2bool,
                            default=False,
                            help='Bool, set to true to get an overlap '
                                'of 50 percent between sub-frames.')
        
        parser.add_argument('--obj-only',
                            metavar='BOOL',
                            type=str2bool,
                            default=True,
                            help='Bool, set to false to export all '
                                'sub-frames and associated annotation '
                                'file.')

        args = parser.parse_args()

        args.height, args.width = parse('{}x{}', args.size)
        args.height, args.width = int(args.height), int(args.width)
                
    elif py_code == 'analyse':

        # COCO annotations analyse settings
        parser = argparse.ArgumentParser(
            prog='COCO-Annotations Analyse',
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

        parser.add_argument('--out-path',
                                   metavar='PATH',
                                   type=str,
                                   default=None,
                                   help='Path to the folder where results will be saved.')

        parser.add_argument('--xa-lab',
                                   default=None,
                                   type=str,
                                   metavar='LABEL',
                                   help='Label of the annotation plot x axis.')

        parser.add_argument('--ya-lab',
                                   default=None,
                                   metavar='LABEL',
                                   help='Label of the annotation plot y axis.')

        parser.add_argument('--a-title',
                                   default=None,
                                   type=str,
                                   metavar='TITLE',
                                   help='Title of the annotation plot.')

        parser.add_argument('--xc-lab',
                                   default=None,
                                   type=str,
                                   metavar='LABEL',
                                   help='Label of the categories plot x axis.')

        parser.add_argument('--yc-lab',
                                   default=None,
                                   type=str,
                                   metavar='LABEL',
                                   help='Label of the categories plot y axis.')

        parser.add_argument('--c-title',
                                   default=None,
                                   type=str,
                                   metavar='TITLE',
                                   help='Title of the categories plot.')

        args = parser.parse_args()

    return args
