import json
from subframes import subframes, subexport
import argparser

args = argparser.parse_cmdline_args('subframes_maker')

#----------------------------------------------------------------  
# Chemin d'accès vers les images
img_root = args.img_path
# Chemin d'accès vers le fichier d'annotations
ann_root = args.ann_path
# Chemin d'accès du dossier de sorties
out_dir = args.out_dir
# Hauteur et largeur des sub-frames
width = args.width
height = args.height
# Type d'annotation en output
ann_type = args.ann_type
#----------------------------------------------------------------  

# Export des sub-frames et nouvelles annotations
results = subexport(img_root, ann_root, width, height, out_dir, ann_type)
