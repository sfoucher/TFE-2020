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
# Overlap
overlap = args.overlap
# Uniquement objets ?
object_only = args.obj_only
#----------------------------------------------------------------  

# Export des sub-frames et nouvelles annotations
results = subexport(img_root=img_root,
                    ann_root=ann_root, 
                    width=width, 
                    height=height, 
                    output_folder=out_dir, 
                    ann_type=ann_type, 
                    overlap=overlap, 
                    object_only=object_only)
