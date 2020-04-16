import os 
import json
import glob
from datetime import date

work_dir = "D:\Annotations_Colabeler"

ann_folders = ['train','validation','test']

# Initialisation des dictionnaires
info = {
    "description": "TFE Alexandre Delplanque Dataset",
    "url": "None",
    "version": "1.0",
    "year": 2020,
    "contributor": "None",
    "date_created": str(date.today())
}

licences = [
    {
        "url": "None",
        "id": 1,
        "name": "Property of University of Gembloux Agro-Bio Tech"
    }
]

categories = [
    {"supercategory": "animal","id": 1,"name": "bubale"},
    {"supercategory": "animal","id": 2,"name": "buffalo"},
    {"supercategory": "animal","id": 3,"name": "hippopotamus"},
    {"supercategory": "animal","id": 4,"name": "kob"},
    {"supercategory": "animal","id": 5,"name": "topi"},
    {"supercategory": "animal","id": 6,"name": "warthog"},
    {"supercategory": "animal","id": 7,"name": "waterbuck"}
]

# Dictionnaires contenant les annotations des 3 jeux de données
dicos = {}

# Boucle sur les dossiers
for dataset in range(len(ann_folders)):

    # Chemin d'accès vers le dossier d'annotations Colabeler
    ann_path = os.path.join(work_dir, ann_folders[dataset])

    # Définir le directory
    os.chdir(ann_path)

    # List des fichiers json du dossier
    ann_files = glob.glob("*.json")

    # Dictionnaires des images
    images = []

    # Dictionnaire des annotations
    annotations =[]

    # Ouverture des fichiers json 1 par 1
    id_img = 0
    id_ann = 0
    for f in range(len(ann_files)):

        with open(ann_files[f]) as json_file:
            ann_file = json.load(json_file)
        
        if ann_file["labeled"] == True:

            id_img += 1

            # Dictionnaire de l'image f

            dico_img = {
                "license": 1,
                "file_name": ann_files[f][:-5] + ".JPG",
                "coco_url": "None",
                "height": ann_file["size"]["height"],
                "width": ann_file["size"]["width"],
                "date_captured": "None",
                "flickr_url": "None",
                "id": id_img
            }

            # Concaténation du dictionnaire avec les autres images
            images.append(dico_img)

            # Dictionnaire des annotations de l'image f

            for o in range(len(ann_file["outputs"]["object"])):

                id_ann += 1
                
                # Conversion de la bounding box
                bndbox = ann_file["outputs"]["object"][o]["bndbox"]

                # Respect des limites de dimensions de l'image
                xmin = bndbox['xmin']
                ymin = bndbox['ymin']
                xmax = bndbox['xmax']
                ymax = bndbox['ymax']

                x_lim_sup = ann_file["size"]["width"]
                y_lim_sup = ann_file["size"]["height"]

                if xmin < 0:
                    xmin = 0
                elif ymin < 0:
                    ymin = 0
                elif xmax > x_lim_sup:
                    xmax = 6000
                elif ymax > y_lim_sup:
                    ymax = 4000

                box_w = xmax - xmin
                box_h = ymax - ymin

                coco_box = [xmin,ymin,box_w,box_h]

                # Calcul de l'aire de la bounding box
                area = box_w*box_h

                # Définition de l'identifiant du label (cfr. categories)
                for label in range(len(categories)):
                    if categories[label]["name"] == ann_file["outputs"]["object"][o]["name"]:
                        label_id = categories[label]["id"]

                # Ajout dans le dictionnaire partiel
                dico_ann = {
                        "segmentation": [[]],
                        "area": area,
                        "iscrowd": 0,
                        "image_id": id_img,
                        "bbox": coco_box,
                        "category_id": label_id,
                        "id": id_ann
                }

                # Concaténation du dictionnaire avec les autres annotations
                annotations.append(dico_ann)

    coco_dico = {
        "info": info,
        "licenses": licences,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    dicos.update({ann_folders[dataset] : coco_dico})

# --- Export en fichier .json ---
from os import path

output_path = os.path.join(work_dir,"COCO_type")

if path.exists(output_path)==True:
    pass
else :
    os.mkdir(output_path)

os.chdir(output_path)

with open("train_cocotype.json","w") as j:
    json.dump(dicos['train'], j)

with open("val_cocotype.json","w") as j:
    json.dump(dicos['validation'], j)

with open("test_cocotype.json","w") as j:
    json.dump(dicos['test'], j)
