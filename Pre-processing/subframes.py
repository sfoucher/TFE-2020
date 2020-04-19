import os
import torch
import torchvision
from PIL import Image
import cv2
from albumentations import Compose, BboxParams, Crop
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import time
import datetime
from datetime import date
import json
import csv
from dataset import CustomDataset

class subframes(object):
    ''' 
    Class allowing the visualisation and the cropping of a labeled 
    image (bbox) into sub-frames whose dimensions are specified 
    by the user.

    Attributes
    -----------
    img_name : str
        name of the image (with extension, e.g. "My_image.JPG").
    image : PIL
        PIL image.
    target : dict
        Must have 'boxes' and 'labels' keys at least.
        'boxes' must be a list in the 'coco' bounding box format : [[xmin, ymin, width, height], ...]
    width : int
        width of the sub-frames
    height : int
        height of the sub-frames
    
    Methods
    --------
    getlist()
        Produces a results list containing, for each row :
        the sub-frame (3D list, dtype=uint8), the bboxes (2D list),
        the labels (1D list) and the filename (str).
    visualise(results)
        Displays ordered sub-frames of the entire image.
    topoints(results)
        Converts the bounding boxes into points annotations.
    displayobjects(results, points_results, ann_type='point')
        Displays only sub-frames containing objects.
    save(results, output_path, object_only=True)
        Saves sub-frames to a specific path.

    Author
    -------
    Alexandre Delplanque
    '''

    def __init__(self, img_name, image, target, width, height):
        '''
        Parameters
        -----------
        img_name : str
            name of the image (with extension, e.g. "My_image.JPG")
        image : PIL
            PIL image
        target : dict
            Must have 'boxes' and 'labels' keys at least.
        width : int
            width of the sub-frames
        height : int
            height of the sub-frames
        '''

        self.img_name = img_name
        self.image = image
        self.target = target
        self.width = width
        self.height = height

        # Infos sur l'image
        self.img_width = image.size[0]
        self.img_height = image.size[1]

        # Subdivision
        self.x_sub = int(1+(self.img_width - (self.img_width % width)) / width)
        self.y_sub = int(1+(self.img_height - (self.img_height % height)) / height)

    def getlist(self):
        '''
        Produces a results list containing, for each row :
        the sub-frame (3D list, dtype=uint8), the bboxes (2D list),
        the labels (1D list) and the filename (str).
        Parameters
        -----------
        None
        Returns
        --------
        list
        '''
        # Initialisation de la variable des résultats
        results = []

        # Prétraitements de l'image
        # tensor_to_PIL = torchvision.transforms.ToPILImage(mode='RGB')
        # img = tensor_to_PIL(self.image)
        
        image_np = np.array(self.image)
        boxes = self.target['boxes']
        labels = self.target['labels']
        annotations = {'image':image_np,'bboxes':boxes,'labels':labels}

        # Crop de l'image
        w0 = 0
        h0 = 0
        w1 = 0
        h1 = 0
        sub = 0

        for y in range(self.y_sub):

            w0 = 0
            w1 = self.width

            if self.img_height % self.height != 0 and y == self.y_sub-1:
                h0 += self.height
                h1 += (self.img_height % self.height)
            else:
                if y == self.y_sub-1:
                    continue
                if y == 0:
                    h0 = 0
                    h1 = self.height
                else:
                    h0 += self.height
                    h1 += self.height

            for x in range(self.x_sub):
                
                if self.img_width % self.width != 0 and x == self.x_sub-1:
                    w0 += self.width
                    w1 += (self.img_width % self.width)
                else:
                    if x == self.x_sub-1:
                        continue
                    if x == 0:
                        w0 = 0
                        w1 = self.width
                    else:
                        w0 += self.width
                        w1 += self.width

                transf = Compose([Crop(w0,h0,w1,h1,p=1.0)], bbox_params=BboxParams(format='coco', label_fields=['labels']))
                augmented  = transf(**annotations)
                sub_name = self.img_name.rsplit('.')[0] + "_S" + str(sub) + ".JPG"
                results.append([augmented['image'],augmented['bboxes'],augmented['labels'],sub_name])
                sub += 1

        return results

    def visualise(self, results):
        '''
        Displays ordered sub-frames of the entire image.
        Parameters
        -----------
        results : list
            The list obtained by the method getlist().
        Returns
        --------
        matplotlib plot
        '''

        plt.figure(1)
        plt.suptitle(self.img_name)
        sub = 1
        for line in range(len(results)):

            if self.img_width % self.width != 0:
                n_col = self.x_sub
                n_row = self.y_sub
            else:
                n_col = self.x_sub - 1
                n_row = self.y_sub - 1

            plt.subplot(n_row, n_col, sub, xlim=(0,self.width), ylim=(self.height,0))
            plt.imshow(Image.fromarray(results[line][0]))
            plt.axis('off')
            plt.subplots_adjust(wspace=0.1,hspace=0.1)

            # Facteurs de position du texte en fonction des dim du crop
            text_x = np.shape(results[line][0])[1]
            text_y = np.shape(results[line][0])[0]

            # Facteur de proportion du fontsize, dépendant du crop
            if self.width > self.height:
                f = self.height
            else:
                f = self.width

            plt.text(0.5*text_x, 0.5*text_y, 
                    "S"+str(line),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=0.02*f,
                    color='w')
            sub += 1

    def topoints(self, results):
        '''
        Converts the bounding boxes into points annotations.
        Parameters
        -----------
        results : list
            The list obtained by the method getlist().
        Returns
        --------
        list
            A 2D list with headers : "id", "filename", "count",
            "locations" where
            - "id" represents the unique id of the sub-frame within 
              the image
            - "filename" is the name of the sub-frame 
              (e.g. "My_image_S1.JPG")
            - "count" is the number of objects into the sub-frame
            - "points" is a list of tuple representing the 
              locations of the objects (y,x)
    
        '''

        points_results = [['id','filename','count','locations']]
        loc = []
        for line in range(len(results)):
            # Vérification de l'existence de la bbox
            if results[line][1]:
                count = len(results[line][1])
                for bbox in range(len(results[line][1])):
                    boxe = results[line][1][bbox]
                    x = int(boxe[0]+(boxe[2])/2)
                    y = int(boxe[1]+(boxe[3])/2)
                    point = (y,x)
                    loc.append(point)
            
                sub_name = self.img_name.rsplit('.')[0] + "_S" + str(line) + ".JPG"
                points_results.append([line, sub_name, count, loc])
                loc = []

        return points_results

    def displayobjects(self, results, points_results, ann_type='point'):
        '''
        Displays only sub-frames containing objects.
        Parameters
        -----------
        results : list
            The list obtained by the method getlist().
        points_results : list
            The list obtained by the method topoints(results).
        ann_type : str, optional
            A string used to specify the annotation type. Choose
            between :
            - 'point' to visualise points
            - 'bbox' to visualise bounding boxes
            - 'both' to visualise both
            (default is 'point')
        Returns
        --------
        matplotlib plot
        '''

        sub_r = 0
        sub_c = 0

        n_row = int(np.round(math.sqrt(len(points_results)-1)))
        n_col = n_row

        if int(len(points_results)-1) > int(n_row*n_col):
            n_row += 1

        fig, ax = plt.subplots(nrows=n_row, ncols=n_col, squeeze=False)

        for r in range(n_row):
            for c in range(n_col):
                ax[r,c].axis('off')
                plt.subplots_adjust(wspace=0.1,hspace=0.1)

        for o in range(1,len(points_results)):

            id_object = points_results[o][0]
            patch_object = results[id_object][0]

            # Facteurs de position du texte en fonction des dim du crop
            text_x = np.shape(results[id_object][0])[1]
            text_y = np.shape(results[id_object][0])[0]

            # Plot
            ax[sub_r,sub_c].imshow(Image.fromarray(patch_object))
            ax[sub_r,sub_c].text(0.5*text_x, 0.5*text_y, 
                    "S"+str(id_object),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=15,
                    color='w',
                    alpha=0.6)

            if ann_type == 'point':
                points = points_results[o][3]
                for p in range(len(points)):
                    ax[sub_r,sub_c].scatter(points[p][1],points[p][0], color='r')
            
            elif ann_type == 'bbox':
                bboxes = results[id_object][1]
                for b in range(len(bboxes)):
                    rect = patches.Rectangle((bboxes[b][0],bboxes[b][1]),bboxes[b][2],bboxes[b][3], linewidth=1, edgecolor='r', facecolor='none')
                    ax[sub_r,sub_c].add_patch(rect)
                
            elif ann_type == 'both':
                points = points_results[o][3]
                bboxes = results[id_object][1]
                for b in range(len(bboxes)):
                    ax[sub_r,sub_c].scatter(points[b][1],points[b][0], color='b')
                    rect = patches.Rectangle((bboxes[b][0],bboxes[b][1]),bboxes[b][2],bboxes[b][3], linewidth=1, edgecolor='r', facecolor='none')
                    ax[sub_r,sub_c].add_patch(rect)

            else:
                raise ValueError('Annotation of type \'{}\' unsupported. Choose between \'point\',\'bbox\' or \'both\'.'.format(ann_type))
                
            if sub_c < n_col-1:
                sub_r = sub_r
                sub_c += 1
            else:
                sub_c = 0
                sub_r += 1
            
    def save(self, results, output_path, object_only=True):
        '''
        Saves sub-frames (.JPG) to a specific path.
        Parameters
        -----------
        results : list
            The list obtained by the method getlist().
        output_path : str
            The path to the folder chosen to save sub-frames.
        object_only : bool, optional
            A flag used to choose between :
            - saving all the sub-frames of the entire image
              (set to False)
            - saving only sub-frames with objects
              (set to True, default)
        Returns
        --------
        None
        '''

        for line in range(len(results)):
            if object_only is True:
                if results[line][1]:
                    subframe = Image.fromarray(results[line][0])
                    sub_name =  results[line][3]
                    subframe.save(os.path.join(output_path, sub_name))
                    
            elif object_only is not True:
                subframe = Image.fromarray(results[line][0])
                sub_name =  results[line][3]
                subframe.save(os.path.join(output_path, sub_name))

def subexport(img_root, ann_root, width, height, output_folder, ann_type, pr_rate=50, object_only=True, export_ann=True):
    '''
    Function that exports sub-frames created on the basis of 
    images loaded by a dataloader, and their associated new 
    annotations.

    This function uses the 'subframes' class for image processing.

    Parameters
    -----------
    img_root : str
        Path to images.

    ann_root : str
        Path to a coco-style dict (.json) containing annotations of 
        the initial dataset.

    width : int
        Width of the sub-frames.
    
    height : int
        Height of the sub-frames.
    
    output_folder : str
        Output folder path where to save sub-frames and new annotations.

    ann_type : 'point', 'bbox' or 'both'
        Type of the annotation to export.
        - If 'bbox' a coco-type JSON file named 'coco_subframes.json'
          is created.
        - If 'point' a CSV file named 'gt.csv' is created. See :
        https://github.com/javiribera/locating-objects-without-bboxes 
        - If 'both', the two above are created.

    pr_rate : int, optional
        Console print rate of image processing progress.
        Default : 50
    
    object_only : bool, optional
        A flag used to choose between :
            - saving all the sub-frames of the entire image
            (set to False)
            - saving only sub-frames with objects
            (set to True, default)

    export_ann : bool, optional
        A flag used to choose between :
            - not exporting annotations with sub-frames
            (set to False)
            - exporting annotations with sub-frames
            (set to True, default)
   
    Returns
    --------
    list or dict
        Specific to the annotation type. if 'both', dict.
    
    '''

    # Téléchargement des annotations (fichiers .json)
    with open(ann_root) as json_file:
        coco_dic = json.load(json_file)

    # Dataset
    dataset = CustomDataset(img_root, ann_root, target_type='coco')

    # Sampler pour loader la data dans l'ordre
    sampler = torch.utils.data.SequentialSampler(dataset)

    # Collate_fn
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=1,
                                            sampler=sampler,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    # Initialisation de "all_points" et "all_results"
    all_points = [['filename','count','locations']]
    all_results = [['filename','boxes','labels','HxW']]

    # Temps initial
    t_i = time.time()

    for i, (image, target) in enumerate(dataloader):

        if i == 0:
            print(' ')
            print('-'*30)
            print('Sub-frames creation started...')
            print('-'*30)
        elif i == len(dataloader)-1:
            print('-'*30)
            print('Sub-frames creation finished!')
            print('-'*30)

        image = image[0]
        target = target[0]

        # id de l'image
        img_id = int(target['image_id'])
        # Nom de l'image
        img_name = coco_dic['images'][img_id-1]['file_name']
        # Obtention des sub-frames
        sub_frames = subframes(img_name, image, target, width, height)
        results = sub_frames.getlist()
        # Sauvegarde
        sub_frames.save(results, output_path=output_folder, object_only=object_only)
        # Obtention du fichier de points
        points = sub_frames.topoints(results)
        points = np.array(points)[1:,1:].tolist()
        # Concaténation pour obtenir un ensemble
        for o in range(len(points)):
            all_points.append(points[o])
        
        for b in range(len(results)):
            if results[b][1]:
                h = np.shape(results[b][0])[0]
                w = np.shape(results[b][0])[1]
                all_results.append([results[b][3],results[b][1],results[b][2],[h,w]])

        if i % pr_rate == 0:
            print('Image [{:<3}/{:<3}] done.'.format(i, len(coco_dic['images'])))

    # Temps final
    t_f = time.time()

    print('Elapsed time : {}'.format(str(datetime.timedelta(seconds=int(np.round(t_f-t_i))))))
    print('-'*30)
    print(' ')

    if ann_type == 'point' or ann_type == 'both':
        # Return
        return_var = all_points

        # Export du fichier de points en .csv
        if export_ann is True:
            file_name = 'gt.csv'
            output_f = os.path.join(output_folder, file_name)
            with open(output_f, 'w', newline="") as outputfile:
                writer = csv.writer(outputfile, delimiter = ',')
                writer.writerows(all_points)
        
            if os.path.isfile(output_f) is True:
                print('File \'{}\' correctly saved at \'{}\'.'.format(file_name, output_folder))
                print(' ')
            else:
                print('An error occurs, file \'{}\' not found at \'{}\'.'.format(file_name, output_folder))

        
    if ann_type == 'bbox' or ann_type == 'both':

        # Return
        return_var = np.array(all_results)[:,:3].tolist()

        # Export du fichier d'annotations adapté aux nouvelles sous-images
        if export_ann is True:
            file_name = 'coco_subframes.json'
            output_f = os.path.join(output_folder, file_name)

            # Initialisations
            images = []
            annotations = []
            id_img = 0
            id_ann = 0

            for i in range(1,len(all_results)):
                
                id_img += 1

                # Dictionnaire de l'image i
                h = all_results[i][3][0]
                w = all_results[i][3][1]

                dico_img = {
                    "license": 1,
                    "file_name": all_results[i][0],
                    "coco_url": "None",
                    "height": h,
                    "width": w,
                    "date_captured": "None",
                    "flickr_url": "None",
                    "id": id_img
                }

                # Concaténation du dictionnaire avec les autres images
                images.append(dico_img)

                # Bounding boxes
                bndboxes = all_results[i][1]

                # Dictionnaire des annotations de l'image i
                for b in range(len(bndboxes)):

                    id_ann += 1

                    # Bounding box
                    bndbox = bndboxes[b]
                    
                    # Conversion de la bounding box
                    x_min = int(np.round(bndbox[0]))
                    y_min = int(np.round(bndbox[1]))
                    box_w = int(np.round(bndbox[2]))
                    box_h = int(np.round(bndbox[3]))

                    coco_box = [x_min,y_min,box_w,box_h]

                    # Calcul de l'aire de la bounding box
                    area = box_w*box_h

                    # Label
                    label_id = all_results[i][2][b]

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
            
            # Changement de date dans la clé "info" du coco_dic
            coco_dic['info']['date_created'] = str(date.today())

            new_dic = {
                'info': coco_dic['info'],
                'licenses': coco_dic['licenses'],
                'images': images,
                'annotations': annotations,
                'categories': coco_dic['categories']
            }

            # Export du dict en .json
            with open(output_f, 'w') as outputfile:
                json.dump(new_dic, outputfile)

            if os.path.isfile(output_f) is True:
                print('File \'{}\' correctly saved at \'{}\'.'.format(file_name, output_folder))
                print(' ')
            else:
                print('An error occurs, file \'{}\' not found at \'{}\'.'.format(file_name, output_folder))

    
    if ann_type == 'both':
        return_var = {
            'bbox': np.array(all_results)[:,:3].tolist(),
            'point': all_points
        }

    return return_var
