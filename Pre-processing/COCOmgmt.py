import json
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import csv
from itertools import chain

class COCOmgmt(object):
    '''
    Class which allows the display and management of a COCO 
    annotation file in JSON format.

    Attributes
    ----------
    coco_path : str
        Path to the JSON annotation file.

    Methods
    -------
    getinfo()
        Print the annotations file information.
    displaycatdist(x_lab=None, y_lab=None, title=None, output_path=None)
        Displays the distribution of annotations by category 
        (histogram).
    displayanndist(x_lab=None, y_lab=None, title=None, output_path=None)
        Displays the distribution of the number of images according 
        to the number of annotations per image (histogram).
    adjustcat(adjust, output_path=None)
        Function used to adjust the categories of an annotation 
        file (JSON format).
    groupcat(categories, group, output_path=None)
        Function used for grouping categories of an annotation file
        (JSON format).
    
    Author
    ------
    Alexandre Delplanque
    '''

    def __init__(self, coco_path):
        '''
        Parameter
        ----------
        coco_path : str
            Path to the JSON annotation file.
        '''

        self.coco_path = coco_path

        with open(coco_path, 'r') as json_file:
            coco_dic = json.load(json_file)

        self.coco_dic = coco_dic
        self.info = coco_dic['info']
        self.licenses = coco_dic['licenses']
        self.images = coco_dic['images']
        self.annotations = coco_dic['annotations']
        self.categories = coco_dic['categories']

    def getinfo(self):
        '''
        Print the annotations file information.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''

        info = self.info
        licenses = self.licenses
        images = self.images
        annotations = self.annotations
        categories = self.categories

        # Résumé des infos
        print('-'*3)
        print('{:<15} | {}'.format('Description', info['description']))
        print('{:<15} | {}'.format('Version', info['version']))
        print('{:<15} | {}'.format('Contributor(s)', info['contributor']))
        print('{:<15} | {}'.format('URL', info['url']))
        print('{:<15} | {}'.format('Creation', info['date_created']))
        print('{:<15} | {}'.format('Images', len(images)))
        print('{:<15} | {}'.format('Annotations', len(annotations)))
        print('{:<15} | {}'.format('Categories', len(categories)))
        print('-'*3)

    def displaycatdist(self, x_lab=None, y_lab=None, title=None, output_path=None):
        '''
        Displays the distribution of annotations by category 
        (histogram).

        Parameters
        ----------
        x_lab : str, optional
            X-axis label, default : None, means 'Categories' on the 
            plot.
        y_lab : str, optional
            Y-axis label, default : None, means 'Count' on the 
            plot.
        title : str, optional
            Plot's title, default : None
        output_path : str, optional
            Path to save the plot and its numerical data.
            Create a folder ('cocoanalyse_results') at the path 
            mentioned, including:
                - Plot in PNG format 
                    (file : 'categories_dist.png')
                - Numerical data in CSV format 
                    (file : 'categories_dist.csv')

        Returns
        -------
        None

        '''

        info = self.info
        licenses = self.licenses
        images = self.images
        annotations = self.annotations
        categories = self.categories

        # Distributions du nombre d'annotations selon les categories
        # ---
        # list contenant les id de chaque annotation
        cat_dist = []
        for a in range(len(annotations)):
            cat_dist.append(annotations[a]['category_id']) 

        # somme du nombre d'annotations par id
        cat_counter = dict(collections.Counter(cat_dist))

        # list des noms correspondant aux id
        n = []
        for c in range(len(categories)):
            n.append([categories[c]['id'],categories[c]['name']])

        n.sort()
        names = list(np.array(n)[:,-1])

        # plot
        fig, ax = plt.subplots(1, figsize=(7,7))
        bins = [x + 0.5 for x in range(0,len(categories)+1)]
        ax.hist(cat_dist, bins=bins, rwidth=0.7, edgecolor=None, color=['cadetblue'])

        if x_lab is None:
            ax.set_xlabel('Categories')
        else:
            ax.set_xlabel(x_lab)

        if y_lab is None:
            ax.set_ylabel('Count')
        else:
            ax.set_ylabel(y_lab)

        if title is not None:
            ax.set_title(title)

        x_ticks = ax.set_xticks(range(1,len(categories)+1))
        ticks = ax.set_xticklabels(names)

        # enregistrements

        if output_path is not None:
            # Sauvegarde de la figure
            save_path = os.path.join(output_path,'cocoanalyse_results')

            if path.exists(save_path) is not True:
                os.mkdir(save_path)
            
            png_file = os.path.join(save_path,'categories_dist.png')
            
            plt.savefig(png_file)

            # Sauvegarde des résultats numériques
            results = [['id','name','count']]
            res = [[key,value] for (key,value) in sorted(cat_counter.items())]
            for row in res:
                for line in n:
                    if row[0]==line[0]:
                        results.append([row[0],line[1],row[1]])

            csv_file = os.path.join(save_path,'categories_dist.csv')

            with open(csv_file,'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(results)

            if os.path.isfile(png_file) is True and os.path.isfile(csv_file) is True :
                print('Files \'{}\' and \'{}\' correctly saved in new folder at directory\'{}\'.'.format('categories_dist.png','categories_dist.csv', save_path))
                print(' ')
            else:
                print('An error occurs, please verify statut of \'{}\'.'.format(save_path))

    def displayanndist(self, x_lab=None, y_lab=None, title=None, output_path=None):
        '''
        Displays the distribution of the number of images according 
        to the number of annotations per image (histogram).

        Parameters
        ----------
        x_lab : str, optional
            X-axis label, default : None, means 'Number of annotations'
            on the plot.
        y_lab : str, optional
            Y-axis label, default : None, means 'Number of images' on 
            the plot.
        title : str, optional
            Plot's title, default : None
        output_path : str, optional
            Path to save the plot and its numerical data.
            Create a folder ('cocoanalyse_results') at the path 
            mentioned, including:
                - Plot in PNG format 
                    (file : 'annotations_dist.png')
                - Numerical data in CSV format 
                    (file : 'annotations_dist.csv')

        Returns
        -------
        None
        
        '''

        info = self.info
        licenses = self.licenses
        images = self.images
        annotations = self.annotations
        categories = self.categories
        
        # Distributions du nombre d'annotations par image
        # ---
        # initialisations
        nb_ann = 0
        ann_dist = []
        results = [['id','filename','count']]

        # list du nombre d'annotations par image
        for image in images:
            img_id = image['id']
            img_name = image['file_name']

            for ann in annotations:
                if ann['image_id']==img_id:
                    nb_ann += 1
            
            ann_dist.append(int(nb_ann))
            results.append([img_id,img_name,nb_ann])

            nb_ann = 0

        # somme du nombre d'annotations par image
        ann_counter = dict(collections.Counter(ann_dist))

        # plot
        fig, ax = plt.subplots(1, figsize=(7,7))
        bins = [x + 0.5 for x in range(0, len(ann_counter)+1)]
        ax.hist([ann_dist], bins=bins, edgecolor='w', color=['cadetblue'])

        if x_lab is None:
            ax.set_xlabel('Number of annotations')
        else:
            ax.set_xlabel(x_lab)

        if y_lab is None:
            ax.set_ylabel('Number of images')
        else:
            ax.set_ylabel(y_lab)

        if title is not None:
            ax.set_title(title)

        # enregistrements

        if output_path is not None:
            # Sauvegarde de la figure
            save_path = os.path.join(output_path,'cocoanalyse_results')

            if path.exists(save_path) is not True:
                os.mkdir(save_path)
            
            png_file = os.path.join(save_path,'annotations_dist.png')

            plt.savefig(png_file)

            # Sauvegarde des résultats numériques
            csv_file = os.path.join(save_path,'annotations_dist.csv')
            with open(csv_file,'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(results)
            
            if os.path.isfile(png_file) is True and os.path.isfile(csv_file) is True :
                print('Files \'{}\' and \'{}\' correctly saved in new folder at directory\'{}\'.'.format('annotations_dist.png','annotations_dist.csv', save_path))
                print(' ')
            else:
                print('An error occurs, please verify statut of \'{}\'.'.format(save_path))

    def adjustcat(self, adjust, output_path=None):
        '''
        Function used to adjust the categories of an annotation
        file (JSON format).

        Parameters
        ----------
        adjust : list of dict
            New 'categories' dict in COCO format:
                [
                    {
                        "supercategory": ...,
                        "id": ...,
                        "name": ...,
                    },
                    {
                        ...
                    }
                ]
            Warning: Must match the original categories length!
        output_path : str, optional
            Path to save the entire new dictionary in JSON format.
            This path must contain the file name and the extension
            '.json'.

        Returns
        -------
        dict
            COCO format new dictionary.
        '''

        # S'assurer que la longueur du nouveau dict correspond à l'ancien
        assert len(adjust)==len(self.categories), 'New categories length doesn\'t match the old ones.'

        # Nouveau dict
        new_dic = self.coco_dic.copy()

        new_dic['categories'] = adjust

        # Export du dict en .json
        if output_path is not None:
            with open(output_path, 'w') as outputfile:
                json.dump(new_dic, outputfile)

        return new_dic
    
    def groupcat(self, categories, group, output_path=None):
        '''
        Function used for grouping categories of an annotation file
        (JSON format).

        Parameters
        ----------
        categories : list of dict
            New 'categories' dict in COCO format:
                [
                    {
                        "supercategory": ...,
                        "id": ...,
                        "name": ...,
                    },
                    {
                        ...
                    }
                ]
        group : dict
            Dictionary containing lists as values, which groups 
            together the IDs of the original categories.
            Example:
                {
                    1: [2,5],
                    2: [3],
                    ...
                }
            Warning: Keys must be integer and sorted!
        output_path : str, optional
            Path to save the entire new dictionary in JSON format.
            This path must contain the file name and the extension
            '.json'.

        Returns
        -------
        dict
            COCO format new dictionary.
        '''

        # S'assurer que les inputs correspondent
        assert len(categories)==len(group) , "\'new_cat\' length doesn\'t match \'regroup\' length."

        # S'assurer que les regroupements correspondent au dict initial, sans d'oubli
        all_group = []
        for r in range(1,len(group)+1):
            all_group.append(group[r])

        all_group = sorted(list(chain.from_iterable(all_group)))

        old_ids = []
        for cat in self.categories:
            old_ids.append(cat['id'])

        assert all_group == old_ids , "regroup\'s categories IDs do not match original ones."

        # Nouveau dict
        new_dic = self.coco_dic.copy()

        # Remplacement des categories
        new_dic['categories'] = categories

        # Remplacement des ids des annotations
        new_ann = []
        for ann in new_dic['annotations']:
            for n in range(1,len(group)+1):
                if (ann['category_id'] in group[n]) is True:

                    anno = {'segmentation': ann['segmentation'],
                            'area': ann['area'],
                            'iscrowd': ann['iscrowd'],
                            'image_id': ann['image_id'],
                            'bbox': ann['bbox'],
                            'category_id': n,
                            'id': ann['id']}

                    new_ann.append(anno)

        new_dic['annotations'] = new_ann

        # Export du dict en .json
        if output_path is not None:
            with open(output_path, 'w') as outputfile:
                json.dump(new_dic, outputfile)

        return new_dic
