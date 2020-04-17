from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import os
from os.path import exists, join, basename, splitext
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def display(predictions, img_path, coco_path, legend=True, title=True, score_thresh=None):
    '''
    Function allowing the visualisation of the results
    of MMDetection's tool predictions, and the associated
    ground-truth. 

    Parameters
    ----------
    predictions : list
        Result of mmdet.apis.inference_detector(model, image).
    img_path : str
        Path to the image.
    coco_path : str
        Path to the COCO-style annotation file in JSON format.
    legend : bool, optional
        Set to False to remove the legend (default: True)
    title : bool, optional
        Set to False to remove title (default: True).
        Title is : "Image : <image name>".
    score_tresh : float, optional
        Threshold to apply to scores of predictions (default: None).
    
    Returns
    -------
    matplotlib.pyplot

    Author
    ------
    Alexandre Delplanque
    '''

    # Ground-truth
    with open(coco_path,'r') as json_file:
        coco_dic = json.load(json_file)

    labels_dic = coco_dic['categories']
    images_dic = coco_dic['images']
    ann_dic = coco_dic['annotations']

    cls_names = []
    for cls in labels_dic:
        cls_names.append(cls['name'])

    for im in images_dic:
        if im['file_name']==basename(img_path):
            id_img = im['id']

    gt_boxes = []
    gt_labels = []
    for ann in ann_dic:
        if ann['image_id']==id_img:
            gt_boxes.append(ann['bbox'])
            gt_labels.append(ann['category_id'])

    # Predictions
    boxes = []
    labels = []
    scores = []
    for n_class in range(len(cls_names)):
        for n_box in range(len(predictions[n_class])):
            box = list(predictions[n_class][n_box][:4])
            score = predictions[n_class][n_box][4]
            boxes.append(box)
            labels.append(n_class+1)
            scores.append(score)

    predictions = {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }

    # Threshold des scores
    if score_thresh is not None:

        boxes = []
        labels = []
        scores = []
        i = 0
        for score in predictions['scores']:

            if score > score_thresh:

                boxes.append(predictions['boxes'][i])
                labels.append(predictions['labels'][i])
                scores.append(predictions['scores'][i])

            i +=1 

        predictions = {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
        }

    # Image
    image = Image.open(img_path)
    image_name = splitext(basename(img_path))[0]

    # Plot de l'image
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.axis('off')
    ax.imshow(image)

    # Fonctions
    def draw_outline(obj, linewidth):
        obj.set_path_effects([patheffects.Stroke(linewidth=linewidth,
                                                foreground='black'),
                              patheffects.Normal()])

    def draw_text(ax, xy, txt, color, size=11):
        text = ax.text(*xy, txt,
                      horizontalalignment='left',
                      color=color,
                      fontsize=size,
                      weight='bold')
        draw_outline(text,1)

    def draw_rect(box, color, linestyle, txt=None):
        rect = patches.Rectangle(box[:2],(box[2]-box[0]),(box[3]-box[1]),
                                linewidth=1.5, edgecolor=color, 
                                linestyle=linestyle,
                                facecolor='none')
        rect = ax.add_patch(rect)
        if txt is not None:
            draw_text(ax, (box[0],box[1]-2), txt, color)

    # Couleurs
    labels_color = ['b','r','c','m','orange','lime','aquamarine','peru','silver']

    # Ajout des boxes de la gt
    i = 0
    for box in gt_boxes:
        label = gt_labels[i]
        color = labels_color[label-1]
        boxe = [box[0],box[1],box[0]+box[2],box[1]+box[3]]
        draw_rect(boxe, color, '-')
        i += 1

    # Ajout des boxes prédites
    i = 0
    color_names = []
    for box in predictions['boxes']:
        # Couleur bboxe et nom pour légende
        label = int(predictions['labels'][i])
        color = labels_color[label-1]
        name = labels_dic[label-1]['name']
        
        txt = '{:.2f}'.format(predictions['scores'][i])
        draw_rect(box, color, '--', txt)
        i += 1

    # Légende = boxes et couleur
    if legend == True:

        rect_solid = patches.Rectangle((0,0), 1, 1,linewidth=1,
                                      edgecolor='black',linestyle='-',
                                      facecolor='none')
        rect_dashed = patches.Rectangle((0,0), 1, 1,linewidth=1,
                                        edgecolor='black',linestyle='--',
                                        facecolor='none')
        extra = patches.Rectangle((0,0), 1, 1,linewidth=1,
                                        edgecolor='none',
                                        facecolor='none')

        rects = []        
        for o in range(len(cls_names)):
            rect = patches.Rectangle((0,0), 1, 1,linewidth=1,
                                    edgecolor=labels_color[o],
                                    facecolor='none')
            rects.append(rect)

        ax.legend([rect_solid, rect_dashed, extra]+rects, 
                  ['Ground-truth','Prediction', '-----------']+cls_names, 
                  bbox_to_anchor=(1.04,1), 
                  fontsize=13)

    # Titre = image
    if title == True:
        plt.title('Image : {}'.format(image_name), 
                  fontsize=20, weight='bold',pad=15.0)

def match(predictions, img_path, coco_path, IoU_tresh, score_thresh=None):
    '''
    Function used to match the ground-truth bounding boxes 
    to predicted ones. The outputs are used to construct a
    confusion matrix.

    Parameters
    ----------
    predictions : list
        Result of mmdet.apis.inference_detector(model, image).
    img_path : str
        Path to the image.
    coco_path : str
        Path to the COCO-style annotation file in JSON format.
    IoU_tresh : float
        IoU treshold.
    score_tresh : float, optional
        Threshold to apply to scores of predictions (default: None).

    Returns
    -------
    2D list
        Matching between gt and predicted bbox

    Author
    ------
    Alexandre Delplanque
    '''

    # Ground-truth
    with open(coco_path,'r') as json_file:
        coco_dic = json.load(json_file)

    labels_dic = coco_dic['categories']
    images_dic = coco_dic['images']
    ann_dic = coco_dic['annotations']

    cls_names = []
    for cls in labels_dic:
        cls_names.append(cls['name'])

    for im in images_dic:
        if im['file_name']==basename(img_path):
            id_img = im['id']

    gt_boxes = []
    gt_labels = []
    for ann in ann_dic:
        if ann['image_id']==id_img:
            box = ann['boxes']
            boxes = [box[0],box[1],box[0]+box[2],box[1]+box[3]]
            gt_boxes.append(boxes)
            gt_labels.append(ann['category_id'])
    
    gt = {
        'boxes': gt_boxes,
        'labels': gt_labels
    }
    
    # Predictions
    boxes = []
    labels = []
    scores = []
    for n_class in range(len(cls_names)):
        for n_box in range(len(predictions[n_class])):
            box = list(predictions[n_class][n_box][:4])
            score = predictions[n_class][n_box][4]
            boxes.append(box)
            labels.append(n_class+1)
            scores.append(score)

    preds = {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }

    # Threshold des scores
    if score_thresh is not None:

        boxes = []
        labels = []
        scores = []
        i = 0
        for score in preds['scores']:

            if score > score_thresh:

                boxes.append(preds['boxes'][i])
                labels.append(preds['labels'][i])
                scores.append(preds['scores'][i])

            i +=1 

        preds = {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
        }

    p_boxes = preds['boxes']

    # Calcul des IoU et rangement des résultats dans un tableau
    p_index = []
    res=[]
    i=0
    for gt_box in gt['boxes']:
        
        # Index de la gt box
        id_gt = i

        # Label de gt box
        gt_label = int(gt['labels'][i])

        # Initialisation du vecteur contenant les IoUs des predictions
        p_iou = []

        # S'il n'y a aucune bbox prédite => FN
        if len(p_boxes)==0:
            res.append([i, gt_label, int(0), int(0), int(0)])
            continue

        for p_box in p_boxes:
            
            # Détermination des coord. (x,y) de l'intersection
            xA = max(gt_box[0],p_box[0])
            yA = max(gt_box[1],p_box[1])
            xB = min(gt_box[2],p_box[2])
            yB = min(gt_box[3],p_box[3])

            # Aire de cette intersection
            area = max(0, xB - xA +1) * max(0, yB - yA +1)

            # Aires de la GT et de la pred
            area_gt = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
            area_p = (p_box[2] - p_box[0] + 1) * (p_box[3] - p_box[1] + 1)

            # Calcul de l'IoU
            IoU = area / float(area_gt + area_p - area)

            # Ajout au vecteur des IoU des preds
            p_iou.append(IoU)

        # Index de l'objet de correspondance. On prend l'IoU max 
        # (correspondance maximale avec la GT).
        p_iou_max = float(max(p_iou))
        index = p_iou.index(p_iou_max)

        p_index.append(index)
        
        # Si l'IoU est trop faible, considéré comme background (FN)
        if p_iou_max < IoU_tresh:
            p_label = int(0)
        else:
            p_label = int(preds['labels'][index])

        res.append([id_gt, gt_label, index, p_label, p_iou_max])

        i += 1

    # Index de p_boxes sans correspondance avec la gt (FP) = p_not_used
    if len(p_boxes) > len(gt['boxes']):
        p_index_unique = np.unique(p_index)
        p_boxes_index = list(range(len(p_boxes)))

        # On retire les index rencontrés, il reste ceux sans correspondance
        p_not_used = [item for item in p_boxes_index if item not in p_index_unique]

        # Ajout à la variable res
        for k in range(len(p_not_used)):
            new_index = p_not_used[k]
            label = int(preds['labels'][new_index])
            res.append([i, int(0), new_index, label, int(0)])
            i += 1

    matching = np.array(res)
    matching = np.delete(matching, [0,2,4], 1)

    return matching    

def matrix(match):
    '''
    Function used to create a confusion matrix, based
    on 2D matching numpy array betwen ground-truth and
    predictions.

    See sklearn.metrics.confusion_matrix

    Parameters
    ----------
    match : np.array
        2D numpy array, with ground-truth labels in
        column 1 and predictions labels in column 2.
    
    Returns
    -------
    2D np.array
        Confusion matrix.
    
    Author
    ------
    Alexandre Delplanque
    '''

    truth = match[:,0]
    predicted = match[:,1]

    labels = list(set(np.concatenate((truth,predicted),axis=0)))
    labels = [int(lab) for lab in labels]

    conf_matrix = confusion_matrix(truth, predicted, labels=labels)

    return conf_matrix

def report(match, coco_path):
    '''
    Function used to create a report with calculation 
    of precision, recall and F1-score for each class 
    and the weighted average of each of these metrics.

    See sklearn.metrics.classification_report

    Parameters
    ----------
    match : np.array
        2D numpy array, with ground-truth labels in
        column 1 and predictions labels in column 2.
    coco_path : str
        Path to the COCO-style annotation file in JSON 
        format.
        Warning: Labels in match need to match id in
        annotation file (cf. 'categories' dict)!

    Returns
    -------
    print
    dict

    Author
    ------
    Alexandre Delplanque
    '''

    with open(coco_path,'r') as json_file:
        coco_dic = json.load(json_file)

    truth = match[:,0]
    predicted = match[:,1]

    labels = list(set(np.concatenate((truth,predicted),axis=0)))
    labels = [int(lab) for lab in labels]

    categories = coco_dic['categories']

    i = 0
    classes = ['Background']
    for lab in labels:
        if categories[i]['id']==lab:
            classes.append(categories[i]['name'])

        i += 1

    print(classification_report(truth, predicted, labels=labels, target_names=classes, digits=2))
    
    out = classification_report(truth, predicted, 
                                labels=labels, 
                                target_names=classes, 
                                digits=2, 
                                output_dict=True)
    return out