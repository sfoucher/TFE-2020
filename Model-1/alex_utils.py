import numpy as np 
from bounding_box import bounding_box as bb
import cv2
import os
from PIL import Image
import torch
from IPython.display import Image as colIm

# Besoin d'une fonction collate_fn() pour créer les batchs
# car les images ne contiennent pas le même nombre d'objets
def collate_fn(batch):
    return tuple(zip(*batch))

# Fonction effectuant un NMS
def NMScustom(boxes, labels, scores, tresh):
    '''
    Fonction permettant d'effectuer le post-processing "Non-Maximum Supression".
    Permet d'obtenir une seule bounding box par objet détecté.
    '''
    boxes = boxes.to('cpu')
    boxes = boxes.detach().numpy()

    labels = labels.to('cpu')
    labels = labels.detach().numpy()

    scores = scores.to('cpu')
    scores = scores.detach().numpy()

    # Si pas de bboxes, renvoie une list vide
    if len(boxes)==0:
        return []

    pick = []

    # Coordonnées des bbox
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Aires des bbox 
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Classement selon les scores
    idxs = scores

    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Recherche des coordonnées (x1,y1) les plus grandes
        # et des coordonnées (x2,y2) les plus petites = intersection
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Calcul de la hauteur et largeur des bbox
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Ratio d'overlapping entre la bbox calculée et celle dans la variable area
        overlap = (w * h) / area[idxs[:last]]

        # Elimination des index de la liste dont l'overlap est supérieur au seuil
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > tresh)[0])))

    return {'boxes':boxes[pick], 'labels':labels[pick], 'scores':scores[pick]}

# Fonction de seuillage sur les scores
def CutOffScores(boxes, labels, scores, tresh):
    '''
    Fonction permettant d'effectuer un seuillage à partir des 
    valeurs des scores.
    '''
    # Si pas de bboxes, renvoie une list vide
    if len(boxes)==0:
        return []

    # Détermination de la liste d'index à garder
    index = []
    for i in range(len(boxes)):

        if scores[i] > tresh:
            index.append(i)

    return {'boxes':boxes[index], 'labels':labels[index], 'scores':scores[index]}

# Obtention de l'ID d'une image selon son chemin d'accès
def getIDfromPath(dic, path):
    '''
    Obtention de l'ID d'une image à partir de son chemin d'accès
    et du dictionnaire contenant ses annotations.
    '''
    r = None
    for row in range(len(dic)):
        if dic[row]['path']==path:
            r = int(row)
        
    if r is None:
        print('PATH NOT FOUND: Chemin d\'accès non-répertorié dans ce dictionnaire.')
    
    return r

# Fonction d'affichage de l'image
def ShowSaveImage(title, image, output_path):
    #cv2.imwrite(output_path, image)
    # Resize pour l'affichage
    image = cv2.resize(image, (1500, 1000))
    # cv2.imshow(title, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    colIm(image)

# Fonction d'affichage de l'image et des bounding box
def showImageBoxes(output, input_path, output_path, pred = True):
    '''
    Fonction permettant l'affichage d'une image et ses annotations.
    '''
    # Librairie PIL évite les rotations d'images inattendues
    image = Image.open(input_path, mode='r')

    # Trans. en array Numpy pour pouvoir être affichée
    image = np.array(image)

    for i in range(len(output[0]['boxes'])):
        
        # Détermination de la couleur de la bbox selon la classe
        if int(output[0]['labels'][i]) == 0:
            color = "gray"

        elif int(output[0]['labels'][i]) == 1:

            color = "aqua"

            if 'scores' in output[0]:
                label = "Buffalo | {}".format("%.2f" % output[0]['scores'][i])
            else:
                label = "Buffalo"

        elif int(output[0]['labels'][i]) == 2:

            color = "olive"

            if 'scores' in output[0]:
                label = "Topi | {}".format("%.2f" % output[0]['scores'][i])
            else:
                label = "Topi"

        elif int(output[0]['labels'][i]) == 3:

            color = "purple"

            if 'scores' in output[0]:
                label = "Ugandese Kob | {}".format("%.2f" % output[0]['scores'][i])
            else:
                label = "Ugandese Kob"

        elif int(output[0]['labels'][i]) == 4:

            color = "lime"

            if 'scores' in output[0]:
                label = "Warthog | {}".format("%.2f" % output[0]['scores'][i])
            else:
                label = "Warthog"

        elif int(output[0]['labels'][i]) == 5:

            color = "red"

            if 'scores' in output[0]:
                label = "Waterbuck | {}".format("%.2f" % output[0]['scores'][i])
            else:
                label = "Waterbuck"
        
        # Ajout des bbox à l'image        
        bb.add(image,int(output[0]['boxes'][i][0]),
                        int(output[0]['boxes'][i][1]),
                        int(output[0]['boxes'][i][2]),
                        int(output[0]['boxes'][i][3]),
                        label = label,
                        color = color)

    if pred is not True:
        pred_txt = "Ground truth (Image ID : {})".format(str(int(output[0]['image_id'])))
        save_name = os.path.join(output_path,"GT_{}".format(str(int(output[0]['image_id']))))
    else:
        pred_txt = 'Predictions'
        save_name = os.path.join(output_path,"Pred")
    
    # Affichage et enregistrement de l'image
    opencv_im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ShowSaveImage(pred_txt, opencv_im, save_name+'.JPG')
 
# Fonction de correspondance entre les bbox de la gt et de la pred
def PredGTMatching(ground_truth, predictions, IoU_tresh):
    '''
    Fonction permettant d'effectuer la correspondance entre les bounding
    box de la ground-truth et celles prédites.

    Les sorties permettent de construire une matrice de confusion.
    '''

    gt_boxes = ground_truth[0]['boxes']
    p_boxes = predictions[0]['boxes']

    # Calcul des IoU et rangement des résultats dans un tableau
    p_index = []
    res=[]
    i=0
    for gt_box in gt_boxes:
        
        # Index de la gt box
        id_gt = i

        # Label de gt box
        gt_label = int(ground_truth[0]['labels'][i])

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
            p_label = int(predictions[0]['labels'][index])

        res.append([id_gt, gt_label, index, p_label, p_iou_max])

        i += 1

    # Index de p_boxes sans correspondance avec la gt (FP) = p_not_used
    if len(p_boxes) > len(gt_boxes):
        p_index_unique = np.unique(p_index)
        p_boxes_index = list(range(len(p_boxes)))

        # On retire les index rencontrés, il reste ceux sans correspondance
        p_not_used = [item for item in p_boxes_index if item not in p_index_unique]

        # Ajout à la variable res
        for k in range(len(p_not_used)):
            new_index = p_not_used[k]
            label = int(predictions[0]['labels'][new_index])
            res.append([i, int(0), new_index, label, int(0)])
            i += 1

    matching = np.array(res)
    matching = np.delete(matching, [0,2,4], 1)

    return matching

# Prédiction d'une image spécifique du dataset à partir de son path
def OneSpecificImagePred(path, dic, model, dataset):
    '''
    Fonction permettant d'obtenir les prédictions d'une image 
    spécifique, renseignée par son chemin d'accès (path).

    Retourne un tuple (ground-truth , predictions).
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx = [getIDfromPath(dic, path)]
    sampler = torch.utils.data.SubsetRandomSampler(idx)

    torch.cuda.empty_cache()

    model.eval()

    one_dataloader = torch.utils.data.DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=collate_fn,
                                sampler=sampler)


    image, target = next(iter(one_dataloader))
    image = list(image.to(device) for image in image)
    target = [{k: v.to(device) for k, v in t.items()} for t in target]

    out = model(image)
    ground_truth = target

    return ground_truth, out