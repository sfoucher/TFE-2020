# How to use ["Locating objects without bounding boxes"](https://github.com/javiribera/locating-objects-without-bboxes)

Documentation de l'auteur : [Installation et utilisation](https://github.com/javiribera/locating-objects-without-bboxes#installation)

## Anaconda
### Téléchargement du repository
Pour une utilisation optimale de l'outil, il est nécessaire d'utiliser deux *Anaconda prompt*.
Ouvrir tout d'abord un premier, et cloner ensuite le repository original :
```
git clone https://github.com/javiribera/locating-objects-without-bboxes.git
```

### Création de l'environnement
Créer ensuite l'environnement fourni dans le repository original :
```
conda env create -f C:\locating-objects-without-bboxes\environment.yml
```
Activer l'environnement :
```
conda activate object-locator
```
Installer les packages :
```
pip install <nom du disque>\locating-objects-without-bboxes\.
```


### Installation des packages manquants
Des packages seront probablement manquants. Il est donc nécessaire de les installer.
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install peterpy
pip install ballpark
pip install visdom
```

### Préparation du fichier *train.py*
Si le sytème d'exploitation utilisé est Windows, une erreur va survenir ([explications](https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection)). Pour la corriger, modifier le code *train.py* à l'emplacement *<chemin d'accès>\envs\object-locator\Lib\site-packages\object-locator\train.py*.

À la **ligne 159**, ajouter :
```
if __name__ == '__main__':
```
et indenter ensuite le code qui suit.

### Initialisation du serveur *visdom*
Afin de visualiser l'évolution des *losses* et les *heatmaps* avec *visdom*, il est nécessaire de lancer au préalable un second *Anaconda prompt*, dans lequel l'environnement devra être activé :
```
conda activate object-locator
```
Ensuite, lancer le serveur *visdom* :
```
python -m visdom.server
```

Enfin, ouvrir dans un navigateur le lien indiqué par le prompt (exemple: *http://localhost:8097*).

### Lancement d'un entrainement
Revenir au premier *Anaconda prompt*, sans fermer le second et sans fermer le navigateur!

Pour obtenir de l'aide sur les paramètres à entrer pour lancer un entrainement :
```
python -m object-locator.train -h
```
Attention de bien spécifier les chemin d'accès et les renseignements du serveur *visdom*.

### Prédictions sur un jeu test
