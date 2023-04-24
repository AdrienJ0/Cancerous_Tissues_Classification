#!/usr/bin/env python
# coding: utf-8

# In[68]:


import csv
import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import skimage
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,f1_score
from joblib import dump,load
import joblib
import sklearn.svm 
import random


# In[69]:


data_path = "C:/Users/Michel Sauvage/Projet_18/data_saved"
base_path = "C:/Users/Michel Sauvage/Projet_18/data_saved_split"
CSVlabels = 'labels.csv'

model = "model2"

#nom du dossier = model2_balanced_100000, a déjà été modifié dans le code pour éviter erreur 


# # Split patients

# In[70]:


#based on folders
def split_patient_local(data_path, model):

    # Initialize sum variable
    sum_actu = 0

    #total patches
    if model == "model1" : 
        total_patches = 2276967 #normal + cancerous
    else :
        total_patches = 1722776 #cancerous

    # Initalize lists
    patient_train = []
    patient_test = []
    patient_valid = []

    for patient in os.listdir(data_path):
        patient_path = os.path.join(data_path, patient)
        for slide in os.listdir(patient_path):
            if model == "model1" :
                x1 = len(os.listdir(patient_path + '/' + slide + '/cancerous_patches'))
                x2 = len(os.listdir(patient_path + '/' + slide + '/normal_patches'))   
                sum_slide = x1 + x2
            else :
                sum_slide = len(os.listdir(patient_path + '/' + slide + '/cancerous_patches'))

            # Add to total 
            sum_actu += sum_slide
            
            # Separate patients into train, test, and valid lists based on sum_actu
            #knowing total number patches = 2276967
            if sum_actu < total_patches * 0.7:
                if patient not in patient_train:
                    patient_train.append(patient)

            elif sum_actu < total_patches * 0.85:
                if patient not in patient_test:
                    patient_test.append(patient)

            else:
                if patient not in patient_valid:
                    patient_valid.append(patient)    

    return patient_train, patient_test, patient_valid


# In[71]:



patient_train, patient_test, patient_valid = split_patient_local(data_path, model)


# # Create train, test and validation folders and fill them with symbolic link to patients folders

# In[ ]:


def symbolink_folders(data_path, base_path):

    base_path = os.path.join(base_path, "model2_balanced_100000")
    os.makedirs(base_path, exist_ok=True)


    train_path = os.path.join(base_path, 'train')
    os.makedirs(train_path, exist_ok=True)

    test_path = os.path.join(base_path, 'test')
    os.makedirs(test_path, exist_ok=True)

    validation_path = os.path.join(base_path, 'validation')
    os.makedirs(validation_path, exist_ok=True)

    #fill folders with symbolic link to patients
    for patient_folder in patient_train:
        os.symlink(os.path.join(data_path, patient_folder), os.path.join(train_path, patient_folder))

    for patient_folder in patient_test: 
        os.symlink(os.path.join(data_path, patient_folder), os.path.join(test_path, patient_folder))

    for patient_folder in patient_valid:#patient_train pour tester hors serveur
        os.symlink(os.path.join(data_path, patient_folder), os.path.join(validation_path, patient_folder))


symbolink_folders(data_path, base_path)


# # patient + patch_path + label

# In[72]:


def CreateTuple(base_path, folder):
    
    # Ouvrir le fichier CSV et créer un lecteur CSV
    with open(CSVlabels, 'r') as file:
        reader = csv.reader(file)

        # Itérer sur chaque ligne du fichier CSV et créer un tuple pour chaque patient avec son nom et son étiquette
        csv_tuple_list = [(row[1], row[2]) for row in reader]
        valid_labels = ['ABC', 'GCB']
        patients_type_tuple = [(patient, label) for (patient, label) in csv_tuple_list if label in valid_labels]
        
    patch_type_tuple = []

    for patient in os.listdir(os.path.join(base_path, "model2_balanced_100000", folder)):
        patient_path = os.path.join(base_path, "model2_balanced_100000", folder, patient)
        for slide in os.listdir(patient_path):
            slide_path = patient_path + '/' + slide + '/cancerous_patches'
            for patch in os.listdir(slide_path):
                patch_path = slide_path + '/' + patch
                subtype = [item[1] for item in patients_type_tuple if item[0] == patient][0]
                patch_type_tuple.append((patient, patch_path, subtype))
                
                
                
    return patch_type_tuple


# In[73]:


train_tuple =  CreateTuple(base_path, "train") + CreateTuple(base_path,"test")

valid_tuple = CreateTuple(base_path, "validation")


# In[74]:


#fonction qui réduit le nombre de patch ABC pour équilibrer le training dataset
def BetterTuple(patch_type_tuple) :
    patch_type_tuple_better =[]
    count = 0
    
    for elem in patch_type_tuple:
        
        if elem[2] == 'ABC':
            count += 1
            
        if count == 3:
            count = 0
        else: 
            patch_type_tuple_better.append(elem)
    
    return patch_type_tuple_better


# In[75]:


train_tuple = BetterTuple(BetterTuple(train_tuple))


# # LPB

# In[76]:


def lpb(patch_type_tuple):

    # Créer des listes vides pour stocker les caractéristiques et les étiquettes
    features = []
    labels = []

    # Charger les images depuis les chemins spécifiés dans la liste patch_type_tuple
    for elem in tqdm(patch_type_tuple):
        path = elem[1]
        img_bgr = cv2.imread(path, 1)

        try:
            height, width, _ = img_bgr.shape
        except AttributeError:
            print(f"Error occurred while processing {path}. Skipping to the next iteration.")
            continue
            
        img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        skimage.feature.local_binary_pattern(img, 8, 1, method='default')
        
        # Taille des patches
        patch_size = 32

        # Boucle pour extraire les patches et calculer les histogrammes
        histograms = []
        for y in range(0, img.shape[0], patch_size):
            for x in range(0, img.shape[1], patch_size):
                # Extraction du patch
                patch = img[y:y+patch_size, x:x+patch_size]


                # Création d'un masque pour le patch
                mask = np.zeros(patch.shape[:2], np.uint8)
                mask[:] = 255


                # Calcul de l'histogramme pour le patch
                hist = cv2.calcHist([patch], [0], mask, [256], [0, 256])


                # Normalisation de l'histogramme
                cv2.normalize(hist, hist)


                # Ajout de l'histogramme normalisé à la liste
                histograms.extend(hist)


        # Concaténation des histogrammes normalisés dans un vecteur
        histogram_vector = np.concatenate(histograms)
        
        # Aplatir chaque image en une ligne de caractéristiques
        features.append(histogram_vector)
        labels.append(elem[2])
     
    return features, labels
        


# # SGD
# 

# In[ ]:





# In[88]:


# nb de patch par batch sachant qu'on a 85% environ des patchs cancéreux dans le train
#1445000 de patchs a 85%
#Initialiser le classifieur SVC

features_valid, label_valid = lpb(valid_tuple)
print("ici1")
countSVC =0
# Entraîner le classifieur SGD sur chaque lot d'entraînement
while True:
   
   train_tuple_shuffled = tuple(random.sample(train_tuple,k=100000))
   svc = sklearn.svm.SVC(kernel='rbf')
   print("ici2")
   countABC =0
   countGCB = 0
   for i in train_tuple_shuffled:
       if i[2] == "ABC":
           countABC +=1
       else :
           countGCB +=1
   # train model
   features_train, label_train = lpb(train_tuple_shuffled)
   
   svc.fit(features_train, label_train)
   print("ici3")
   # predict by batch for validation set (15% of total cancerous folder)    
   listPred = []
   num_validbatches = 10 
   valid_batch_size = 25000 # définir la taille du lot (batch size)
   for i in tqdm(range(num_validbatches)):
       batch_labels = svc.predict(features_valid[i * valid_batch_size:(i + 1) * valid_batch_size])
       listPred.extend(batch_labels)

   # dernier batch pour bien prendre toutes les données restantes
   batch_labels = svc.predict(features_valid[num_validbatches * valid_batch_size:])
   listPred.extend(batch_labels)
   target_names = ['ABC', 'GCB']
   
   
   # metrics
   f1 = f1_score(label_valid, listPred, average=None)
   print("Validation F1-score:", f1)
   tabmetric = classification_report(label_valid, listPred, target_names=target_names)
   print(tabmetric)
   # Convertir le résultat en liste de listes
   lines = tabmetric.split('\n')
   data = []
   for line in lines[2:]:
       row = line.split()
       data.append(row)
  
   #save training model
   os.makedirs('JoblibData_SGD', exist_ok=True)
   # Enregistrer les données dans un fichier CSV
   with open('JoblibData_SGD/classification_report_balancedSVC'+str(countSVC)+'.csv', 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(['', 'precision', 'recall', 'f1-score', 'support'])
       writer.writerows(data)
       writer.writerows(['ABC'])
       writer.writerows(['', '{}'.format(countABC)])
       writer.writerows(['GCB'])
       writer.writerows(['', '{}'.format(countGCB)])
   
   file = 'JoblibData_SGD/balanced_SVC_trained_batch'+str(countSVC)+'.joblib'
   joblib.dump(svc,file)
   countSVC +=1
   del features_train, label_train


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




