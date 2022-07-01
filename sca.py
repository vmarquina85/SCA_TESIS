from ast import If
from turtle import shape
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import cv2
import os
import time

Modotest = False
#crc o cmc
#rank 1 o rank 2
# Inicializar MTCNN y el modelo InceptionResnetV1 
# verificamos que tipo de procesamiento se usará CPU O GPU para este estudio se usara tomando en cuenta la GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
mtcnn0 = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=20,device=device)
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, min_face_size=20, device=device) 
resnet = InceptionResnetV1(pretrained='vggface2').eval() # preentrenado con vggface 2

# leemos los datos del folder de fotos (para crear dataset)
dataset = datasets.ImageFolder('fotos') # path del folder fotos para el dataset
 # Accedemos a los nombres tomando en cuenta los nombres en las carpetas
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}


def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

#lista de nombres correspondientes a las fotos Cortadas (cropped)
name_list = [] 
# lista de matriz de Incrustación despues de la conversion de rostros cortados a matrix de incrustación usando resnet
embedding_list = [] 
for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True) 
    if face is not None and prob>0.92:
        emb = resnet(face.unsqueeze(0)) 
        embedding_list.append(emb.detach()) 
        name_list.append(idx_to_class[idx])  

# grabamos la bd
data = [embedding_list, name_list] 
torch.save(data, 'data.pt') # grabamos la data en data.pt

# cargamos data.pt
load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 
# Usando la webcam para el reconocimiento
cam = cv2.VideoCapture(1) 
#usamos la camara por default de la laptop pero tambien podriamos usar 
#otro dispositivo (como webcam) modificando el indice a cv2.VideoCapture(0) cambiándolo a 1
identified_names =[]
identified_dists=[]
identified_frames=[]
while True:
    ret, frame = cam.read(0)
    if not ret:
        print("Error al tomar fotograma")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    if img_cropped_list is not None:
        # boxes, _ = mtcnn.detect(img, landmarks=True)
        boxes, probs, landmarks  = mtcnn.detect(img, landmarks=True)       
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
                dist_list = [] # 
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # obtener la minima distancia de la lista
                min_dist_idx = dist_list.index(min_dist) # obtener indice
                name = name_list[min_dist_idx] # obtner nombre
                
                box = boxes[i] 
               

                
                original_frame = frame.copy() # guardamos una copia del frame antes de modificarla
                
                if min_dist<0.90:
                    frame = cv2.putText(frame, name+' '+str(min_dist), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
                    # si flag activado
                    if Modotest ==True :
                        frame = cv2.putText(frame,'TEST MODE',(20,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                        identified_names.append(name)
                        identified_dists.append(str(min_dist))
                        identified_frames.append(frame)
                    elif Modotest ==False: 
                        frame = cv2.putText(frame,'NORMAL MODE',(20,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                    # grabar el frame
                else:
                    frame = cv2.putText(frame, 'Desconocido', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (0,0,255), 2)
                    if Modotest ==True :# si flag activado
                        frame = cv2.putText(frame,'TEST MODE',(20,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                        identified_names.append('Desconocido')
                        identified_dists.append(str(min_dist))
                        identified_frames.append(frame)
                    elif Modotest ==False: 
                        frame = cv2.putText(frame,'NORMAL MODE',(20,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)

    cv2.imshow("IMG", frame)

   
    
    k = cv2.waitKey(1)
    if k%256==27: # Salir de Programa con Esc
        print('Se preciona ESC, Cerrando...')
        break
        
    elif k%256==32: # usar barra para registrar imagen de webcam
        print('Ingrese su Nombre :')
        name = input()
        
        # crear el directorio si este no existe
        if not os.path.exists('photos/'+name):os.mkdir('photos/'+name)
            
        img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, frame)
        print(" saved: {}".format(img_name))

    elif k%256==13: # usar el boton enter activamos el modo de test
        Modotest = not Modotest
        identified_names =[]
        identified_dists=[]
        identified_frames=[]
        # if Modotest == True :
        # inicia el contador

        
        # una ves terminado
        # while Modotest is True:

    elif k%256==115:
        if Modotest == True:
            Modotest = not Modotest
        identifiedData= pd.DataFrame({"Identified_Name":identified_names,"Identified_dist":identified_dists})
        print('Ingrese su Nombre :')
        name = input()
        # se graba el archivo csv con los datos
        if not os.path.exists('results/'+name):os.mkdir('results/'+name)
        identifiedData.to_csv('results/'+name+'/'+name+'_identified.csv')
        # # se graban los frames en otra carpeta
        if not os.path.exists('results/'+name+'/'+name+'_frames'):os.mkdir('results/'+name+'/'+name+'_frames')
        for i,imgFrame in enumerate(identified_frames):
            cv2.imwrite(os.path.join('results/'+name+'/'+name+'_frames', str(i)+'.jpg'),imgFrame)
     
cam.release()
cv2.destroyAllWindows()
    