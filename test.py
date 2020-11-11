import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
from colorama import Fore, Style
from datetime import date
import requests
from datetime import datetime

today = date.today()
model=load_model("./model2-001.model")

labels_dict={0:'Sin mascarilla',1:'Con mascarilla'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
def Escaner():
    webcam = cv2.VideoCapture(0) 
    l = 0
    contador = 0
    # Se importa el archivo de opencv que ayuda en deteccion de rostros
    classifier = cv2.CascadeClassifier('/home/cris/.local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    while True:
        
        (rval, im) = webcam.read()
        im=cv2.flip(im,1,1) #Se rota la imagen para que funcionene como un espejo

        # Se reduce la imagen para acelerar la deteccion
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detectar MultiScale / rostros
        faces = classifier.detectMultiScale(mini)

        #Se dibuja un rectangulo alrededor del rostro
        for f in faces:
            
            (x, y, w, h) = [v * size for v in f]  #Escala el respaldo del tamaño de la forma
            #Guarda sólo las caras de los rectángulos en SubRecFaces
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(150,150))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,150,150,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)
            #print(result)
            
            label=np.argmax(result,axis=1)[0]
        
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            if labels_dict[label] == "Con mascarilla":
                contador = contador + 1
                
            #print(labels_dict[label])-
            #Emitir un sonido de alerta en caso de que no lleve mascarilla CAT
            #Consumir endpoint para enviar datos de analisis.
        l = l + 1
        # Se muestra la imagen
        cv2.imshow('LIVE',   im)
        key = cv2.waitKey(10)
        
        if l == 30:
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            date = date + '.png'
            cv2.imwrite(date,im)
            key = 27
        # Si se presiona la tecla Esc se termina el loop
        if key == 27: #Tecla Esc
            break
    # Detiene el video
    webcam.release()

    # Se cierran todas las pestanas abiertas
    cv2.destroyAllWindows()
    porcentaje = (contador / 30)*100
    d1 = datetime.now().strftime('%Y-%m-%d')
    if porcentaje < 60:
        task = {"userWithMask": 0, "userDate": d1 }
        #resp = requests.post('http://ec2-3-84-233-38.compute-1.amazonaws.com:3000/api/sarscov2/postInsertAnalysis', json=task)
        wavFile = "alarm.wav"
        playsound(wavFile)
        print(Fore.RED + "No utiliza mascarilla")
        print(Style.RESET_ALL)
    else:
        task = {"userWithMask": 1, "userDate": d1 }
        resp = requests.post('http://ec2-3-84-233-38.compute-1.amazonaws.com:3000/api/sarscov2/postInsertAnalysis', json=task)

ifmenu = True

while ifmenu == True:
    print("-----------------------------------------------------------------------------------")
    print("Bienvenido al escaner de mascarillas.")
    print("Este escaner tiene diferentes opciones:")
    print("1.Escanear una persona.")
    print("2.Salir del programa.")
    opcion = input('Ingrese el numero de la opcion deseada:')
    if opcion == "1":
        Escaner()
    elif opcion == "2":
        print("Gracias por usar nuestro sistema de reconocimiento de mascarillas")
        ifmenu = False
    else:
        print("Error comando no encontrado, ingrese el numero de la opcion que desea")


