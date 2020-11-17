#!/usr/bin/env python3
import boto3
from botocore.exceptions import NoCredentialsError
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

#claves de acceso para el S3 Bucket
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")
SESSION_KEY = os.environ.get("SESSION_KEY")

def upload_to_aws(local_file, bucket, s3_file):
    #se crea la conexion con el S3
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY,aws_session_token=SESSION_KEY)

    try:
        #se trata de cargar la imagen
        s3.upload_file(local_file, bucket, s3_file)
        print("Se ha subido correctamente")
        return True
    except FileNotFoundError:
        #si el archivo no existe
        print("El archivo no existe")
        return False
    except NoCredentialsError:
        #credenciales invalidas
        print("Credenciales no validas")
        return False




