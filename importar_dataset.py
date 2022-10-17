from __future__ import print_function 
import pickle 
import os.path 
import io 
import shutil 
import requests 
from mimetypes import MimeTypes 
from googleapiclient.discovery import build 
from google_auth_oauthlib.flow import InstalledAppFlow 
from google.auth.transport.requests import Request 
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload 
  
class DriveAPI: 
    global SCOPES 
    SCOPES = ['https://www.googleapis.com/auth/drive'] 
    def __init__(self): 
        self.creds = None
        if os.path.exists('token.pickle'): 
            with open('token.pickle', 'rb') as token: 
                self.creds = pickle.load(token) 
        if not self.creds or not self.creds.valid: 

            if self.creds and self.creds.expired and self.creds.refresh_token: 
                self.creds.refresh(Request()) 
            else: 
                flow = InstalledAppFlow.from_client_secrets_file( 
                    'credentials.json', SCOPES) 
                self.creds = flow.run_local_server(port=0) 
            with open('token.pickle', 'wb') as token: 
                pickle.dump(self.creds, token) 
        self.service = build('drive', 'v3', credentials=self.creds) 

        results = self.service.files().list(q="name='data.pt'",pageSize=100, fields="files(id, name)").execute() 
        items = results.get('files', []) 
        print("Esta es la Lista de bd creados: \n") 
        print(*items, sep="\n", end="\n\n")

  
    def FileDownload(self, file_id, file_name): 
        request = self.service.files().get_media(fileId=file_id) 
        fh = io.BytesIO() 
        downloader = MediaIoBaseDownload(fh, request, chunksize=204800) 
        done = False
        try:  
            while not done: 
                status, done = downloader.next_chunk() 
            fh.seek(0) 
            with open(file_name, 'wb') as f: 
                shutil.copyfileobj(fh, f) 
            print("Archivo Descargado") 
            return True
        except: 
            print("Algo sucedió") 
            return False
  
if __name__ == "__main__": 
    obj = DriveAPI() 
    f_id = input("ingrese el Id: ") 
    f_name ="data.pt"
    obj.FileDownload(f_id, f_name) 

