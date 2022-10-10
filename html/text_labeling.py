import textrazor
import pymysql
import configparser
import urllib.error
import json 
import time

def get_data_from_db(h, u, p, db, id):
    scrap_db = pymysql.connect(host=h, user=u, password=p, db=db) 
    cursor = scrap_db.cursor()
    query ="SELECT * FROM publish where JSON_LENGTH(lib1)<=1 order by number asc"
    cursor.execute(query)
    datos=cursor.fetchall()
    cursor.close()
    scrap_db.close()
    return datos

def get_text_labeing(data):
    textrazor.api_key = "9d297de49b535c0e8cfaed3994450f90c75acd6e5b8c292d19ea2fa5"
    client = textrazor.TextRazor(extractors=["entities", "topics"])
    response = client.analyze(data)
    return response.json

def actualizar(h, u, p, db, jsonContent, id):
    scrap_db = pymysql.connect(host=h, user=u, password=p, db=db) 
    cursor = scrap_db.cursor()
    query = "UPDATE publish SET lib1 = (%s) where number = " + str(id)+";"
    cursor.execute(query,  jsonContent )
    datos=cursor.fetchall()
    scrap_db.commit()
    scrap_db.close()

def main_function(data_prueba):
    for data in data_prueba:
        id=data[0]
        data_html=data[3]
        try :
            jsonContent = get_text_labeing(data_html)
            jsonContent=json.dumps(jsonContent, indent = 4)
            actualizar(h, u, p, db, jsonContent, id)
            print (id)
            time.sleep(0.25)
        except :
            #Escribir el error en archivo log a
            f= open ("/home/profesor/html/log1.txt", "a")
            f.write("err"+  "::"+str(id) +"\n ")
            f.close()
            #Escribir hasta donde llego
            f= open ("/home/profesor/html/service1.txt", "w")
            f.write(str(id))
            f.close()
            exit()

# Lectura de archivos
config = configparser.RawConfigParser()
config.read(filenames = 'access.properties')
h = config.get('mysql','host')
u = config.get('mysql','user')
p = config.get('mysql','password')
db = config.get('mysql','db')

f= open ("/home/profesor/html/service1.txt", "r")
id_number= int (f.readline())
f.close()

#data [id-1][indice para obtener html]
data_prueba=get_data_from_db(h, u, p, db, id_number)
main_function(data_prueba) 