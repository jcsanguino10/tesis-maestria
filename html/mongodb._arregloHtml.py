from importlib.resources import path
import pymongo
from pymongo import MongoClient
import pymysql
import certifi
import os

from bson.objectid import ObjectId
from bs4 import BeautifulSoup
import configparser
import json 

#Conexion con la bd
ca = certifi.where()
cluster = MongoClient(host="mongodb+srv://read_db:Dak2ZIvwL7ZNqIt6@gcflearnfree.ivza6.azure.mongodb.net/gcfglobal?retryWrites=true&w=majority", tlsCAFile=ca)
db = cluster["gcfglobal"]
# Donde se va a apuntar 
collection2= db["tutorial"]
collection3=db["lesson"]
#Filtro a colocar
body = {"language":"en"} 


#coleccion
resultsTutorial =collection2.find(body) 

#devuelve una tupla (html en texto plano, si se cambió el html)
def cambiarHtml(html):
    entro =False
    soup = BeautifulSoup(html, "html.parser")
    html_modificado = soup
    #Si hay div class="activity"
    listaDivs= soup.find_all("div", {'class':'sidebar'})
    if len (listaDivs)>0:
        entro=True
        for div in listaDivs: 
            div.decompose()
    #en soup se guarda el html modificado

    #Si hay <pre>
    listaPres= soup.find_all("pre")
    if len (listaPres)>0:
        entro=True
        for pre in listaPres: 
            pre.decompose()

    #Si hay <code>, cambiar por ""
    listaCode= soup.find_all("code")
    if len (listaCode)>0:
        entro=True 
        for element in listaCode:
            aux=str(element.text).strip()
            element.replaceWith("'"+aux+"'")
            
    if entro:
        #Se quitan el resto de los tags
        for data in soup(['style', 'script']):
            data.decompose() # Remove tags
        # return data by retrieving the tag content
        return (' '.join(soup.stripped_strings), True)
    return ("", False)

#Realiza la lista de publishers que se van a ingresar a la bd 
    #En caso de que si se cuemplan con todas las condiciones, se retorna la info a guardar. 
    #En caso de que no se cumplan las condiciones, se retorna ""
def hacerComprobacionData(lesson, path ):
    listNotCourses=["Spanish","Basic Spanish Skills","English","Apprendre l'anglais","Englisch lernen","Arabic","Chinese","Aprenda Inglês","Aprenda Inglés","Korean","Simple English","Français","Français","Initiation all'informatique","Pусский","Интернет-безопасность","Интернет-безопасность для детей","русский","中文","中文","计算机基础","计算机基本技能","儿童因特网安全","因特网安全","平面设计入门","谷歌地图","Windows基础","谷歌使用技巧","Norsk","Grunnleggende IT-ferdigheter","Grunnleggende om Datamaskiner","Grunnleggende om Windows","Internett-sikkerhet","Bahasa Indonesia","Dasar-Dasar Komputer","Dasar-Dasar Windows","Excel 2016 (Bahasa Indonesia)","Google Maps (Bahasa Indonesia)","Keamanan Internet","Keamanan Internet Untuk Anak-anak","Kemampuan-Kemampuan Komputer Dasar","Word 2016 (Bahasa Indonesia)","Greek","Χάρτες Google","Μικρές συμβουλές Google","Λογαριασμός Google","Gmail (Greek)","Google Drive (Greek)","Φόρμες Google","Έγγραφα Google","Παρουσιάσεις Google","Παρουσιάσεις Google","Υπολογιστικά Φύλλα Google","YouTube (Greek)","Arabic","أساسيات الكمبيوتر","أكسيل 2016","Kyrgyz","Балдар үчүн интернет коопсуздугу","Интернет коопсуздугу","Google Карталар","Компьютерде иштөө негиздери","Графикалык дизайн негиздери","Google колдонуу кеңештери","Windows негиздери","Компьютер негиздери","PowerPoint 2016 Негиздери","Excel 2016 Негиздери","Word 2016 Негиздери","Dutch","Computer Basiskennis","Computer basisvaardigheden","Polski","Excel (Polski)" ]    
    
    #Revisar si es una lesson o una pagina informativa 
    if (len(lesson["tutorials"])>0 and lesson["published"]=="true"):
            course=lesson["tutorials"][0]["title"]
            if  not(course in listNotCourses ):
                #(html en texto plano, si se cambió el html) 
                tuplaHtml=cambiarHtml(lesson["publish"]["pages"]["1"])
                # Si el html sufrio cambios y si el html no es ""             
                if tuplaHtml[0]!="" and tuplaHtml[1]:
                    lesson_id= str(lesson["_id"])
                    html=tuplaHtml[0].replace("\'","'").replace('"', "'")
                    data=(  html, json.dumps({}, indent = 4) ,json.dumps({}, indent = 4), lesson_id, path)
                    return data
    return ""


def main_function(resultsTutorial):
    #lista con datos para colocar en la bd 
    list_data=[]
    for result in resultsTutorial:
        if not (result["path"].__contains__('tr_'))  and not( result["description"].__contains__('your native language')):
            for unidades in result["units"]:
                for idPublish in unidades["ids"].split(","):
                    if (idPublish!=""):
                        lesson = collection3.find_one({"_id": ObjectId(idPublish)})
                        #metodo para hacer data 
                        datoLesson =hacerComprobacionData(lesson, result["path"] )
                        if datoLesson != "":
                            list_data.append(datoLesson)
    return list_data


#lesson_id, path, html, json.dumps({}, indent = 4) ,json.dumps({}, indent = 4)
# number Int PRIMARY KEY, id_lesson VARCHAR(100), course_path VARCHAR(100), html TEXT, links json, id_links json, lib1 json, lib2 json);"
  
def actualizar(h, u, p, db, list_data):
    scrap_db = pymysql.connect(host=h, user=u, password=p, db=db) 
    cursor = scrap_db.cursor()
    con=1
    for i in list_data:
        query = "UPDATE publish SET  html= \" "+str(i[0])+ " \", lib1= '"+ str(i[1])+"' , lib2= '"+str(i[2]) + "' where id_lesson ='"+ str(i[3]) +"' and  course_path= '"+str(i[4])+"' ;" 
        print(query) 
        print(con)
        con+=1
        cursor.execute(query )
    cursor.fetchall()
    scrap_db.commit()
    scrap_db.close()

config = configparser.RawConfigParser()
config.read(filenames = 'access.properties')

h = config.get('mysql','host')
u = config.get('mysql','user')
p = config.get('mysql','password')
db = config.get('mysql','db')

list_data=main_function(resultsTutorial)
actualizar(h, u, p, db, list_data)



"""
markup = '<a>This is not div <div class="1">This is div 1</div><div class="2">This is div 2</div></a><div class="2">This is div sdfdfdf</div><pre> hola bbabab </pre><code> holabb </code>'
soup = BeautifulSoup(markup,"html.parser") 
listaDivs= soup.find_all("div", {'class':"2"})
a_tag = soup
entro=False
if len (listaDivs)>0:
    entro=True
    for div in listaDivs: 
        div.decompose()
#soup.find('div',class_='2').decompose()
print (a_tag, entro)

listaPres= soup.find_all("pre")
if len (listaPres)>0:
    entro=True
    for pre in listaPres: 
        pre.decompose()
print (a_tag, entro)
print("sadfasdfasdf sfasdfsaf ",a_tag)
#Si hay <code>, cambiar por ""
listaCode= soup.find_all("code")
if len (listaCode)>0:
    entro=True 
    for element in listaCode:
        aux=str(element.text).strip()
        element.replaceWith("'"+aux+"'")
"""