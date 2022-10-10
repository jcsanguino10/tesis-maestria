import json
import gzip
import pymysql
import urllib
import urllib.parse
import urllib.request
import configparser
import time

from py_babelnet.calls import BabelnetAPI
from io import BytesIO


def get_data_from_db(h, u, p, db, id):
    scrap_db = pymysql.connect(host=h, user=u, password=p, db=db) 
    cursor = scrap_db.cursor()
    query = "SELECT * FROM publish where JSON_LENGTH(lib2)<=1 order by number desc"
    cursor.execute(query)
    datos=cursor.fetchall()
    cursor.close()
    scrap_db.close()
    return datos

def get_text_labeing(data):
    service_url = 'https://babelfy.io/v1/disambiguate'

    lang = 'EN'
    key  = 'fae845b8-3d3e-49bc-a5f0-505e3921ae07'

    params = {
        'text' : data,
        'lang' : lang,
        'key'  : key
    }

    url = service_url + '?' +  urllib.parse.urlencode(params)
    request = urllib.request.Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urllib.request.urlopen(request)

    if response.info().get('Content-Encoding') == 'gzip':
        buf = BytesIO( response.read())
        f = gzip.GzipFile(fileobj=buf)
        dataJson = json.loads(f.read())
        return dataJson

def actualizar(h, u, p, db, jsonContent, id):
    scrap_db = pymysql.connect(host=h, user=u, password=p, db=db) 
    cursor = scrap_db.cursor()
    query = "UPDATE publish SET lib2 = (%s) where number = " + str(id)+";"
    cursor.execute(query,  jsonContent )
    datos=cursor.fetchall()
    scrap_db.commit()
    scrap_db.close()

def convertir_1_json(todos_los_json):
    #Conversion de todos_los_json a diccionario
    i = 0 
    dic_final={}
    for lista in todos_los_json:
        for lista_interna in lista:
            dic_final[str(i)]=lista_interna
            i+=1
    return json.dumps(dic_final, indent = 4)

def get_text_labeing_py(data):
    key  = 'fae845b8-3d3e-49bc-a5f0-505e3921ae07'
    api = BabelnetAPI(key)
    senses = api.get_senses(lemma = data, searchLang = "EN")
    return senses

def manejar_Error(id, e):
    print ("error", str(e))
    #Escribir el error en archivo log a
    f= open ("/home/profesor/html/log2.txt", "a") 
    f.write("err"+  "::" +str(e)+ " : "+str(id) +"\n ")
    f.close()
    #Escribir hasta donde lleg  
    f= open ("/home/profesor/html/service2.txt", "w")
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

f= open ("/home/profesor/html/service2.txt", "r")
id_number= int (f.readline())
f.close()

data_prueba=get_data_from_db(h, u, p, db, id_number)
for data in data_prueba:
    id=data[0]
    data_html=data[3]
    listaHTML= data_html.split(".")

    #Variable que va a guardar la parte de htmll que se va a usar 
    str_Data_html=""

    #Cuenta el numero de palabras que hay en lista_Data_html.
    #Lo ideal esque no se pase de 700 para eviar el error de url muy larga
    largo_total=0

    #Variable que contiene los json de parte del html guardado en  list_data_html 
    todos_los_json=[]
    
    #Ultima frase de la lista html 
    ultimaFrase=listaHTML[len(listaHTML)-1]
    contador=1

    for frase in listaHTML:
        largo_frase= len(frase.split(" "))
        largo_total+=largo_frase
        entra=True

        #Se toman 3 casos : 
        #1. que se supere el largo de 700 palabras
        #2. que no se supere el largo de 700 palabras 
        #3. Que haya frases ""
        if (ultimaFrase== frase and contador==len(listaHTML)):
            entra =False
            str_Data_html+=frase
        if (largo_total==0 or largo_total<=700  )and entra :
            str_Data_html+=frase
        elif str_Data_html=="":
            pass
        else:
            try :
                #llamo al servicio
                jsonContent =get_text_labeing(str_Data_html)
                print(type(jsonContent), " len:", len(jsonContent))
                time.sleep(1)
                todos_los_json.append(jsonContent)
                
                #reinicio variables 
                largo_total=0
                str_Data_html=""

            except urllib.error.HTTPError or  Exception  as e :
                manejar_Error(id,e)
            except:
                manejar_Error(id, " error inesperado ")
        contador+=1

    jsonContent= convertir_1_json(todos_los_json)

    actualizar(h, u, p, db, jsonContent, id)
    print("---------------------------------- actualizado", str(id) ,"--------------------------- ")

"""import textrazor
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
            #Escribir hasta donde llegó 
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
main_function(data_prueba)"""