from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import csv
import time
import numpy as np

def initialize_analyticsreporting():
  """Initializes an Analytics Reporting API V4 service object.

  Returns:
    An authorized Analytics Reporting API V4 service object.
  """
  file= open("/home/profesor/data/prueba/funciona.txt", "a")
  file.write("funciona /n")
  print ("ta corriendo")
  file.close()

  credentials = ServiceAccountCredentials.from_json_keyfile_name(
      KEY_FILE_LOCATION, SCOPES)

  # Build the service object.
  analytics = build('analyticsreporting', 'v4', credentials=credentials)
  return analytics

def get_report(user_id,startDate, endDate, pageToken):
    #print(" get report ",user_id,startDate, endDate, pageToken)
    """Queries the Analytics Reporting API V4.
    Args:
      analytics: An authorized Analytics Reporting API V4 service object.
    Returns:
      The Analytics Reporting API V4 response.
    """
    body_request = {
            "viewId": "33488641",
            "user": {
                "type": "CLIENT_ID",
                "userId": user_id
            },
            "activityTypes": [
                "PAGEVIEW"
            ],
            "dateRange": {
                "startDate": startDate, #"2020-10-28",
                "endDate": endDate #"2021-12-31"
            }
        }
    if pageToken != None:
        body_request["pageToken"] = pageToken
    return analytics.userActivity().search(
        body=body_request
    ).execute()

def generate_data_to_csv_file(user_info, user_id, doc_writer):
    for session in user_info["sessions"]:
        id_session = session['sessionId']
        for activity in session["activities"]:
            try:
                doc_writer.writerow([user_id,id_session, activity["pageview"]["pagePath"],activity["pageview"]["pagePath"].split("/")[2],activity["pageview"]["pagePath"].split("/")[3], activity["activityTime"]])
            except:
                print("err in csv", user_id)
                f = open("/home/profesor/data/log.txt", "a")
                f.write("err", user_id)
                f.close()

def extrac_procces_data(starDate, endDate, file_name):
    file = open("/home/profesor/data/persistence.txt", "r")
    data= file.readline().split("-")
    file.close()
    print ("los datos son:", data)
    numFile= int(data[1])+1

    with open('/home/profesor/data/users.npy', 'rb') as f:
        users_id = np.load(f)
    with open(file_name, mode='w', newline="") as data_user_file:
        doc_writer = csv.writer(data_user_file, delimiter=',', quotechar='"')
        doc_writer.writerow(["user_id","id_session","url_course","course_name","lesson_name","timestamp"])

        last_visited = int(data[0]) # guardar en algun archivo
        i = 0
        file = open("/home/profesor/data/persistence.txt", "w")
        print ("el largo de users es : ", str(len(users_id)))
        if last_visited != 0:
            i = last_visited
        while i < len(users_id):
            #print(f'progreso : {str(i/len(users_id))}')
            user_id = users_id[i]
            print( "  valores i  ", i)
            try:
                time.sleep(2)
                user_info = get_report(user_id,starDate,endDate,None)
                total_rows = int(user_info["totalRows"])
                while (total_rows > 0):
                    generate_data_to_csv_file(user_info, user_id, doc_writer)
                    pageToken = None
                    if ("pageToken" in user_info):
                        pageToken = user_info["pageToken"]
                        time.sleep(1)
                        user_info = get_report(user_id,starDate,endDate, pageToken)
                    total_rows -= 1000
                i += 1
                file.write(str(i)+"-"+str(numFile))
            except Exception as e:
                file.close()
                f = open("/home/profesor/data/log.txt", "a")
                f.write("err"+ str(e)+ "::"+str(user_id) +"\n ")
                print("exception "+ str (e) +" en "+ str(i))
                if "429" in str(e):
                    f.write("Limite alcanzado, ultimo usuario : ", str(i), " : " ,user_id +"\n ")
                    f.close()

                    file = open("/home/profesor/data/persistence.txt", "w")
                    file.write(str(i)+"-"+str(numFile))
                    file.close()

                    print("Limite alcanzado, ultimo usuario : ", str(i), " : " ,user_id)
                    exit()
                f.close()
            except:
                file.close()
                f = open("/home/profesor/data/log.txt", "a")
                f.write("err"+ str(e)+ "::"+str(user_id) +"\n ")
                print("exception "+ str (e) +" en "+ str(i))
                if "429" in str(e):
                    f.write("Limite alcanzado, ultimo usuario : ", str(i), " : " ,user_id +"\n ")
                    f.close()

                    file = open("/home/profesor/data/persistence.txt", "w")
                    file.write(str(i)+"-"+str(numFile))
                    file.close()

                    print("Limite alcanzado, ultimo usuario : ", str(i), " : " ,user_id)
                    exit()
                f.close()

def extrac_procces_data2(starDate, endDate, file_name):
    numFile      = 0
    last_visited = 0
    with open("persistence.txt") as r:
        data= r.readline().split("-")
        numFile      = int(data[1])+1
        last_visited = int(data[0])

    with open('/home/profesor/data/users.npy', 'rb') as f, open(file_name, mode='w', newline="") as data_user_file:
        users_id = np.load(f)

        doc_writer = csv.writer(data_user_file, delimiter=',', quotechar='"')
        doc_writer.writerow(["user_id","id_session","url_course","course_name","lesson_name","timestamp"])

        i = last_visited
        while i < len(users_id) and i<last_visited+11000:
            print(f'progreso : {str(i/len(users_id))}')
            user_id = users_id[i]
            print("Avance actual:", i-last_visited)
            try:
                time.sleep(0.25)
                user_info = get_report(user_id,starDate,endDate,None)
                total_rows = int(user_info["totalRows"])
                while (total_rows > 0):
                    generate_data_to_csv_file(user_info, user_id, doc_writer)
                    pageToken = None
                    if ("pageToken" in user_info):
                        pageToken = user_info["pageToken"]
                        time.sleep(0.25)
                        user_info = get_report(user_id,starDate,endDate, pageToken)
                    total_rows -= 1000
                i += 1
            except Exception as e:
                with open("log.txt", "a") as f1:
                    f1.write("err"+ str(e)+ "::"+str(user_id) +"\n ")
                    if "429" in str(e):
                        f1.write("Limite alcanzado, ultimo usuario : "+ str(i)+ " : " +str(user_id) +"\n ")
                break
        with open("persistence.txt","w") as r:
            r.write(str(i)+"-"+str(numFile))
                

def extract_ids():
    users_id = []
    for i in range(0,70):
        name_file = f'Analytics English - People Served Explorador de usuarios 20210701-20220129 ({str(i)}).csv'
        with open(name_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count != 0:
                        users_id.append(row[0].split(",")[0])
                    line_count += 1
    len(users_id)
    with open('/home/profesor/data/users.npy', 'wb') as f:
        np.save(f, np.array(users_id))

SCOPES = ['https://www.googleapis.com/auth/analytics']
KEY_FILE_LOCATION = '/home/profesor/data/mindful-oath-331115-6d3d2816c169.json'
VIEW_ID = '33488641'

analytics = initialize_analyticsreporting()

file = open("/home/profesor/data/persistence.txt", "r")
data= file.readline().split("-")
file.close()
nameFile= "users_"+data[1]+"_"+data[0]

#extrac_procces_data("2021-01-29", "2022-01-29", nameFile+".csv") # se llame diferente
extrac_procces_data2("2021-01-29", "2022-01-29", nameFile+".csv") # se llame diferente
print (" nombre archivo ", nameFile+".csv")
print ("Termino")

file= open("/home/profesor/data/prueba/funciona.txt", "a")
file.write("termino /n")
file.close()
