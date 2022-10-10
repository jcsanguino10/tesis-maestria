from flask import Flask
from flask import request
import json
import pandas as pd

app = Flask(__name__)


courses = pd.read_csv('train.csv', index_col="user_id")
matrix_cf = pd.read_csv('matrix_cf.csv', index_col="user_id")
matrix_cb = pd.read_csv('matrix_cb.csv', index_col="user_id")
matrix_hb = (matrix_cf*0.9) + (matrix_cb *0.1)


@app.post("/recommend")
#@app.route("/recommend", methods=['POST'])
def recommend():
    user_info = request.json
    userid = user_info["id"]
    if userid in courses.index:
        viewed_courses = courses.loc[userid]["course_name"].unique().tolist()
        cf_recommendation = matrix_cf.loc[userid].dropna().sort_values(ascending=False).head(5).keys().tolist()
        cb_recommendation = matrix_cb.loc[userid].dropna().drop(labels=viewed_courses).sort_values(ascending=False).head(5).keys().tolist()
        hb_recommendation = matrix_hb.loc[userid].dropna().sort_values(ascending=False).head(5).keys().tolist()
        return json.dumps({"viewed": viewed_courses, "cf_rec": cf_recommendation, "cb_rec": cb_recommendation, "hb_rec": hb_recommendation})
    else:
        return json.dumps({"error": "invalid user"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4300)