from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from tokenize import String
from typing import Optional
import pandas as pd

app = FastAPI()


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}
 
courses = pd.read_csv('train.csv', index_col="user_id")
matrix_cf = pd.read_csv('matrix_cf.csv', index_col="user_id")
matrix_cb = pd.read_csv('matrix_cb.csv', index_col="user_id")
matrix_hb = (matrix_cf*0.9) + (matrix_cb *0.1)


@app.post("/api/recommend") 
def recommend(userid:str):
    if userid in courses.index:
        viewed_courses = courses.loc[userid]["course_name"].unique().tolist()
        cf_recommendation = matrix_cf.loc[userid].dropna().sort_values(ascending=False).head(5).keys().tolist()
        cb_recommendation = matrix_cb.loc[userid].dropna().drop(labels=viewed_courses).sort_values(ascending=False).head(5).keys().tolist()
        hb_recommendation = matrix_hb.loc[userid].dropna().sort_values(ascending=False).head(5).keys().tolist()
        cf_recommendation = cf_recommendation[1:]
        cb_recommendation = cb_recommendation[1:]
        hb_recommendation = hb_recommendation[1:]
        return {"viewed": viewed_courses, "cf_rec": cf_recommendation, "cb_rec": cb_recommendation, "hb_rec": hb_recommendation}
    else:
        raise HTTPException(status_code=404, detail="User not found")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4300)