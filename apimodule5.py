# model file for you to edit
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, confloat

from pathlib import Path

import logging
import click
import pickle

app=FastAPI()

logging.basicConfig(filename="apifilename.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%d/%m/%Y %I:%M:%S %p",
                    filemode='w',
                    force = True)

def read_logfile(log_filepath: str):
    with open(log_filepath) as f:
        for line in f:
            print(line)

# Created a startup
@app.on_event("startup")
def loadmodel():
    #calling pickle 
    global mdl
    with open("arun_model.pkl", "rb") as my_stream:
        mdl=pickle.load(my_stream)

class toyotacar(BaseModel):
    age: int
    km: int

@app.post("/arun")
def make_prediction(user_car: toyotacar):
    user_car_dict = user_car.dict()
    user_age=user_car_dict["age"]
    logging.info(user_age)
    user_km=user_car_dict["km"]
    logging.info(user_km)
    predicted_cost= mdl.predict([[user_age, user_km]])
    logging.info("cost predicted from model")
    logging.info(predicted_cost)
    predicted_cost=predicted_cost[0]
    logging.info("cost predicted and ready to display, first value from array")
    logging.info(predicted_cost)
    print(predicted_cost)
    read_logfile("apifilename.log")
    return {"Predicted Cost": predicted_cost}

