import argparse
from dataclasses import dataclass
from typing import Tuple
import os
import json

import numpy as np


from .median_method import MedianMethod



@dataclass
class CustomParameters:
    neighbourhood_size: int = 100
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_dict() -> 'AlgorithmArgs':
        args: dict = read_json_file()
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]: # not necessary
    return np.genfromtxt(path,
                         skip_header=1,
                         delimiter=",",
                         usecols=[1])

def execute(config, ts):
    set_random_state(config)
    anom_timeseries_1d = ts #load_data(config.dataInput)
    mm = MedianMethod(timeseries=anom_timeseries_1d,
                      neighbourhood_size=config.customParameters.neighbourhood_size)

    scores = mm.fit_predict()
    return scores

def read_json_file():
     #read json file and append necessary attributes
    abspath = os.path.dirname(os.path.realpath(__file__))
    jsonfile= os.path.join(abspath ,"manifest.json")
    f = open(jsonfile)
    jsondict = json.load(f)
    f.close()
    jsondict['executionType']='execute'
    return jsondict


def run_mm_algorithm(ts):
    scores = []

    config = AlgorithmArgs.from_dict()

    if config.executionType == "train":
        print("Nothing to train, finished!")
        exit(0)
    elif config.executionType == "execute":
        scores = execute(config, ts)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected 'execute'!")
    return list(scores)
# Code for testing
# import pandas as pd
# abspath = os.path.abspath("D_Model")
# file = os.path.join(abspath, "test_data","sby_need_full.csv") 
# data = pd.read_csv(file,usecols= ['value'])
# data = data['value'].to_numpy()
# test = run_mm_algorithm(data)