import argparse
from dataclasses import dataclass
from typing import Tuple
import sys
import json

import numpy as np

from median_method import MedianMethod

# read json file and append necessary attributes
jsonfile = r"C:\Users\Iris\Documents\IU-Studium\Masterarbeit\01_ML_QC_MARNET\D_Model\median_method\manifest.json"
f = open(jsonfile)
jsondict = json.load(f)
f.close()
jsondict['executionType']='execute'



@dataclass
class CustomParameters:
    neighbourhood_size: int = 100
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    # @staticmethod
    # def from_sys_args() -> 'AlgorithmArgs':
    #     #args: dict = json.loads(sys.argv[1])
    #     #args: dict = json.loads(sys.argv[1].replace("'", '"'))
    #     args: dict = json.loads(sys.argv[1].replace("'", '"'))
    #     custom_parameter_keys = dir(CustomParameters())
    #     filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
    #     args["customParameters"] = CustomParameters(**filtered_parameters)
    #     return AlgorithmArgs(**args)
    @staticmethod
    def from_dict() -> 'AlgorithmArgs':
        args: dict = jsondict
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
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
    return scores

# from utilities import get_filestring, read_station_data
# from common_variables import datapath
# filestring = get_filestring()
# data = read_station_data()
import pandas as pd
file = r"C:\Users\Iris\Documents\IU-Studium\Masterarbeit\01_ML_QC_MARNET\D_Model\test_data\sby_need_full.csv" 
data = pd.read_csv(file,usecols= ['value'])
data = np.array(data)
test = run_mm_algorithm(data)