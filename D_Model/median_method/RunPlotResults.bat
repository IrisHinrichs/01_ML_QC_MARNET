echo off
:: script for anaconda powershell prompt
:: This script
:: 1. read content of a json file and stores it in a string variable
:: 2. adds new attribute-value pairs to the string variable
:: 3. calls a specific anomaly detection algorithm
:: 4. plots the resulting anomaly scores in addition to the time series  
:: read contents of json file and assign it to variable jstring, without \n or \t
FOR /F "tokens=*" %%g IN ('jq -rc . manifest.json') do (SET jstring=%%g)

:: some fields are missing in json file manifest.json, need to be added
set dataInput=C:/Users/Iris/Documents/IU-Studium/Masterarbeit/01_Code/test_data/sby_need_full.csv 
set dataOutput=C:/Users/Iris/Documents/IU-Studium/Masterarbeit/01_Code/median_method_results.csv
set executionType=execute

:: add new attributes to json-string
:: leave out first and last string of jstring
set trunc=%jstring:~1,-1% 
set jstring={'dataInput':'%dataInput%','dataOutput':'%dataOutput%','executionType':'%executionType%',%trunc%}

:: double quotes need to be replaced with single quotes
:: because otherwise anaconda powershell will swallow them
set jstring="%jstring:"='%"

:: invoke specific anomaly detection algorithm and pass variable jstring as input argument
python algorithm_iris.py %jstring%
 
:: plot results
python ../plot-scores.py -d %dataInput% -s %dataOutput%
