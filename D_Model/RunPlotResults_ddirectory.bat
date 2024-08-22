::echo off
::===========================================================================
:: script for anaconda powershell prompt
:: This script
:: 1. reads content of a json file and stores it in a string variable
:: 2. adds new attribute-value pairs to the string variable
:: 3. calls a specific anomaly detection algorithm
:: 4. plots the resulting anomaly scores in addition to the time series
:: INPUT: #1=name of directory, all files in that directory are processed
::		  in anomaly detection algorithm 
::		  #2= name of algorithm
:: OUTPUT:
::=========================================================================== 
set ddirectory =%1
set algstring = %2 
:: read contents of json file and assign it to variable jstring, without \n or \t
echo %algstring% 
FOR /F "tokens=*" %%g IN ('jq -rc . manifest.json') do (SET jstring=%%g)

:: some fields are missing in json file manifest.json, need to be added 
set executionType=execute
:: loop over all files in ddirectory
FOR %%f in (%ddirectory%\*) do (
	set dataInput=%f%
	set dataOutput=C:/Users/Iris/Documents/IU-Studium/Masterarbeit/02_Results/%algstring%/%f~n%_results.csv 
	:: add new attributes to json-string
	:: leave out first and last string of jstring
	set trunc=%jstring:~1,-1% 
	set jstring={'dataInput':'%dataInput%','dataOutput':'%dataOutput%','executionType':'%executionType%',%trunc%}

	:: double quotes need to be replaced with single quotes
	:: because otherwise anaconda powershell will swallow them
	set jstring="%jstring:"='%"

	:: invoke specific anomaly detection algorithm and pass variable jstring as input argument
	python %algstring%/algorithm_iris.py %jstring%
	:: plot results
	python ../plot-scores.py -d %dataInput% -s %dataOutput%
)



 

