import subprocess
import pandas as pd
from io import StringIO
from datetime import datetime
# Define your Go program command
go_program = 'go run main.go'
def btc_scrapper(now_str):

    # Use subprocess to run the Go program and capture the output
    process = subprocess.Popen(go_program, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # If there's an error, print it out
    if process.returncode != 0:
        print(f'Error occurred: {stderr.decode()}')
    else:
        # If there's no error, parse the output into a DataFrame
        output = stdout.decode()
        data = StringIO(output)  # Convert the string output to a file-like object for pandas
        df = pd.read_csv(data, sep='\t')  # Assume the output is tab-separated; adjust as necessary
        print(df)   
        df.to_csv('btc'+now_str+'.csv') 
    # The DataFrame df now contains the output of the Go program
