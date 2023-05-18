
import sqlite3
from check import runCam
import os

conn = sqlite3.connect(str(os.getcwd()) + '/Database.db')

query = "select * from camm"

data = conn.execute(query) 

result = data.fetchall()

print(data)
print(result[0][4])

for cam in result : 
    print(cam)
    runCam(cam)
