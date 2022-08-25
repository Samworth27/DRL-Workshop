import math

def timeString(seconds):
  seconds_elapsed = math.floor(i)
  minutes_elapsed = math.floor(seconds_elapsed/60)
  hours_elapsed = math.floor(minutes_elapsed/60)
  days_elapsed = math.floor(hours_elapsed/24)
  return f'{days_elapsed} days  {(hours_elapsed%24):02} hours  {(minutes_elapsed%60):02} mins  {(seconds_elapsed%60):02} seconds'
  
for i in range(0,100,1):
  print(timeString(i))
  
