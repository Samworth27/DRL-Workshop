import base64
import math
import numpy as np
import PIL
from PIL import Image
import base64

def timeString(time):
    seconds_elapsed = math.floor(time)
    minutes_elapsed = math.floor(seconds_elapsed/60)
    hours_elapsed = math.floor(minutes_elapsed/60)
    days_elapsed = math.floor(hours_elapsed/24)
    return f'{days_elapsed} days  {(hours_elapsed%24):02} hours  {(minutes_elapsed%60):02} mins  {(seconds_elapsed%60):02} seconds'
  
def cleanValue(value, config_type):
    valueType = type(value)
    if (valueType == dict):
        value = cleanDict(value, config_type)
    elif (valueType == list):
        value = cleanList(value, config_type)
    elif (valueType) == float:
        value == float(round(value, 5))
    elif (valueType in [str, int, bool, type(None)]):
        pass
    elif (valueType in [np.float32, np.float64]):
        value = float(round(value, 5))
    elif (valueType == np.ndarray):
        value = cleanList(value, config_type)
    elif (valueType == config_type):
        value == cleanDict(value, config_type)
    else:
        value = str(value)
    return value

def cleanList(dirtyList, config_type):
    cleanList = []
    for value in dirtyList:
        cleanList.append(cleanValue(value, config_type))
    return cleanList


def cleanDict(dictionary, config_type):
    for key, value in zip(dictionary.keys(), dictionary.values()):
        dictionary[key] = cleanValue(value, config_type)
    return dictionary

def encodeArray(array):
    shape = array.shape
    buffer = bytes(array.astype("float64"))
    encoded_string = base64.b64encode(buffer).decode("utf-8")
    return encoded_string, shape

def decodeArray(encoded_string, shape):
    decoded_bytes = base64.b64decode(bytes(encoded_string, 'utf-8'))
    array = np.frombuffer(decoded_bytes, "float64").reshape(shape)
    return array