import os
import json
import math

def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def read_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    
    return content

def round_up_dividend(num, divisor):
    return num - (num % divisor) + (divisor * num % divisor)

def round(num):
    num = math.ceil(num) if num - math.floor(num) > 0.5 else math.floor(num) 
    
    return num