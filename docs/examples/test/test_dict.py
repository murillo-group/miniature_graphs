import json
import sys

with open(sys.argv[1],'r') as json_file:
   targets_and_weights = json.load(json_file)
