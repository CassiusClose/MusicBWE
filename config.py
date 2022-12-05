import yaml
try:
    file = open('config.yml', 'r')
    CONFIG = yaml.safe_load(file)
except:
    print("Error opening config file: config.yml")
    exit()
