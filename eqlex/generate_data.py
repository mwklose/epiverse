import yaml
import distributions 

with open("yaml/example.yaml", "r") as t: 
    config = yaml.safe_load(t)

print(config)




if __name__ == "__main__":
    ...