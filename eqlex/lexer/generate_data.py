import yaml

import pkgutil
from pprint import pprint

pprint([mod for mod in pkgutil.iter_modules()])


from lexer.tokens.scanner import Scanner

def run_example() -> None:

    with open("yaml/example.yaml", "r") as t:
        config = yaml.safe_load(t)


    print(config)

    print(Scanner("1 + 2 + X + Norm(1, 5)"))

if __name__ == "__main__":
    run_example()
