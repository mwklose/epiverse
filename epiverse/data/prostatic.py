import pandas as pd
import os


class ProstaticData:

    def retrieve_data():
        data = pd.read_excel(
            "epiverse/data/Prostatic.xlsx", "Data", na_values="-")
        return data

    def retrieve_codebook():
        codebook = pd.read_excel("epiverse/data/Prostatic.xlsx", "Codebook")
        return codebook
