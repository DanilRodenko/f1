from src.data.data_loader import load_all_cvs

data_dict = load_all_cvs("C:/Users/danil/Desktop/PROJECTS/formula1/datasets/raw/")
data_dict['races'].head()