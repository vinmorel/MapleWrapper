import os
import pathlib
from os.path import join

def get_expbase(lvl):
    exp_table_pth = join(pathlib.Path(__file__).resolve().parents[0],"exp.txt")
    with open(exp_table_pth, "r") as f:
        l = f.readlines() 
        
        exp_list = [(line.split(" ")[0], line.split(" ")[2].strip())  for line in l]
        exp_table = dict(exp_list)
        
        base_exp = exp_table[str(lvl)]
        return base_exp

if __name__ == "__main__":
    b_exp = get_expbase(7)
    print(b_exp)