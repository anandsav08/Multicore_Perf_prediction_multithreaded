from dataclasses import replace
import pandas as pd

df = pd.read_csv("data.csv")
size_dict = {"simsmall" : 0, "simmedium" : 1, "simlarge" : 2}

def remove_cols_with_few_vals():
    counts = df.nunique()
    print(counts)
    to_del = [df.columns[i] for i,v in enumerate(counts) if (float(v)/df.shape[0]*100) < 3]
    print("Cols to be deleted: ",to_del)
    df.drop(to_del,axis=1,inplace=True)
    
def replace_string_with_unique_vals():
    p = df['problem_size']
    print(p)
    df.problem_size = [size_dict[item] for item in df.problem_size]

def saveDataFrame():
    df.to_csv("clean.csv",sep=",",encoding="utf-8")

def loadCleanData():
    df = pd.read_csv("clean.csv")

def main():
    replace_string_with_unique_vals()
    remove_cols_with_few_vals() 
    saveDataFrame()
    loadCleanData()

if __name__ == "__main__":
    main()