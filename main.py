import pandas as pd
import numpy as np

input_len = 128

def read_data(filename):
    df=pd.read_csv(filename, sep="\t",
                   usecols=["product_name","brands","additives"])
    df=df.dropna()
    print("[INFO] Data loaded")
    return df


def parse_additives(expr):
    l_expr = [x.split("->")[0].strip() for x in expr.split('[') if "en:" in x]
    l_expr = [x for x in l_expr if x]
    return l_expr
    

def main():
    global df, hist, names
    
    df = read_data("data.tsv")
    df["parsed_additives"] = df["additives"].apply(parse_additives)
    df["length"] = df["parsed_additives"].apply(lambda x: len(x))
    df = df[df["length"] > 0]    
    print("[INFO] Data Parsed")

    hist = {}
    def add_to_hist(el):
        for x in el:
            if x in hist:
                hist[x] += 1
            else:
                hist[x] = 1
    
    df["parsed_additives"].apply(add_to_hist)
    print("[INFO] Histogram Created")
    hist = sorted(hist.items(), key=lambda item: item[1], reverse=True)
    hist = np.array(hist[:input_len])
    names = hist[:, 0]
    hist = hist[:,1].astype(np.uint32)
    print("[INFO] Histogram Sorted")
    
    def additive_to_vec(additives):
        vec = np.zeros(input_len)
        for x in additives:
            if x in names:
                vec[np.where(names==x)] = 1
        return vec
    
    df["additives"] = df["parsed_additives"].apply(additive_to_vec)
    print("[INFO] Additive Vector Created")
    
    df = df.reset_index()
    df = df.drop(columns=["index", "parsed_additives", "length"])
    print("[INFO] Dataframe is created")
    

if __name__ == "__main__":
    main()

