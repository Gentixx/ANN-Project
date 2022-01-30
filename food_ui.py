import tkinter as tk
from enum import Enum, auto
import pandas as pd
import numpy as np
from food_pred import *


class FoodPredictor:
    class State(Enum):
        Start = auto()
        Train = auto()
        Test = auto()
    
    def __init__(self, input_len=128, test_len=15):
        self.state = FoodPredictor.State.Start
        self.input_len = input_len
        self.test_len = test_len
        self.is_open = True
    
    def run(self):
        self.__initialize()
        while(self.is_open):
            self.__update()
        self.window.destroy()
    
    def __initialize(self):
        # initialize data
        self.df = self.read_data("data.tsv")
    
        def parse_additives(expr):
            l_expr = [x.split("->")[0].strip() for x in expr.split('[') if "en:" in x]
            l_expr = [x for x in l_expr if x]
            return l_expr
        
        self.df["parsed_additives"] = self.df["additives"].apply(parse_additives)
        self.df = self.clear_data(self.df)
        # initialize predictor
        self.ai = Predictor(self.df)
        
        # initialize state
        self.__transition(FoodPredictor.State.Train)
    
    def __update(self):
        def get_grade():
            while True:
                a = input("What is your grade?\n")
                if a in ["1", "2", "3", "4"]:
                    return int(a)
            
        
        def show_item(item):
            print("Name: " + str(item["product_name"]))
            print("Brand: " + str(item["brands"]))
            print("1 - don't like it")
            print("2 - neutral")
            print("3 - like it")
            print("4 - never tried")
            print()
        
        if self.state == FoodPredictor.State.Train:
            if self.vars["item"] < self.test_len:
                print("Item: {}/{}".format(self.vars["item"]+1, self.test_len))
                x = self.df.iloc[self.ai.get_random()[0]]
                show_item(x)
                y = get_grade()
                if y < 4:
                    self.vars["X"].append(np.array(x["additives"]))
                    self.vars["y"].append(y-2)
                    self.vars["item"] += 1
            else:
                self.ai.init_train(np.array(self.vars["X"]), np.array(self.vars["y"]))
                self.__transition(FoodPredictor.State.Test)
        elif self.state == FoodPredictor.State.Test:
            idx, conf = self.ai.get_best_random()
            x = self.df.iloc[idx]
            print("Confidance: {:.2f}%".format(100*conf))
            show_item(x)
            y = get_grade()
            if y < 4:
                self.vars["X"].append(np.array(x["additives"]))
                self.vars["y"].append(y-2)
                # breakpoint()
                self.ai.update(np.array(self.vars["X"]), np.array(self.vars["y"]))
        
    def __transition(self, next_state):
        if next_state == FoodPredictor.State.Train:
            self.vars = {}
            self.vars["item"] = 0
            self.vars["X"] = []
            self.vars["y"] = []
            print(f"We will show you different food items and you should grade {self.test_len} of them that you tried before.")
            print()
        elif next_state == FoodPredictor.State.Test:
            print("We will show you some items that we think you can also like, please grade them.")
            print()
        
        self.state = next_state
        
    def read_data(self, filename):
        df = pd.read_csv(filename, sep="\t",
                       usecols=["product_name","brands","additives"])
        df = df.dropna()
        print("[INFO] Data Loaded")
        return df
        
    
    def clear_data(self, df):
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
        hist = np.array(hist[:self.input_len])
        names = hist[:, 0]
        hist = hist[:,1].astype(np.uint32)
        print("[INFO] Histogram Sorted")
        
        def additive_to_vec(additives):
            vec = np.zeros(self.input_len)
            for x in additives:
                if x in names:
                    vec[np.where(names==x)] = 1
            return vec
        
        df["additives"] = df["parsed_additives"].apply(additive_to_vec)
        print("[INFO] Additive Vector Created")
        
        df = df.reset_index()
        df = df.drop(columns=["index", "parsed_additives", "length"])
        print("[INFO] Dataframe is created")
        return df

    