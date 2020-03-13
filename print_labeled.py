import argparse
import cv2 
import pandas as pd 
import os 
from core.classes_dict import lbl_to_name

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="this script allows to print  labeled images from csv file. By default it shows images from seg_pred dataset predicted by tis nn")
        self.parser.add_argument("--csv_file", dest="csv_file",metavar="(str)", default="csv/predicted.csv", help="path to csv")
        self.parser.add_argument("--data_dir", dest="data_dir",metavar="(str)", default="./dataset", help="path to directory with data")

    def get_args(self):
        return self.parser.parse_args()

def show(args):
    df = pd.read_csv(args.csv_file)
    #shuffle data
    df = df.sample(frac=1)
    
    print("PRESS Q TO QUIT")
    for _, row in df.iterrows():
        rel_path, label = row

        img_path = os.path.join(args.data_dir, rel_path)
        cv2.imshow('img', cv2.imread(img_path))
        img_label = lbl_to_name[label]
        print(img_label)
        q = cv2.waitKey(0)
        if q == ord('q'):
            break

if __name__ == "__main__":
    parser = Parser()
    args = parser.get_args()
    show(args)




