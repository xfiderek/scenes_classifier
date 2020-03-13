from core.Classifier import Classifier
import argparse 

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="this script allows to classify images from csv file and save it to another csv")
        self.parser.add_argument("--csv_file", dest="csv_file",metavar="(str)", default="csv/intel_predict.csv", help="path to csv with images")
        self.parser.add_argument("--data_dir", dest="data_dir",metavar="(str)", default="./dataset", help="path to directory with data")
        self.parser.add_argument("--write_to", dest="write_to",metavar="(str)", default="csv/predicted.csv", help="path to csv to which write images and labels")
        self.parser.add_argument("--model_path", dest="model_path",metavar="(str)", default="saved_model/model.pth", help="path to saved model")
        self.parser.add_argument("--disable_cuda", dest="disable_cuda", metavar="(bool)", default=False)
    
    def get_args(self):
        return self.parser.parse_args()


def classify(args):
    classifier = Classifier(args.model_path, args.disable_cuda)
    classifier.classify_from_csv(args.data_dir, args.csv_file, args.write_to)

if __name__ == "__main__":
    parser = Parser()
    classify(parser.get_args())
