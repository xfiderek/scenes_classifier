import argparse
import json 

from core.ModelTrainer import ModelTrainer

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="this script allows to re-train neural network or resume training from saved model. " +  
                                                            "Config is available in cfg/training.json")
        self.parser.add_argument("--model_path", dest="load_path",metavar="(str)", default="", help="path to saved model. used to resume training")
        
    def _parse_args(self):
        args = self.parser.parse_args()
        return args 
    
    def _get_json(self):
        with open("cfg/training.json") as f:
            cfg = json.load(f)
            params = cfg["params"]
            paths = cfg["paths"]
        
        return (params, paths)

    def get_cfg(self):
        saved_model = self._parse_args().load_path
        params, paths = self._get_json()

        return (saved_model, params, paths)


if __name__ == "__main__":
    parser = Parser()
    saved_model, params, paths = parser.get_cfg()
    trainer = ModelTrainer(save_path=paths["save_path"], train_csv=paths["train_csv"],
                           valid_csv=paths["test_csv"], data_dir=paths["data_dir"],
                           load_path=saved_model, disable_cuda=params["disable_cuda"])
    trainer.train(params["epochs"], params["batch_size"], 
                    params["print_loss_every"], params["save_model_every"],
                    params["validate_every"])