import os
import site
from pathlib import Path

class MADApp:

    def __init__(self):
        print("INFO: Initializing MAD Framework")
        print("INFO: Installing required packages and models.")
        self.install_farm()
        self.pretrained_model()

    
    def install_farm(self):
        try:
            os.system("pip install git+https://github.com/imdiptanu/FARM.git --quiet")
        except Exception as e:
            print("ERROR: Installing FARM! Error:", e)


    def pretrained_model(self):
        try:
            print("INFO: Downloading mad-bert-uncased model from remote.")
            cur_path = site.getsitepackages()[0] + "/mad_fw/"
            model_path = cur_path + "mad-bert-uncased.zip"
            if not os.path.isfile(model_path):
                os.system(f"wget https://www.dropbox.com/s/sfvwyn3bw49x93j/mad-bert-uncased.zip -P {cur_path}")
            self.model_path = cur_path + "content/trained-models/mad-bert-uncased/"
            if not os.path.isdir(self.model_path):
                os.system(f"unzip {model_path} -d {cur_path}")
            print("INFO: Model downloaded and unpacked Successfully.")
        except Exception as e:
            print("ERROR: Downloading pre-trained model! Error:", e)


    def mad_predict(self, text):
        basic_texts = [
            {"text": text},
        ]
        save_dir = Path(self.model_path)
        from mad_fw.mtl_processor import MTLProcessor
        from farm.infer import Inferencer
        model = Inferencer.load(save_dir)
        result = model.inference_from_dicts(dicts=basic_texts)
        model.close_multiprocessing_pool()
        label_predictions_list, tokens_predictions_list = [], []
        for idx, chunk_res in enumerate(result):
            if idx % 2 == 0:
                label_predictions_list += chunk_res["predictions"]
            else:
                tokens_predictions_list += chunk_res["predictions"]

        # Tokens predictions
        tokens_list = []
        for idx, pred_ind_list in enumerate(tokens_predictions_list):
            ind_list = []
            for val_dict in pred_ind_list:
                label_val = val_dict["label"]
                ind_list.append(0 if label_val == "X" else int(label_val))
            tokens_list.append(ind_list)

        # Labels predictions
        label_list = []
        for idx, pred_dict in enumerate(label_predictions_list):
            label_list.append(pred_dict["label"])
        print ("="*15 + " Prediction " + "="*15)
        print("Text:", text)
        print("Label:", label_list[0])
        print("Tokens:", tokens_list[0])
