import os
import json
import yaml
import shutil
import random
import tempfile
from labelling import labelmetococo as ltoc
from Custom import Utils as U
'''
1. import Cyolo
2. Cyolo(ratio=["float : float : *float"])
3. Select path to import images and jsons
4. Select path to export data
'''
class Customyolo():
    def __init__(self, ratio):
        self.import_path = None
        self.export_path = None
        self.ratio = ratio
        self.__main__()
        
    def __initialize_path__(self):
        if self.import_path is None and self.export_path is None:
            self.import_path = U.requestpath(path_name="Image Import")
            self.export_path = U.requestpath(path_name="Dataset Export")

    def __make_path__(self):
        os.mkdir(f"{self.export_path}/images")
        os.mkdir(f"{self.export_path}/labels")
        os.mkdir(f"{self.export_path}/images/train")
        os.mkdir(f"{self.export_path}/images/val")
        os.mkdir(f"{self.export_path}/labels/train")
        os.mkdir(f"{self.export_path}/labels/val")

    def __label2cat__(self, sub_import_path):
        label_path = f"{sub_import_path}/labels.txt"
        labels = open(label_path, "r")
        datas = labels.readlines()
        del datas[0:2]
        list_dict = {}
        i = 0
        for data in datas:
            list_dict[ data.strip()] = i
            i += 1
        return list_dict

    def __json2txt__(self, list_dict, sub_import_path, sub_export_path):
        for file in os.scandir(sub_import_path):
            if file.name.lower().endswith('.json'):
                with open(file) as f:
                    data = json.load(f)
                    size = [data["imageWidth"], data["imageHeight"]]
                    d_txt = {}
                    lines = 0
                    for shape in data["shapes"]:
                        label = int(list_dict[shape["label"]])
                        norm_lefttop = [float(shape["points"][0][0]) / size[0], float(shape["points"][0][1]) / size[1]]
                        norm_rightbottom = [float(shape["points"][1][0]) / size[0], float(shape["points"][1][1]) / size[1]]
                        x = (norm_lefttop[0] + norm_rightbottom[0]) / 2
                        y = (norm_lefttop[1] + norm_rightbottom[1]) / 2
                        w = norm_rightbottom[0] - norm_lefttop[0]
                        h = norm_rightbottom[1] - norm_lefttop[1]
                        d_txt[lines] = [label, x, y, w, h]
                        lines += 1
                    result = open(f"{sub_export_path}/{file.name.replace(".json", ".txt")}", "w")
                    text = ""
                    for i in range(0, lines):
                        text = f"{d_txt[i][0]} {d_txt[i][1]} {d_txt[i][2]} {d_txt[i][3]} {d_txt[i][4]}\n"
                        result.write(text)
                    result.close()

    def __separator__(self, sub_import_path, sub_export_path, sub_ratio):
        ratios = sub_ratio.split(":")
        if len(ratios) > 3:
            print("Invalid Input Ratio")
        else:
            files = os.listdir(sub_import_path)
            files_filtered = [file for file in files if file.endswith(".jpg")]

            filtered_count = len(files_filtered)
            sum = 0
            for i in ratios:
                sum = sum + float(i)

            train_count = 0
            val_count = 0

            if len(ratios) == 2:
                train_count = int((filtered_count / sum) * float(ratios[0]))
                val_count = filtered_count - train_count
                train_datas = random.sample(files_filtered, train_count)
                val_datas = list(set(files_filtered) - set(train_datas))

                for file in train_datas:
                    t_file = file.replace(".jpg", ".txt")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/images/train/{file}")
                    shutil.copy(f"{sub_import_path}/{t_file}", f"{sub_export_path}/labels/train/{t_file}")

                for file in val_datas:
                    t_file = file.replace(".jpg", ".txt")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/images/val/{file}")
                    shutil.copy(f"{sub_import_path}/{t_file}", f"{sub_export_path}/labels/val/{t_file}")

                
                return True
            
            else:
                train_count = int((filtered_count / sum) * float(ratios[0]))
                val_count = int((filtered_count / sum) * float(ratios[1]))
                train_datas = random.sample(files_filtered, train_count)
                val_datas = random.sample(list(set(files_filtered) - set(train_datas)), val_count)
                test_datas = list(set(files_filtered) - set(train_datas) - set(val_datas))

                for file in train_datas:
                    t_file = file.replace(".jpg", ".txt")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/images/train/{file}")
                    shutil.copy(f"{sub_import_path}/{t_file}", f"{sub_export_path}/labels/train/{t_file}")

                for file in val_datas:
                    t_file = file.replace(".jpg", ".txt")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/images/val/{file}")
                    shutil.copy(f"{sub_import_path}/{t_file}", f"{sub_export_path}/labels/val/{t_file}")

                for file in test_datas:
                    t_file = file.replace(".jpg", ".txt")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/images/test/{file}")
                    shutil.copy(f"{sub_import_path}/{t_file}", f"{sub_export_path}/labels/test/{t_file}")
                
                return False
            
    def __create_yaml__(self, list_dict, sub_import_path, sub_export_path, withTest=False):
        names_dict = {value: key for key, value in list_dict.items()}

        yaml_structure = {
            "path": sub_export_path,
            "train": "images/train",
            "val": "images/val"
        }
        
        if withTest:
            yaml_structure["test"] = "images/test"
        else:
            yaml_structure["test"] = None

        yaml_structure["names"] = names_dict
           
        with open(f"{sub_export_path}/data.yml", "w") as y:
            yaml.dump(yaml_structure, y, default_flow_style=False, sort_keys=False)

    def __copy_image__(self, sub_import_path, sub_export_path):
        for file in os.scandir(sub_import_path):
            if file.name.lower().endswith('.jpg'):
                shutil.copy(file.path, os.path.join(sub_export_path, file.name))

    def __main__(self):
        self.__initialize_path__()
        self.__make_path__()
        list = self.__label2cat__(sub_import_path=self.import_path)
        with tempfile.TemporaryDirectory() as tempdir1:
            self.__json2txt__(list_dict=list, sub_import_path=self.import_path, sub_export_path=tempdir1)
            self.__copy_image__(sub_import_path=self.import_path, sub_export_path=tempdir1)
            self.__separator__(sub_import_path=tempdir1, sub_export_path=self.export_path, sub_ratio=self.ratio)
            self.__create_yaml__(list_dict=list, sub_import_path=self.import_path, sub_export_path=self.export_path)
        print("Converting Success!")