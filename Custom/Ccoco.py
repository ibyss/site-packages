import os
import random
import shutil
import tempfile
from Custom import labelmetococo as ltoc
from Custom import Utils as U
'''
1. import Customcoco
2. Customcoco(ratio=["float : float : *float"])
3. Select path to import images and jsons(created with labelme)
4. Select path to export data
'''
class Customcoco():
    def __init__(self, ratio):
        self.import_path = None
        self.export_path = None
        self.ratio = ratio
        self.__main__()

    def __initialize_path__(self):
        if self.import_path is None and self.export_path is None:
            self.import_path = U.requestpath(path_name="Image Import")
            self.export_path = U.requestpath(path_name="Dataset Export")

    def __separator__(self, ratio: list, sub_import_path, sub_export_path):
        '''
        ratio = [float : float : *float](train : val : *test의 비율)
        '''
        ratios = ratio.split(":")
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
                os.mkdir(f"{sub_export_path}/train")
                os.mkdir(f"{sub_export_path}/val")
                for file in train_datas:
                    j_file = file.replace("jpg", "json")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/train/{file}")
                    shutil.copy(f"{sub_import_path}/{j_file}", f"{sub_export_path}/train/{j_file}")

                for file in val_datas:
                    j_file = file.replace("jpg", "json")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/val/{file}")
                    shutil.copy(f"{sub_import_path}/{j_file}", f"{sub_export_path}/val/{j_file}")
                
                return True
            
            else:
                train_count = int((filtered_count / sum) * float(ratios[0]))
                val_count = int((filtered_count / sum) * float(ratios[1]))
                train_datas = random.sample(files_filtered, train_count)
                val_datas = random.sample(list(set(files_filtered) - set(train_datas)), val_count)
                test_datas = list(set(files_filtered) - set(train_datas) - set(val_datas))
                os.mkdir(f"{sub_export_path}/train")
                os.mkdir(f"{sub_export_path}/val")
                os.mkdir(f"{sub_export_path}/test")
                for file in train_datas:
                    j_file = file.replace("jpg", "json")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/train/{file}")
                    shutil.copy(f"{sub_import_path}/{j_file}", f"{sub_export_path}/train/{j_file}")

                for file in val_datas:
                    j_file = file.replace("jpg", "json")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/val/{file}")
                    shutil.copy(f"{sub_import_path}/{j_file}", f"{sub_export_path}/val/{j_file}")

                for file in test_datas:
                    j_file = file.replace("jpg", "json")
                    shutil.copy(f"{sub_import_path}/{file}", f"{sub_export_path}/test/{file}")
                    shutil.copy(f"{sub_import_path}/{j_file}", f"{sub_export_path}/test/{j_file}")
                
                return False

    def __labelme2coco__(self, set_type, sub_import_path, sub_export_path):
        label_path = f"{sub_import_path}/labels.txt"
        if set_type == "train":
            import_data_path = f"{sub_import_path}/train"
            export_data_path = f"{sub_export_path}"
        elif set_type == "val":
            import_data_path = f"{sub_import_path}/val"
            export_data_path = f"{sub_export_path}"
        else:
            import_data_path = f"{sub_import_path}/test"
            export_data_path = f"{sub_export_path}"

        ltoc.create_dataset(input_dir=import_data_path, output_dir=export_data_path, labels_file=label_path, data_type=set_type, noviz=True)

    def __main__(self):
        ratio = self.ratio
        self.__initialize_path__()
        with tempfile.TemporaryDirectory() as tempdir1:
            shutil.copy(f"{self.import_path}/labels.txt", f"{tempdir1}/labels.txt")
            with_test = self.__separator__(ratio=ratio, sub_import_path=self.import_path, sub_export_path=tempdir1)
            if with_test:
                self.__labelme2coco__(set_type="train", sub_import_path=tempdir1, sub_export_path=self.export_path)
                self.__labelme2coco__(set_type="val", sub_import_path=tempdir1, sub_export_path=self.export_path)
                self.__labelme2coco__(set_type="test", sub_import_path=tempdir1, sub_export_path=self.export_path)
            else:
                self.__labelme2coco__(set_type="train", sub_import_path=tempdir1, sub_export_path=self.export_path)
                self.__labelme2coco__(set_type="val", sub_import_path=tempdir1, sub_export_path=self.export_path)