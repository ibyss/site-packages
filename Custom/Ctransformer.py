import torch
import os
import json
import tempfile
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
from Custom import Utils as U
'''
1. from Customtransformer import Custom as C
2. from Custom.Customtransformer import Compose/SeqCompose
'''
class Transform:
    def __init__(self, withJSON=True):
        self.import_path = None
        self.export_path = None
        self.withJSON = withJSON
    
    def __initialize_path__(self):
        if self.import_path == None and self.import_path == None:
            self.import_path = U.requestpath(path_name="Raw Images and JSONs")
            self.export_path = U.requestpath(path_name="Transformed Images and JSONs")
    
    def __call__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def __jsontransform__(self, type: str, import_path: str, export_path: str, name: str, t_size=[], degree=0):
        '''
        type=str(rotate, hflip, vflip), import_path=str(불러올 경로), export_path=str(저장할 경로), *t_size=[int, int](이미지 크기), *degree=int(각도)
        '''
        with open(import_path) as f:
            data = json.load(f)

        size = (data["imageWidth"], data["imageHeight"])
        #imagepath = data["imagePath"]

        def _main(box_data, t_size):
            lefttop = [box_data[0][0], box_data[0][1]]
            rightbottom = [box_data[1][0], box_data[1][1]]

            r_lefttop = [lefttop[0] / size[0], lefttop[1] / size[1]]
            r_rightbottom = [rightbottom[0] / size[0], rightbottom[1] / size[1]]
        
            if type == "hflip":
                t_lefttop = [size[0] - rightbottom[0], lefttop[1]]
                t_rightbottom = [size[0] - lefttop[0], rightbottom[1]]
            elif type == "vflip":
                t_lefttop = [lefttop[0], size[1] - rightbottom[1]]
                t_rightbottom = [rightbottom[0], size[1] - lefttop[1]]
            elif type == "resize":
                if not t_size:
                    raise ValueError("Transformed size (t_size) is an empty sequence")
                t_lefttop = [r_lefttop[0] * t_size[0], r_lefttop[1] * t_size[1]]
                t_rightbottom = [r_rightbottom[0] * t_size[0], r_rightbottom[1] * t_size[1]]
            elif type == "rotate":
                if degree == 0:
                    raise ValueError("Degree (degree) is zero, no rotation needed")
                elif degree == 90:
                    t_lefttop = [lefttop[1], size[0] - rightbottom[0]]
                    t_rightbottom = [rightbottom[1], size[0] - lefttop[0]]
                elif degree == 180:
                    t_lefttop = [size[0] - rightbottom[0], size[1] - rightbottom[1]]
                    t_rightbottom = [size[0] - lefttop[0], size[1] - lefttop[1]]
                elif degree == 270:
                    t_lefttop = [size[1] - rightbottom[1], lefttop[0]]
                    t_rightbottom = [size[1] - lefttop[1], rightbottom[0]]
                else:
                    raise ValueError("Invalid degree(degree)")
            else:
                t_lefttop = lefttop
                t_rightbottom = rightbottom

            return [t_lefttop, t_rightbottom]
        
        if type == "resize":
            data["imageWidth"] = t_size[0]
            data["imageHeight"] = t_size[1]
            data["imagePath"] = name

        elif type == "rotate":
            if degree == 90 or degree == 270:
                data["imageWidth"] = size[1]
                data["imageHeight"] = size[0]
            data["imagePath"] = name

        elif type == "hflip" or type == "vflip":
            data["imagePath"] = name
        
        elif type == "normalize" or type == "colorjit":
            data["imagePath"] = name
                
        for shape in data["shapes"]:
            shape["points"] = _main(shape["points"], t_size=t_size if type == "resize" else [])

        with open(export_path, 'w') as f:
            json.dump(data, f)
    
class image2tensor(Transform):
    def __init__(self, image):
        self.image = image
    def __call__(self):
        _image2tensor = transforms.Compose([transforms.ToTensor()])
        return _image2tensor(self.image), "Convert Image to Tensor"
    
class tensor2image(Transform):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
    def __call__(self):
        _tensor2image = transforms.Compose([transforms.ToPILImage()])
        return _tensor2image(self.tensor), "Convert Tensor to Image"

class resize(Transform):
    def __init__(self, size=(), pixels=None):
        '''
        size=[가로, 세로]
        '''
        super().__init__()
        if len(size) == 0:
            self.width = 0
            self.height = 0
        else:
            self.width, self.height = size
        self.pixels = pixels
    def __call__(self, compose_path=None, temp_path=None):
        sub_import_path = compose_path if compose_path else self.import_path
        sub_export_path = temp_path if temp_path else self.export_path    
        for file in os.scandir(sub_import_path):
            if file.is_file() and file.name.lower().endswith('.jpg'):
                target = Image.open(file.path)
                if self.pixels:
                    width, height = target.size
                    ratio = height/width
                    width = int((self.pixels / ratio)**0.5)
                    height = int(width * ratio)
                else:
                    width = self.width
                    height = self.height
                _resize = transforms.Compose([transforms.Resize((height, width))])
                result = _resize(target)
                target_name = f"resize_{os.path.basename(file.path)}"
                path = os.path.join(sub_export_path, target_name)
                result.save(path, "jpeg")

                if self.withJSON:
                    j_import_path = file.path.replace("jpg", "json")
                    j_export_path = path.replace("jpg", "json")
                    j_name = target_name
                    self.__jsontransform__(type="resize", import_path=j_import_path, export_path=j_export_path, name=j_name, t_size=[width, height])
        return sub_export_path, f"Resize to {width} X {height}"

class rotate(Transform):
    def __init__(self, degree: int):
        '''
        degree=int(각도)
        '''
        super().__init__()
        self.degree = degree
    def __call__(self, compose_path=None, temp_path=None):
        sub_import_path = compose_path if compose_path else self.import_path
        sub_export_path = temp_path if temp_path else self.export_path
        for file in os.scandir(sub_import_path):
            if file.is_file() and file.name.lower().endswith('.jpg'):
                target = Image.open(file.path)
                result = F.rotate(target, angle=self.degree, expand=True)
                target_name = f"{self.degree}rotate_{os.path.basename(file.path)}"
                path = os.path.join(sub_export_path, target_name)
                result.save(path, "jpeg")

                if self.withJSON:
                    j_import_path = file.path.replace("jpg", "json")
                    j_export_path = path.replace("jpg", "json")
                    j_name = target_name
                    self.__jsontransform__(type="rotate", import_path=j_import_path, export_path=j_export_path, name=j_name, degree=self.degree)
        return sub_export_path, f"Rotate {self.degree} degree"

class hflip(Transform):    
    def __init__(self, probability=1, compose_path=None, temp_path=None):
        '''
        probability=float(0~1의 확률)
        '''
        super().__init__()
        self.probability = probability
        self._hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=probability)])
    def __call__(self, compose_path = None, temp_path=None):
        sub_import_path = compose_path if compose_path else self.import_path
        sub_export_path = temp_path if temp_path else self.export_path
        for file in os.scandir(sub_import_path):
            if file.is_file() and file.name.lower().endswith('.jpg'):
                target = Image.open(file.path)
                result = self._hflip(target)
                target_name = f"hflip_{os.path.basename(file.path)}"
                path = os.path.join(sub_export_path, target_name)
                result.save(path, "jpeg")

                if self.withJSON:
                    j_import_path = file.path.replace("jpg", "json")
                    j_export_path = path.replace("jpg", "json")
                    j_name = target_name
                    self.__jsontransform__(type="hflip", import_path=j_import_path, export_path=j_export_path, name=j_name)
        return sub_export_path, "Horizontal Flip"

class vflip(Transform):
    def __init__(self, probability=1, compose_path=None, temp_path=None):
        '''
        probability=float(0~1의 확률)
        '''
        super().__init__()
        self.probability = probability
        self._vflip = transforms.Compose([transforms.RandomVerticalFlip(p=probability)])
    def __call__(self, compose_path=None, temp_path=None):
        sub_import_path = compose_path if compose_path else self.import_path
        sub_export_path = temp_path if temp_path else self.export_path
        for file in os.scandir(sub_import_path):
            if file.is_file() and file.name.lower().endswith('.jpg'):
                target = Image.open(file.path)
                result = self._vflip(target)
                target_name = f"vflip_{os.path.basename(file.path)}"
                path = os.path.join(sub_export_path, target_name)
                result.save(path, "jpeg")

                if self.withJSON:
                    j_import_path = file.path.replace("jpg", "json")
                    j_export_path = path.replace("jpg", "json")
                    j_name = target_name
                    self.__jsontransform__(type="vflip", import_path=j_import_path, export_path=j_export_path, name=j_name)
        return sub_export_path, "Vertical Flip"

class normalize(Transform):
    def __init__(self, mean: list, std: list):
        '''
        mean=[float, float, float](R,G,B의 평균), std=[float, float, float](R,G,B의 표준편차)
        '''
        super().__init__()
        self.mean = mean
        self.std = std
        self._normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), transforms.ToPILImage()])
    def __call__(self, compose_path=None, temp_path=None):
        sub_import_path = compose_path if compose_path else self.import_path
        sub_export_path = temp_path if temp_path else self.export_path
        for file in os.scandir(sub_import_path):
            if file.is_file() and file.name.lower().endswith('.jpg'):
                target = Image.open(file.path)
                result = self._normalize(target)
                target_name = os.path.basename(file.path)
                path = os.path.join(sub_export_path, target_name)
                result.save(path, "jpeg")

                if self.withJSON:
                    j_import_path = file.path.replace("jpg", "json")
                    j_export_path = path.replace("jpg", "json")
                    j_name = target_name
                    self.__jsontransform__(type="normalize", import_path=j_import_path, export_path=j_export_path, name=j_name)
        return sub_export_path, f"normalize(Mean:{self.mean}, Standard Deviation:{self.std})"

class colorjit(Transform):
    def __init__(self, brightness, contrast, saturation, hue):
        '''
        brightness=float(o~1의 밝기), contrast=float(0~1의 대비), contrast=float(0~1의 채도), hue=float(0~1의 색조)
        '''
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._colorjit = transforms.Compose([transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)])
    def __call__(self, compose_path=None, temp_path=None):
        sub_import_path = compose_path if compose_path else self.import_path
        sub_export_path = temp_path if temp_path else self.export_path
        for file in os.scandir(sub_import_path):
            if file.is_file() and file.name.lower().endswith('.jpg'):
                target = Image.open(file.path)
                result = self._colorjit(target)
                target_name = f"colorjit_{os.path.basename(file.path)}"
                path = os.path.join(sub_export_path, target_name)
                result.save(path, "jpeg")

                if self.withJSON:
                    j_import_path = file.path.replace("jpg", "json")
                    j_export_path = path.replace("jpg", "json")
                    j_name = target_name
                    self.__jsontransform__(type="colorjit", import_path=j_import_path, export_path=j_export_path, name=j_name)
        return sub_export_path, f"Color Jitter(Brightness:{self.brightness}, Contrast:{self.contrast}, Saturation: {self.saturation}, Hue: {self.hue})"
    
class SeqCompose(Transform):
    def __init__(self, transforms):
        self.__initialize_path__()
        self.transforms = transforms
        super().__init__()
        with tempfile.TemporaryDirectory() as tempdir1, tempfile.TemporaryDirectory() as tempdir2:
            tempdirs = [tempdir1, tempdir2]
            i = 0
            compose_path = self.import_path
            for transform in self.transforms:
                temp_path = tempdirs[i % 2]
                if i == len(self.transforms) - 1:
                    temp_path = self.export_path
                return_value = transform(compose_path=compose_path, temp_path=temp_path)
                compose_path = return_value[0]
                print(f"Transform ({i+1}) Done: Transform Type: {return_value[1]}")
                i += 1

class Compose(Transform):
    def __init__(self, transforms):
        self.__initialize_path__()
        self.transforms = transforms
        super().__init__()
        compose_path = self.import_path
        temp_path = self.export_path
        i = 0
        for transform in self.transforms:
            return_value = transform(compose_path=compose_path, temp_path=temp_path)
            print(f"Transform ({i+1}) Done: Transform Type: {return_value[1]}")
            i += 1

class Rename(Transform):
    def __init__(self):
        #super().__init__()
        #self.__initialize_path__()
        self.path="C:/Users/chanj/Desktop/123/새 폴더/testre"
    def __call__(self):
        i = 1
        for file in os.listdir(self.path):
            if file.endswith('.jpg'):
                _jpg = os.path.join(self.path, file)
                _json = _jpg.replace(".jpg", ".json")
                new_jpg = os.path.join(self.path, f"{str(i)}.jpg")
                new_json = os.path.join(self.path, f"{str(i)}.json")
                os.rename(_jpg, new_jpg)
                if os.path.exists(_json):
                    with open(_json) as f:
                        data = json.load(f)
                        data["imagePath"] = f"{str(i)}.jpg"
                        with open(new_json, 'w') as f:
                            json.dump(data, f)

                    os.remove(_json)
                
                i += 1
                