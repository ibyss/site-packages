# site-packages
For data-processing utilities 

Use Utils.py to select paths, visualize custom model report from json file(wip)

Ctransformer.py: Using jsons created with labelme. With torchvision-transform and some custom json transforms, it can augment your images and annotations, without
                labelling your data again. It can perform image-tensor conversion, resize, rotate(90, 180, 270), horizontal-flip, vertical-flip, normalize, colorjit.

To perform each transform individually,

                Compose{[Ctransformer.A transform(), Ctransformer.B transform(), ...]}
                
-> result: A transformed images, B transformed images


To perform each transform sequentially,

                SeqCompose{[Ctransformer.transformA(), Ctransformer.transformB(), ...]} to perform each transform together.
                
-> result: A+B transformed image


Ccoco.py: Using jsons created with labelme. Select ratios between train : val : test(optional) datas and choose import, export path.
          Then it will convert your datas to cocodataset.


Cyolo.py: Using jsons created with labelme. Select ratios between train : val : test(optional) datas and choose import, export path.
          Then it will convert your datas to yolodataset.
