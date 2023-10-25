"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Oct 25 2023
*  File : omnidata_infer.py
******************************************* -->

"""

import os 
import numpy as np
import imageio
import time
from  run import OmnidataModel
from pathlib import Path
import matplotlib.pyplot as plt
from run import colorize
import glob

class BaseInfer(object):
    def __init__(self) -> None:
        self.load_models()
        pass
    def collect_files(self, image_dir, file_patter):
        image_files = f'{image_dir}/**/{file_patter}'
        image_files = glob.glob(image_files, recursive=True)
        image_files.sort()
        return image_files
    def process_files(self, image_files, base_dir):
        for src_file in image_files:
            cam_name = os.path.dirname(src_file)
            cam_name = os.path.basename(cam_name)

            frame_num = os.path.basename(src_file)
            frame_num = frame_num.split('rgb_')[1].split('.jpg')[0]

            tgt_monodepth_dir =  f"{base_dir}/mono_depth/{cam_name}"
            os.makedirs(tgt_monodepth_dir, exist_ok= True)
            tgt_monodepth_file = f"{tgt_monodepth_dir}/mono_depth_{frame_num}.npy"


            tgt_normal_dir =  f"{base_dir}/normal/{cam_name}"
            os.makedirs(tgt_normal_dir, exist_ok= True)
            tgt_normal_file = f"{tgt_normal_dir}/normal_{frame_num}.npy"

           
            use_depth, img = self.infer_image(src_file)
            tgt_file = tgt_normal_file
            if use_depth:
                tgt_file = tgt_monodepth_file

            np.save(tgt_file, img)
            print(f"saved file {tgt_file}")

        return
    def load_models(self):
        return
    def infer_image(self):
        return None
    def run(self):
        # return self.infer_image("/home/levin/workspace/nerf/mini-omnidata/assets/image.png", vis = True)
        # return self.infer_image("/media/levin/DATA/zf/workspace/data/vkitti/Scene06/clone/frames/rgb/Camera_0/rgb_00200.jpg", vis = True)

        image_dir = '/media/levin/DATA/zf/workspace/data/vkitti/Scene06/clone/frames/rgb'
        base_dir = os.path.dirname(image_dir)

        file_patter = "*.jpg"
        image_files = self.collect_files(image_dir, file_patter)
        self.process_files(image_files, base_dir)

        
        return  


class App(BaseInfer):
    def __init__(self):
        super().__init__()
        return
    def load_models(self):
        self.use_depth = True
        args_task = 'depth'
        if not self.use_depth:
            args_task = 'normal'
        args_model_path = None
        print("Loading model...")
        start = time.time()
        self.omnidata = OmnidataModel(args_task, args_model_path, device="cuda:0")
        end = time.time()
        print(f"Loading finished in {end-start} secs.")
        return
    
    
    
    def infer_image(self, image_path, vis = False):
        omnidata = self.omnidata
        image_path = Path(image_path)
        
        output = omnidata(image_path, down_factor=1.0)

        # #single frame visualization
        if vis:
            img = imageio.imread(image_path)
            output_vis = colorize(output)

            _, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
            ax1.imshow(img)
            ax2.imshow(output)
            ax3.imshow(output_vis)
            plt.show()

        return self.use_depth, output.squeeze()
    

if __name__ == "__main__":   
    obj= App()
    obj.run()
