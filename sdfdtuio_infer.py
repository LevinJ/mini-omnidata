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
import imageio

class BaseInfer(object):
    def __init__(self) -> None:
        self.load_models()
        pass
    def collect_files(self, image_dir, file_patter):
        image_files = f'{image_dir}/**/{file_patter}'
        image_files = glob.glob(image_files, recursive=True)
        image_files.sort()
        return image_files
    def get_dir_ids(self, src_file):
        frame_num = os.path.basename(src_file)
        frame_num = frame_num.split('_rgb.png')[0]
        base_dir = os.path.dirname(src_file)
        return frame_num, base_dir
    def process_files(self, image_files):
        for src_file in image_files:
            frame_num, base_dir = self.get_dir_ids(src_file)
            use_depth, img = self.infer_image(src_file)
      
            if use_depth:
                tgt_file =  f"{base_dir}/{frame_num}_depth.npy"
                png_img = colorize(img[..., None])[...,0]
            else:
                tgt_file = f"{base_dir}/{frame_num}_normal.npy"
                png_img = colorize(img)
                img =np.transpose(img, (2, 0, 1))

            imageio.imwrite(tgt_file.replace('.npy', '.png'), png_img)
            np.save(tgt_file, img)
            print(f"saved file {tgt_file}")

        return
    def load_models(self):
        return
    def infer_image(self):
        return None
    def run(self):
        # return self.infer_image("/home/levin/workspace/nerf/mini-omnidata/assets/image.png", vis = True)
        # return self.infer_image("/home/levin/workspace/nerf/sdfstudio/data/sdfstudio-demo-data/replica-room0/000000_rgb.png", vis = True)

        image_dir = '/home/levin/workspace/nerf/sdfstudio/data/sdfstudio-demo-data/replica-room0'
        # base_dir = os.path.dirname(image_dir)

        file_patter = "*_rgb.png"
        image_files = self.collect_files(image_dir, file_patter)
        self.process_files(image_files)

        
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
        
        output = omnidata(image_path, down_factor=1.0).squeeze()

        #single frame visualization
        if vis:
            img = imageio.imread(image_path)
            if self.use_depth:
                output_vis = colorize(output[...,None])
            else:
                output_vis = colorize(output)

            _, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
            ax1.imshow(img)
            ax2.imshow(output)
            ax3.imshow(output_vis)
            plt.show()
        return self.use_depth, output
    

if __name__ == "__main__":   
    obj= App()
    obj.run()
