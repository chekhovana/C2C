# ,path,wsi,label
# 0,/project/GutIntelligenceLab/ys5hd/MSDS/images_512x512_non_resized/threshold_0.5/valid/Celiac/C14-70_01__4096_5632.jpg,C14-70_01,True

import pandas as pd
filename = 'data/original/annotations/train.csv'
image_folder = '/mnt/hdd_xfs3/old_data/home/chekhovana/projects/research/ubc_ocean/data/tiles/thumbnails/size_224_overlap_10'
df = pd.read_csv(filename)
for df_row in df.iterrows():
    pass

# df_result