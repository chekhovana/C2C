# ,path,wsi,label
# 0,/project/GutIntelligenceLab/ys5hd/MSDS/images_512x512_non_resized/threshold_0.5/valid/Celiac/C14-70_01__4096_5632.jpg,C14-70_01,True
import os

import pandas as pd
filename = 'data/original/annotations/train.csv'
image_folder = ('/mnt/hdd_xfs3/old_data/home/chekhovana/projects/research/'
                'ubc_ocean/data/tiles/thumbnails/'
                'size_256_overlap_10_threshold_50')
df = pd.read_csv(filename)
data = dict(path=[], wsi=[], is_valid=[], label=[])
for i, df_row in df.iterrows():
    if df_row['is_tma'] == 1:
        continue
    image_id = df_row['image_id']
    label = df_row['label'] == 'HGSC'
    is_valid = int(df_row['fold'] == 5)
    folder = os.path.join(image_folder, f'{image_id}_thumbnail')
    filenames = os.listdir(folder)
    for fn in filenames:
        data['path'].append(os.path.join(folder, fn))
        data['wsi'].append(image_id)
        data['is_valid'].append(is_valid)
        data['label'].append(label)

df = pd.DataFrame(data)
df.to_csv('include/cluster2conquer/data/thumbnails.csv', index=False)
print(df.shape)
print(df.head())
