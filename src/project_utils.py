import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import imread
from PIL import Image

def plot_examples(df, _target_class="healthy", _rows=3, _columns=4, _figsize=(20,9)):
    
    df_filtered = df[df[_target_class] == 1]

    fig = plt.figure(figsize=_figsize)
    
    columns = _columns
    rows = _rows
    
    img_idx_list = np.random.randint(0, len(df_filtered), int(columns * rows))
    
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        img_name = df_filtered.iloc[img_idx_list[i-1]].image_id
        img = imread("images/{}.jpg".format(img_name))
        plt.imshow(img)
        plt.axis('off')
        
    #plt.tight_layout()
    fig.suptitle("Target class: {}".format(_target_class), fontsize=20)
    plt.show()


def plot_patches(patches_list, _rows=10, _columns=10, _figsize=(15,15)):
    
    fig = plt.figure(figsize=_figsize)
    
    columns = _columns
    rows = _rows
    
    img_idx_list = np.random.randint(0, len(patches_list), int(columns * rows))
    
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        img = patches_list[img_idx_list[i-1]]
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        
    #plt.tight_layout()
    fig.suptitle("Patch examples", fontsize=20)
    plt.show()
    
def load_images(df, size=(224,224)):
    img_list = [np.asarray(Image.fromarray(imread("images/{}.jpg".format(img_name))).resize(size)) for img_name in df.image_id]
    return img_list
    
    