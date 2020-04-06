import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import imread
from shutil import copyfile
from sklearn.model_selection import train_test_split
from PIL import Image
import os

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
    #fig.suptitle("Target class: {}".format(_target_class), fontsize=20)
    plt.show()

def plot_hist(df, _target_class="healthy", _rows=3, _columns=4, _figsize=(20,9)):
    # most logic from https://discuss.pytorch.org/t/plot-a-histogram-for-multiple-images-full-dataset/67600/15
    df_filtered = df[df[_target_class] == 1]

    fig = plt.figure(figsize=_figsize)
    
    columns = _columns
    rows = _rows
    
    img_idx_list = np.random.randint(0, len(df_filtered), int(columns * rows))
    
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        img_name = df_filtered.iloc[img_idx_list[i-1]].image_id
        img = imread("images/{}.jpg".format(img_name))
        plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'Red', alpha = 0.5)
        plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
        plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
        plt.xlabel('intensity')
        plt.ylabel('count')
        plt.title(str(img_name))
        
    
    
    plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'], loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False)

    plt.tight_layout()
#     fig.suptitle("Target class: {}".format(_target_class), fontsize=20)
    plt.show()
    
    
def plot_agg_hist(df, _target_class="healthy"):
    
    # most logic from https://discuss.pytorch.org/t/plot-a-histogram-for-multiple-images-full-dataset/67600/15
    
    target_list = "./images/" + df[df[_target_class]==1].image_id.values + ".jpg"

    nb_bins = 256
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)

    for image in target_list:
        img = Image.open(image)
        x = np.array(img)
        x = x.transpose(2, 0, 1)
        hist_r = np.histogram(x[0], bins=nb_bins, range=[0, 255])
        hist_g = np.histogram(x[1], bins=nb_bins, range=[0, 255])
        hist_b = np.histogram(x[2], bins=nb_bins, range=[0, 255])
        count_r += hist_r[0]
        count_g += hist_g[0]
        count_b += hist_b[0]

    bins = hist_r[1]
    fig = plt.figure()
    plt.bar(bins[:-1], count_r, color='r', alpha=0.7)
    plt.bar(bins[:-1], count_g, color='g', alpha=0.7)
    plt.bar(bins[:-1], count_b, color='b', alpha=0.7)
    plt.xlabel('intensity')
    plt.ylabel('count')
    plt.title(_target_class)
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
    
    
def change_file_structure(df, src_dir="images", tar_dir="train_data", pred_list=["healthy", "multiple_diseases", "rust", "scab"], val_size=0.2):
    
    train_dir = os.path.join(tar_dir, "train")
    val_dir = os.path.join(tar_dir, "val")
    
    for dir_ in [tar_dir, train_dir, val_dir]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    
    for pred in pred_list:
        if not os.path.exists(os.path.join(train_dir, pred)):
            os.mkdir(os.path.join(train_dir, pred))

        if not os.path.exists(os.path.join(val_dir, pred)):
            os.mkdir(os.path.join(val_dir, pred)) 
            
    df.index = df.image_id
    del df["image_id"]
    df["pred"] = df[pred_list].idxmax(axis=1)

    preds = df.pred.values
    images = df.index.values
    
    img_train, img_val, pred_train, pred_val = train_test_split(images, preds, stratify=preds, test_size=0.2)
    
    for img_id, img_pred in zip(img_train, pred_train):
        src_path = os.path.join(src_dir, img_id+".jpg")
        tar_path = os.path.join(tar_dir, "train", img_pred, img_id+".jpg")
        copyfile(src_path, tar_path)
    
    for img_id, img_pred in zip(img_val, pred_val):
        src_path = os.path.join(src_dir, img_id+".jpg")
        tar_path = os.path.join(tar_dir, "val", img_pred, img_id+".jpg")
        copyfile(src_path, tar_path)