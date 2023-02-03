import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from dataset import *
from multiprocess_dataset import *
from utils import *
from model import *
from losses import *
from callbacks import *
from prepare import *
from custom_callbacks import *

settings = get_settings()
globals().update(settings)

os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

set_memory_growth()

img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

use_cate_int = False
if label_mode == 'cate_int':
    use_cate_int = True

epochs = cycle_epoch * n_cycle
print('epochs:', epochs)

seedEverything(seed)
print('BATCH_SIZE:', BATCH_SIZE)
    
X_train, Y_train, cams_train = get_paths_ids_more_dataset(filename='train_files.txt')
X_train, Y_train = auto_split_data_from_X_Y(X_train, Y_train)

X_valid, Y_valid, cams_test = get_paths_ids_more_dataset(filename='test_files.txt')

Y_full = list(Y_train) + list(Y_valid)
all_class = np.unique(Y_full)

train_n_images = len(Y_train)
train_dataset = build_dataset_from_X_Y(X_train, Y_train, all_class, train_with_labels, label_mode, img_size,
                                       BATCH_SIZE, train_repeat, train_shuffle, train_augment, im_size_before_crop)

valid_n_images = len(Y_valid)
# valid_dataset = build_dataset_from_X_Y(X_valid, Y_valid, all_class, valid_with_labels, label_mode, img_size,
#                                        BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

n_labels = len(all_class)

print('n_labels', n_labels)
print('train_n_images', train_n_images)
print('valid_n_images', valid_n_images)

with open("note.txt", mode='w') as f:
    f.write("n_labels: " + str(n_labels) + "\n")
    f.write("train_n_images: " + str(train_n_images) + "\n")
    f.write("valid_n_images: " + str(valid_n_images) + "\n")

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

strategy = auto_select_accelerator()

with strategy.scope():
    model = create_model(base_name, input_shape, do_dim, atrous_dim, dilation_rates, final_dropout, 
                         have_emb_layer, emb_dim, n_labels, use_normdense, 
                         append_norm, use_cate_int)

    model.summary()

    emb_name = 'bottleneck_bn'

    if not append_norm:
        losses = {
            'cate_output' : ArcfaceLoss(from_logits=True, 
                                        label_smoothing=arcface_label_smoothing,
                                        margin1=arcface_margin1,
                                        margin2=arcface_margin2,
                                        margin3=arcface_margin3),
            emb_name : SupervisedContrastiveLoss(temperature=sup_con_temperature),
        }
    else:
        losses = {
            'cate_output' : AdaFaceLoss(from_logits=True, 
                                        batch_size=BATCH_SIZE,
                                        label_smoothing=arcface_label_smoothing,
                                        margin=arcface_margin2),
            emb_name : SupervisedContrastiveLoss(temperature=sup_con_temperature),
        }

    loss_weights = {
        'cate_output' : arc_face_weight,
        emb_name : sup_con_weight,
    }

    metrics = {
        'cate_output' : tf.keras.metrics.CategoricalAccuracy()
    }

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=losses,
              loss_weights=loss_weights,
              metrics=metrics)

if pretrain_full_model is not None:
    try:
        model.load_weights(pretrain_full_model)
        print('Loaded pretrain from', pretrain_full_model)
    except:
        print('Failed to load pretrain from', pretrain_full_model)

save_path = f'best_model_motor_reid_{base_name}_{im_size}_{emb_dim}_{n_labels}.h5'

callbacks = get_callbacks(monitor, mode, save_path, max_lr, min_lr, cycle_epoch, save_weights_only)
callbacks.append(CustomValidate())

his = model.fit(train_dataset, 
                steps_per_epoch = train_n_images//BATCH_SIZE,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks)

# metric = 'loss'
# visual_save_metric(his, metric)

# metric = 'categorical_accuracy'
# visual_save_metric(his, metric)




