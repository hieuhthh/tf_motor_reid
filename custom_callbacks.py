import tensorflow as tf
import numpy as np
import sklearn
import scipy.spatial.distance
import matplotlib
from matplotlib import pyplot
from sklearn.metrics import average_precision_score

from dataset import *
from multiprocess_dataset import *
from utils import *
from model import *
from losses import *
from callbacks import *
from prepare import *

settings = get_settings()
globals().update(settings)

os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

set_memory_growth()

img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

use_cate_int = False
if label_mode == 'cate_int':
    use_cate_int = True

class CustomValidate(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CustomValidate, self).__init__()
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_model = None

        X_valid, Y_valid, cams_test = get_paths_ids_more_dataset(filename='test_files.txt')
        all_class = np.unique(list(Y_valid))

        X_A = []
        Y_A = []
        X_B = []
        Y_B = []

        for i in range(len(X_valid)):
            if cams_test[i] == 'A':
                X_A.append(X_valid[i])
                Y_A.append(Y_valid[i])
            else:
                X_B.append(X_valid[i])
                Y_B.append(Y_valid[i])

        print("len(X_A):", len(X_A))
        print("len(X_B):", len(X_B))

        valid_n_images_A = len(Y_A)
        self.valid_dataset_A = build_dataset_from_X_Y(X_A, None, all_class, None, "cate", img_size,
                                                BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

        valid_n_images_B = len(Y_B)
        self.valid_dataset_B = build_dataset_from_X_Y(X_B, None, all_class, None, "cate", img_size,
                                                BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

        self.metric = 'cosine'

        self.ids_camA = np.expand_dims(Y_A, 1)
        self.ids_camB = np.expand_dims(Y_B, 1)

    def on_train_begin(self, logs=None):
        self.best_mAP = 0
        self.best_rank_1 = 0

    def on_epoch_end(self, epoch, logs=None):
        emb_model = Model([self.model.input], [self.model.get_layer("bottleneck_bn").output])

        embs_A = emb_model.predict(self.valid_dataset_A, verbose=0)

        norm_embs_A = sklearn.preprocessing.normalize(embs_A)

        embs_B = emb_model.predict(self.valid_dataset_B, verbose=0)

        norm_embs_B = sklearn.preprocessing.normalize(embs_B)

        probe_features = norm_embs_A
        gallery_features = norm_embs_B

        cmc_curve = np.zeros(gallery_features.shape[0])
        ap_array = []
        all_dist = np.dot(probe_features, gallery_features.T)

        for idx, p_dist in enumerate(all_dist):
            rank_p = np.argsort(p_dist,axis=None)
            ranked_ids = self.ids_camB[rank_p]
            pos = np.where(ranked_ids == self.ids_camA[idx])
            cmc_curve[pos[0][0]]+=1
            y_true = np.zeros(np.shape(self.ids_camB))
            y_true[np.where(self.ids_camB == self.ids_camA[idx])] = 1
            y_pred = 1/(p_dist + 1e-8) # remove /0
            y_pred = np.squeeze(y_pred)
            ap = average_precision_score(y_true, y_pred)
            ap_array.append(ap)

        cmc_curve = np.cumsum(cmc_curve)/probe_features.shape[0]
        rank_1 = cmc_curve[0]
        mAP = np.mean(ap_array)
        
        print("\n" + "*"*10)
        print("rank 1:", rank_1)
        print("mAP:", mAP)
        print("*"*10)

        with open("valid_rank_1_mAP.txt", "a+") as f:
            f.write(f"Epoch: {epoch} ~ Rank 1: " + str(rank_1) + f" ~ mAP: " + str(mAP) + "\n")

        if rank_1 > self.best_rank_1:
            self.best_rank_1 = rank_1
            self.best_rank_1_epoch = epoch
            self.best_model = self.model

        if mAP > self.best_mAP:
            self.best_mAP = mAP
            self.best_mAP_epoch = epoch

    def on_train_end(self, logs=None):
        print("best_rank_1:", self.best_rank_1, " at epoch ", self.best_rank_1_epoch)
        print("best_mAP:", self.best_mAP, " at epoch ", self.best_mAP_epoch)
        print("Saving good_rank_1_model.h5")
        self.best_model.save("good_rank_1_model.h5")