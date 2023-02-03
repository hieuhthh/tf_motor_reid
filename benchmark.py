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

# model = tf.keras.models.load_model(, custom_objects={
#     'NormDense':NormDense,
#     'ArcfaceLoss':ArcfaceLoss,
#     'SupervisedContrastiveLoss':SupervisedContrastiveLoss
# })
# model.summary()

settings = get_settings()
globals().update(settings)

os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

set_memory_growth()

img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

use_cate_int = False
if label_mode == 'cate_int':
    use_cate_int = True

model_weight = "good_rank_1_model.h5"
# model_weight = "best_model_motor_reid_EfficientNetV2S_256_128_3827.h5"
n_labels = 3827

strategy = auto_select_accelerator()

with strategy.scope():
    model = create_model(base_name, input_shape, do_dim, atrous_dim, dilation_rates, final_dropout, 
                         have_emb_layer, emb_dim, n_labels, use_normdense, 
                         append_norm, use_cate_int)

    model.load_weights(model_weight)

    model.summary()

    model = Model([model.input], [model.get_layer("bottleneck_bn").output])

    model.summary()
    
    model.save("emb_model.h5")

# with strategy.scope():
#     model = tf.keras.models.load_model(model_weight)

#     model.summary()

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
valid_dataset_A = build_dataset_from_X_Y(X_A, None, all_class, None, "cate", img_size,
                                         BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)
embs_A = model.predict(valid_dataset_A)
# embs_A = sklearn.preprocessing.normalize(embs_A)

valid_n_images_B = len(Y_B)
valid_dataset_B = build_dataset_from_X_Y(X_B, None, all_class, None, "cate", img_size,
                                         BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)
embs_B = model.predict(valid_dataset_B)
# embs_B = sklearn.preprocessing.normalize(embs_B)

probe_features = embs_A
gallery_features = embs_B
metric = 'cosine'
ids_camA = np.expand_dims(Y_A, 1)
ids_camB = np.expand_dims(Y_B, 1)

cmc_curve = np.zeros(gallery_features.shape[0])
ap_array = []
all_dist = scipy.spatial.distance.cdist(probe_features, gallery_features, metric=metric)

# all_dist = np.dot(probe_features, gallery_features.T)
# all_dist = -all_dist

print(all_dist)
print(all_dist.shape)

for idx, p_dist in enumerate(all_dist):
    rank_p = np.argsort(p_dist,axis=None)
    ranked_ids = ids_camB[rank_p]
    pos = np.where(ranked_ids == ids_camA[idx])
    cmc_curve[pos[0][0]]+=1
    y_true = np.zeros(np.shape(ids_camB))
    y_true[np.where(ids_camB == ids_camA[idx])] = 1
    y_pred = 1/(p_dist + 1e-8) # remove /0
    y_pred = np.squeeze(y_pred)
    ap = average_precision_score(y_true, y_pred)
    ap_array.append(ap)

cmc_curve = np.cumsum(cmc_curve)/probe_features.shape[0]

name = 'cmc_curve'
curve_label = 'On Train Data'
curve_color = 'red' 

matplotlib.use('Agg')
pyplot.title('CMC Curve - R1: %.2f / mAP: %.2f'%(cmc_curve[0]*100, np.mean(ap_array)*100) )
pyplot.ylabel('Recognition Rate (%)')
pyplot.xlabel('Rank')
pyplot.plot(cmc_curve,label=curve_label, color=curve_color)
pyplot.legend(loc='upper left')
pyplot.savefig(name+'.png')
pyplot.close()

print("rank 1:", cmc_curve[0])
print("mAP:", np.mean(ap_array))