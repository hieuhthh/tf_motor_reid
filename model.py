from layers import *

# !pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q
from keras_cv_attention_models import efficientnet, convnext

def get_base_model(name, input_shape):
    if name == 'EfficientNetV2S':
        return efficientnet.EfficientNetV2S(num_classes=0, input_shape=input_shape, pretrained="imagenet21k")

    if name == 'EfficientNetV1B1':
        return efficientnet.EfficientNetV1B1(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B2':
        return efficientnet.EfficientNetV1B2(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B3':
        return efficientnet.EfficientNetV1B3(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B4':
        return efficientnet.EfficientNetV1B4(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B5':
        return efficientnet.EfficientNetV1B5(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B6':
        return efficientnet.EfficientNetV1B6(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B7':
        return efficientnet.EfficientNetV1B7(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'ConvNeXtTiny':
        return convnext.ConvNeXtTiny(num_classes=0, input_shape=input_shape, pretrained="imagenet21k-ft1k")

    if name == 'ResNet50':
        return tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')

    raise Exception("Cannot find this base model:", name)

def get_out_layers(name):
    if name == 'EfficientNetV2S':
        return [
                'stack_0_block1_output',
                'stack_1_block3_output',
                'stack_2_block3_output',
                'stack_4_block8_output',
                'post_swish'
            ]
    if name == 'ConvNeXtTiny':
        return [
            'stack2_downsample_ln',
            'stack3_downsample_ln',
            'stack4_downsample_ln',
            'stack4_block3_output'
        ]
    return None

def create_model(base_name, input_shape, do_dim, atrous_dim, dilation_rates, final_dropout, 
                 have_emb_layer, emb_dim, n_labels, use_normdense, 
                 append_norm, use_cate_int):
    """
    append_norm = True if use Adaface
    use_cate_int: if output have categorical output and int output
        EX: [[0, 0, 1, 0], 2]
    """
    base = get_base_model(base_name, input_shape)
    backbone_layer_names = get_out_layers(base_name)
    backbone_layers = [base.get_layer(layer_name).output for layer_name in backbone_layer_names]

    list_features = []
    for backbone_layer in backbone_layers:
        f = atrous_conv(backbone_layer, atrous_dim, dilation_rates)
        f = Concatenate()(f)
        f = self_attention(f, do_dim)
        
        skip = conv(backbone_layer, do_dim, 1)
        f = f + skip
        f = GlobalAveragePooling2D()(f)

        list_features.append(f)

    x = Concatenate()(list_features)

    x = Dense(do_dim, activation='swish')(x)
    x = Dropout(final_dropout)(x)
    x = Dense(do_dim)(x)
    x = Dropout(final_dropout)(x)

    if have_emb_layer:
        x = Dense(emb_dim, use_bias=False, name='bottleneck')(x)
        x = BatchNormalization(name='bottleneck_bn')(x)

    model = Model(base.input, x)
    
    if use_normdense:
        cate_output = NormDense(n_labels, name='cate_output', append_norm=append_norm)(x)
    else:
        cate_output = Dense(n_labels, name='cate_output')(x)

    if not use_cate_int:
        model = Model([base.input], [cate_output])
    else:
        model = Model([base.input], [cate_output, x])
    
    return model

if __name__ == "__main__":
    import os
    from utils import *
    from multiprocess_dataset import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)

    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)

    use_cate_int = False
    if label_mode == 'cate_int':
        use_cate_int = True

    n_labels = 100

    model = create_model(base_name, input_shape, do_dim, atrous_dim, dilation_rates, final_dropout, 
                         have_emb_layer, emb_dim, n_labels, use_normdense, 
                         append_norm, use_cate_int)
    model.summary()