"""
Downloading and saving Keras models as h5 files.

including the top, in order to do inference later on.

@Hans Dhondea

Created: 4 August 2020

Changelog:
Tuesday 11 August 2020: fixed file creation.

"""
import sys
import os
import tensorflow as tf
# --------------------------------------------------------------------------- #
print('TensorFlow version')
print(tf.__version__)
# --------------------------------------------------------------------------- #
file_name =  os.path.basename(sys.argv[0]);

if file_name[-3:] == '.py':
    script_name = file_name[:-3];
elif file_name[-3:] == '.ipynb':
    script_name = file_name[:-6];
else:
    script_name = 'main_xx';
# --------------------------------------------------------------------------- #
print('VGG16')
vgg16_model = tf.keras.applications.VGG16(weights='imagenet',include_top=True)
vgg16_model.summary()

tf.keras.utils.plot_model(vgg16_model,to_file=script_name+'_vgg16_model_architecture.png',show_shapes=True)
tf.keras.utils.plot_model(vgg16_model,to_file=script_name+'_vgg16_model_architecture.pdf',show_shapes=True)
vgg16_model.save("vgg16_model.h5")

print('VGG19')
vgg19_model = tf.keras.applications.VGG19(weights='imagenet',include_top=True)
vgg19_model.summary()

tf.keras.utils.plot_model(vgg19_model,to_file=script_name+'_vgg19_model_architecture.png',show_shapes=True)
tf.keras.utils.plot_model(vgg19_model,to_file=script_name+'_vgg19_model_architecture.pdf',show_shapes=True)
vgg19_model.save("vgg19_model.h5")

print('Xception')
xception_model = tf.keras.applications.Xception(weights='imagenet',include_top=True)
xception_model.summary()

tf.keras.utils.plot_model(xception_model,to_file=script_name+'_xception_model_architecture.png',show_shapes=True)
tf.keras.utils.plot_model(xception_model,to_file=script_name+'_xception_model_architecture.pdf',show_shapes=True)
xception_model.save("xception_model.h5")

print('ResNet50')
resnet50_model = tf.keras.applications.ResNet50(weights='imagenet',include_top=True)
resnet50_model.summary()

tf.keras.utils.plot_model(resnet50_model,to_file=script_name+'_resnet50_architecture.png',show_shapes=True)
tf.keras.utils.plot_model(resnet50_model,to_file=script_name+'_resnet50_architecture.pdf',show_shapes=True)
resnet50_model.save("resnet50_model.h5")

print('ResNet50v2')
resnet50_v2_model = tf.keras.applications.ResNet50V2(weights='imagenet',include_top=True)
resnet50_v2_model.summary()

tf.keras.utils.plot_model(resnet50_v2_model,to_file=script_name+'_resnet50_v2_model_architecture.png',show_shapes=True)
tf.keras.utils.plot_model(resnet50_v2_model,to_file=script_name+'_resnet50_v2_model_architecture.pdf',show_shapes=True)
resnet50_v2_model.save("resnet50_v2_model.h5")

print('InceptionResNetV2')
inceptionresnet_v2_model = tf.keras.applications.InceptionResNetV2(weights='imagenet',include_top=True)
inceptionresnet_v2_model.summary()

tf.keras.utils.plot_model(inceptionresnet_v2_model,to_file=script_name+'_inceptionresnet_v2_model_architecture.png',show_shapes=True)
tf.keras.utils.plot_model(inceptionresnet_v2_model,to_file=script_name+'_inceptionresnet_v2_model_architecture.pdf',show_shapes=True)
inceptionresnet_v2_model.save("inceptionresnet_v2_model.h5")

print('Models saved. Architectures saved.')
