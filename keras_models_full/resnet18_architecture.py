"""
Verifying that the architecture for the resnet18 is fine.

"""
import tensorflow as tf
import os
import sys

file_name =  os.path.basename(sys.argv[0]);
script_name = file_name[:-3]
"""
model_path = 'vgg16_model.h5'
loaded_model = tf.keras.models.load_model(model_path)
loaded_model.summary()
tf.keras.utils.plot_model(loaded_model,to_file=os.path.join(script_name+'_vgg16_full.png'),show_shapes=True)

"""
model_path = 'resnet18_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

loaded_model.summary()

original_model_config = loaded_model.get_config()
original_model_config['name'] = 'resnet18' # Make sure the name of the model is resnet18.

customObjects = []
renamed_model = tf.keras.Model.from_config(original_model_config,custom_objects = customObjects)

def fn_set_weights(smaller_model,original_model):
    modelLayers = [i.name for i in original_model.layers]
    for l in smaller_model.layers:
        orig = l.name
        if orig in modelLayers:
            lWeights = original_model.get_layer(orig)
            l.set_weights(lWeights.get_weights())
    return smaller_model

# Make sure to set the weights in the model recreated from the configuration dictionary.
renamed_model = fn_set_weights(renamed_model,loaded_model)

with open(original_model_config['name']+'.txt','w') as fh:
    renamed_model.summary(print_fn = lambda x: fh.write(x + '\n'))

renamed_model.save('resnet18_model.h5')

renamed_model.summary()
tf.keras.utils.plot_model(renamed_model,to_file=os.path.join(script_name+'_resnet18_full.png'),show_shapes=True)
