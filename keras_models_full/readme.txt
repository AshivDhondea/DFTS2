readme.txt

Tue Aug 11 15:24

~/DFTS_TF2/keras_models/

This directory contained:

vgg16_model.h5
vgg19_model.h5
xception_model.h5
inceptionresnet_v2_model.h5
resnet18_model.h5
resnet50_v2_model.h5

To create the resnet18_model.h5 for ResNet-18,make use of this library:
https://github.com/qubvel/classification_models

For the other models, you can fetch them through tensorflow.keras by running the script
main_save_keras_models.py. It will download them with Imagenet weights.

If you are using Cedar, you can execute the script main_save_keras_models.sh 
(don't forget to change the email address to yours).

This .py script will also output an image of each keras_model architecture and will name it according to the following format:

main_save_keras_models_keras_model_architecture.png/pdf


