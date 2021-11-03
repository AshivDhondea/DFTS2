"""
visualize_demo_tiles.py

Draw grid on channels to indicate location of packets within them.
Scale up the images for nicer visualization.
"""
from PIL import Image
import numpy as np
import os,sys
import glob
import tensorflow as tf
# ---------------------------------------------------------------------------- #
sim_data_dir = 'simData'
dataset = 'smallTest'
deep_model = 'resnet18'
splitLayer = 'add_1'
demo_str = 'NoChannel' #'lp_0.3_Bl_2'#
img_str = ['original','corrupted','CALTeC','ALTeC','HaLRTC','SiLRTC','InpaintNS']

batch_size = 4
tensor_id = 0

rowsPerPacket = 8
num_channels = 64
channel_width = 56
num_pkts_per_channel = 7

scaling_factor = 5

tilesPerRow = 8
str_to_remove = '_'
str_replacement = ""
# ---------------------------------------------------------------------------- #
input_dir = os.path.join(sim_data_dir,dataset,deep_model,'demo',splitLayer+'_'+demo_str)
output_dir = os.path.join(sim_data_dir,dataset,deep_model,'demo_scaled',splitLayer+'_'+demo_str)
os.makedirs(output_dir,exist_ok=True)

batches = 3
batch_id = [batch_size,batch_size,1]
"""
for i_batch in range(batches):
    for item_id in range(batch_id[i_batch]):
        for i_str in range(0,7):
            for i_c in range(num_channels):
                original_str = img_str[i_str]+'_batch_'+str(i_batch)+'_item_'+str(item_id)+'_tensor_0_channel_'+str(i_c)+'_8Bits_8Bits_.png'
                channel_img_fname = original_str.replace(str_to_remove,str_replacement)
                img = Image.open(os.path.join(input_dir,original_str))
                im = np.array(img,dtype=np.uint8)
                imRGB = np.repeat(im[:, :, np.newaxis], 3, axis=2)
                newsize = (channel_width*scaling_factor,channel_width*scaling_factor)
                img_RGB = Image.fromarray(imRGB.astype(np.uint8))
                img_RGB_rescaled = img_RGB.resize(newsize,Image.BICUBIC)
                im_RGB_rescaled = np.array(img_RGB_rescaled)

                im_RGB_rescaled[:,0,0] = 0*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                im_RGB_rescaled[:,-1,0] = 0*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                im_RGB_rescaled[:,0,1] = 255*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                im_RGB_rescaled[:,-1,1] = 255*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                im_RGB_rescaled[:,0,2] = 0*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                im_RGB_rescaled[:,-1,2] = 0*np.ones([scaling_factor*channel_width],dtype=np.uint8)

                for i_x in range(num_pkts_per_channel):
                    dx = i_x*rowsPerPacket*scaling_factor
                    im_RGB_rescaled[dx,:,0] =0*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                    im_RGB_rescaled[dx,:,1] =255*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                    im_RGB_rescaled[dx,:,2] =0*np.ones([scaling_factor*channel_width],dtype=np.uint8)

                im_RGB_rescaled[-1,:,0] = 0*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                im_RGB_rescaled[-1,:,1] = 255*np.ones([scaling_factor*channel_width],dtype=np.uint8)
                im_RGB_rescaled[-1,:,2] = 0*np.ones([scaling_factor*channel_width],dtype=np.uint8)

                im_gridded = Image.fromarray(im_RGB_rescaled.astype(np.uint8))
                im_gridded.save(os.path.join(output_dir,channel_img_fname[:-4]+'.jpg'), "JPEG", optimize=True)
"""
# ---------------------------------------------------------------------------- #
def tensor_to_tiled(tensor_np,num_channel_rows,num_channel_columns,height_channel,width_channel):
    """ Function for changing tensor to tiled image
        :param A,B dummy variable
    """
    A = np.reshape(tensor_np,(num_channel_rows,num_channel_columns,height_channel,width_channel))
    B = np.swapaxes(A,1,2)
    tiled = np.reshape(B,(num_channel_rows*height_channel, num_channel_columns*width_channel))
    return tiled

def fn_simple_tensor_tiling(tensor_in,cr,cc):
    w,h,d = np.shape(tensor_in)
    temp2 = np.moveaxis(tensor_in,-1,0)
    X = tensor_to_tiled(temp2,cr,cc,w,h)
    tiled_tensor = np.repeat(X[:,:,np.newaxis],3,axis=2)
    return tiled_tensor

output_dir = os.path.join(sim_data_dir,dataset,deep_model,'demo_tiled',splitLayer+'_'+demo_str)
os.makedirs(output_dir,exist_ok=True)


for i_batch in range(batches):
    for item_id in range(batch_id[i_batch]):
        for i_str in range(0,1):
            tensor_in = np.zeros([channel_width,channel_width,num_channels],dtype=np.uint8)
            for i_c in range(num_channels):
                original_str = img_str[i_str]+'_batch_'+str(i_batch)+'_item_'+str(item_id)+'_tensor_0_channel_'+str(i_c)+'_8Bits_8Bits_.png'
                img = Image.open(os.path.join(input_dir,original_str))
                im = np.array(img,dtype=np.uint8)
                tensor_in[:,:,i_c] = im
            tiled_tensor = fn_simple_tensor_tiling(tensor_in,tilesPerRow,tilesPerRow)
            tf.keras.preprocessing.image.save_img(os.path.join(output_dir,img_str[i_str]+'_batch_'+str(i_batch)+'_item_'+str(item_id)+'_tiled.jpg'),tiled_tensor,scale=False,file_formt='jpeg',quality=100)

            tiled_tensor_gridded = np.copy(tiled_tensor)
            tiled_tensor_gridded[:,0,0] = 0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[:,-1,0] = 0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[:,0,1] = 255*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[:,-1,1] = 255*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[:,0,2] = 0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[:,-1,2] = 0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)

            for i_x in range(tilesPerRow):
                dx = i_x*channel_width
                tiled_tensor_gridded[dx,:,0] =0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
                tiled_tensor_gridded[dx,:,1] =255*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
                tiled_tensor_gridded[dx,:,2] =0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)

                dx = i_x*channel_width
                tiled_tensor_gridded[:,dx,0] =0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
                tiled_tensor_gridded[:,dx,1] =255*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
                tiled_tensor_gridded[:,dx,2] =0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)

            tiled_tensor_gridded[-1,:,0] = 0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[-1,:,1] = 255*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[-1,:,2] = 0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)

            tiled_tensor_gridded[:,-1,0] = 0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[:,-1,1] = 255*np.ones([tilesPerRow*channel_width],dtype=np.uint8)
            tiled_tensor_gridded[:,-1,2] = 0*np.ones([tilesPerRow*channel_width],dtype=np.uint8)

            tf.keras.preprocessing.image.save_img(os.path.join(output_dir,img_str[i_str]+'_batch_'+str(i_batch)+'_item_'+str(item_id)+'_tiled_gridded.jpg'),tiled_tensor_gridded,scale=False,file_formt='jpeg',quality=100)
