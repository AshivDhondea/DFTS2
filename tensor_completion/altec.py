"""
ALTeC method with a DFTS packetization and quantization scheme.

"""
import sys
sys.path.append('..')

import numpy as np
from models.packetModel import PacketModel as PacketModel
# ---------------------------------------------------------------------------- #
def fn_compute_altec_weights_pkts(pkt_model):
    """
    Compute ALTeC weights with a DFTS packet model.

    Based on ALTeC_train().
    Adapted from canonical implementation developed by Lior Bragilevsky (SFU
    Multimedia Lab).
    """
    num_examples = np.shape(pkt_model.packet_seq)[0]
    num_pkts = np.shape(pkt_model.packet_seq)[1]
    rowsPerPacket = np.shape(pkt_model.packet_seq)[2]
    channel_width = np.shape(pkt_model.packet_seq)[3]
    num_channels = np.shape(pkt_model.packet_seq)[4]

    train_example = PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(pkt_model.data_tensor))
    channel_weights = np.zeros([num_examples,num_channels+1,num_channels],dtype=np.float32)
    # Process each example in the packetized tensor.
    for i_example in range(num_examples):
        for i_c in range(num_channels):
            # Loop through all channels in tensor
            w_pkts = np.zeros([num_channels+1,num_pkts],dtype=np.float32)
            for i_pkt in range(num_pkts):
                # Initialize for each packet in a channel
                X_i_pkt = np.zeros([num_channels+1,channel_width*rowsPerPacket],dtype=np.float32)
                # fill in X_i_pkt
                i_k_channel = 0
                for i_k in range(num_channels):
                    if i_k == i_c:
                        # Co-located packet in the same channel is excluded.
                        #print(f'Processing packet {i_pkt} in channel {i_c}. Excluding channel {i_k} in colocated packets.')
                        continue
                    else:
                        #print(f'Colocated packet in other channel {i_k}. Saved in channel {i_k_channel}')
                        # Co-located packets in other channels.
                        X_i_pkt[i_k_channel,:] = np.reshape(train_example.packet_seq[i_example,i_pkt,:,:,i_k],(channel_width*rowsPerPacket))
                        i_k_channel += 1
                if i_pkt > 0:
                    # Spatial neighbor below in-channel.
                    X_i_pkt[-2,:] = np.reshape(train_example.packet_seq[i_example,i_pkt-1,:,:,i_c],(channel_width*rowsPerPacket,))
                if i_pkt < num_pkts - 1:
                    # Spatial neighbor above in-channel.
                    X_i_pkt[-1,:] = np.reshape(train_example.packet_seq[i_example,i_pkt+1,:,:,i_c],(channel_width*rowsPerPacket,))
                # Compute weights.
                pseudo = np.linalg.pinv(X_i_pkt).transpose()
                w_pkts[:,i_pkt] = pseudo.dot(np.reshape(train_example.packet_seq[i_example,i_pkt,:,:,i_c],(channel_width*rowsPerPacket)))
            avg_w_pkts = np.mean(w_pkts,axis=1)
            # Accumulate weights for a given channel.
            channel_weights[i_example,:,i_c] = avg_w_pkts
    return np.mean(channel_weights,axis=0) # Average weights by the number of examples in a batch.


def fn_complete_tensor_altec_pkts_star(corrupted_tensor,rowsPerPacket,altec_weights_pkt,loss_map):
    """
    Complete a corrupted tensor in a DFTS packet model using ALTeC weights.
    Use a loss map to figure out which packet needs to be repaired.
    Adapted from canonical implementation developed by Lior Bragilevsky (SFU
    Multimedia Lab).
    Compatible with weights produced by fn_compute_altec_weights_pkts.
    adapted tuesday jan 20. for star
    """
    repaired_pkt_model = PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(corrupted_tensor))
    num_examples = np.shape(repaired_pkt_model.packet_seq)[0]
    num_pkts = np.shape(repaired_pkt_model.packet_seq)[1]
    channel_width = np.shape(repaired_pkt_model.packet_seq)[3]
    num_channels = np.shape(repaired_pkt_model.packet_seq)[4]

    for i_example in range(num_examples):
        # Process all items in packetized tensor.
        item_loss_map = loss_map[i_example,:,:]

        for i_c in range(num_channels):
            # Loop through all channels in tensor
            for i_pkt in range(num_pkts):
                if item_loss_map[i_pkt,i_c] == False:
                    # If packet was lost, repair it.
                    # Initialize for each packet in a channel
                    X_i_pkt = np.zeros([num_channels+1,channel_width*rowsPerPacket],dtype=np.float32)
                    # fill in X_i_pkt
                    i_k_channel = 0
                    for i_k in range(num_channels):
                        if i_k == i_c:
                            # Co-located packet in the same channel is excluded.
                            continue
                        else:
                            # Co-located packets in other channels.
                            X_i_pkt[i_k_channel,:] = np.reshape(repaired_pkt_model.packet_seq[i_example,i_pkt,:,:,i_k],(channel_width*rowsPerPacket))
                            i_k_channel += 1
                    if i_pkt > 0:
                        X_i_pkt[-2,:] = np.reshape(repaired_pkt_model.packet_seq[i_example,i_pkt-1,:,:,i_c],(channel_width*rowsPerPacket,))
                    if i_pkt < num_pkts - 1:
                        X_i_pkt[-1,:] = np.reshape(repaired_pkt_model.packet_seq[i_example,i_pkt+1,:,:,i_c],(channel_width*rowsPerPacket,))

                    estimated_pkt_reshaped = np.dot(X_i_pkt.T,altec_weights_pkt[:,i_c])
                    estimated_pkt = np.reshape(estimated_pkt_reshaped,(rowsPerPacket,channel_width))
                    repaired_pkt_model.packet_seq[i_example,i_pkt,:,:,i_c] = estimated_pkt

    repaired_tensor = np.copy(repaired_pkt_model.data_tensor)
    return repaired_tensor
