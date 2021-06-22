"""
Content-Adaptive Linear Tensor Completion.

"""
import numpy as np
import copy
# ---------------------------------------------------------------------------- #
def find_nearest_index(array,value):
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array)):
        return idx-1
    else:
        return idx
# ---------------------------------------------------------------------------- #
def fn_caltec(lossMatrix,pkt_obj,item_index):
    # figure out the number of channels in the tensor, the dimensionality of a
    # channel, the number of packets in the channel.
    num_channels = np.shape(pkt_obj.packet_seq)[4]
    channel_width = np.shape(pkt_obj.packet_seq)[3]
    rowsPerPacket = np.shape(pkt_obj.packet_seq)[2]
    num_pkts_per_channel = np.shape(pkt_obj.packet_seq)[1]

    lost_map = lossMatrix[item_index,:,:]

    if np.size(lost_map) - np.count_nonzero(lost_map) == 0:
        print(f"No packets lost in item {item_index}.")
        return pkt_obj
    # ------------------------------------------------------------------------ #
    for i_c in range(num_channels):
        if np.all(lost_map[:,i_c] == False) == True:
            # The entire channel has been knocked out. cannot recover from that,
            # so ignore this damaged channel and continue.
            print(f'All packets were lost in this channel. Cannot repair channel {i_c}')
            continue
        # -------------------------------------------------------------------- #
        for i_pkt in range(num_pkts_per_channel):
            if lost_map[i_pkt,i_c] == False:
                #print(f"Repairing packet {i_pkt} in channel {i_c}")
                existing_colocated_pkts_list = np.sort(np.where(lost_map[i_pkt,:] == True)[0])
                # print('list of existing colocated packets in other channels')
                # print(existing_colocated_pkts_list)

                existing_pkts_in_channel_list = np.sort(np.where(lost_map[:,i_c] == True)[0])
                nearest_neighbor_in_channel_index = find_nearest_index(existing_pkts_in_channel_list,i_pkt)

                for test_pkt_id in range(len(existing_pkts_in_channel_list)):
                    nearest_neighbor_in_channel_idx = existing_pkts_in_channel_list[nearest_neighbor_in_channel_index - test_pkt_id]
                    existing_colocated_pkts_neighbor_list = np.where(lost_map[nearest_neighbor_in_channel_idx,:] == True)[0]

                    candidate_channels = np.intersect1d(existing_colocated_pkts_list,existing_colocated_pkts_neighbor_list)
                    if len(candidate_channels) != 0:
                        # print('found a candidate')
                        break
                if len(candidate_channels) == 0:
                    print(f'No candidate found. Cannot repair packet {i_pkt}')
                    continue

                candidate_channels = np.hstack(([i_c],candidate_channels))

                # print("The candidate channels for the correlation test are")
                # print(candidate_channels)

                nearest_neighbor_pkt = pkt_obj.packet_seq[item_index,nearest_neighbor_in_channel_idx,:,:,i_c]
                corrcoeff_matrix = np.zeros([len(candidate_channels),len(candidate_channels)])
                #for i_row in range(rowsPerPacket):
                #    corrcoeff_matrix += np.corrcoef([pkt_obj.packet_seq[item_index,nearest_neighbor_in_channel_idx,i_row,:,i] for i in candidate_channels])

                corrcoeff_matrix = np.corrcoef([np.reshape(pkt_obj.packet_seq[item_index,nearest_neighbor_in_channel_idx,:,:,i],(rowsPerPacket*channel_width)) for i in candidate_channels])
                row_corrcoef_below = corrcoeff_matrix[0,:]
                idx = np.argpartition(row_corrcoef_below,-2)[-2:]
                indices_below = idx[np.argsort((-row_corrcoef_below)[idx])]
                # print(f"The highest correlated channel is {candidate_channels[indices_below[1]]}")

                # select colocated packet which gives the second maximum value in corrcoeff_matrix.
                pkt_from_other_channel = pkt_obj.packet_seq[item_index,i_pkt,:,:,candidate_channels[indices_below[1]]]
                neighbor_out_channel = pkt_obj.packet_seq[item_index,nearest_neighbor_in_channel_idx,:,:,candidate_channels[indices_below[1]]]

                # reshape both neighbor packets into vectors and then run least squares.
                vec_in_channel = np.reshape(nearest_neighbor_pkt,(np.shape(nearest_neighbor_pkt)[0]*np.shape(nearest_neighbor_pkt)[1]))
                vec_out_channel = np.reshape(neighbor_out_channel,(np.shape(neighbor_out_channel)[0]*np.shape(neighbor_out_channel)[1]))

                lumi_transf = np.polyfit(vec_out_channel,vec_in_channel,1)
                lumi_transf_fn = np.poly1d(lumi_transf)

                vec_corrected = lumi_transf_fn(np.reshape(pkt_from_other_channel,(np.shape(pkt_from_other_channel)[0]*np.shape(pkt_from_other_channel)[1])))
                pkt_corrected_1 = np.reshape(vec_corrected,(np.shape(pkt_from_other_channel)))

                pkt_obj.packet_seq[item_index,i_pkt,:,:,i_c] = pkt_corrected_1
                #print(f'Packet {i_pkt} in channel {i_c} repaired.')
    return pkt_obj
