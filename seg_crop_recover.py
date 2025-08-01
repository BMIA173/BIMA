import SimpleITK as sitk
import numpy as np
import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from dice_loss import DC_and_CE_loss
import pickle

def save_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

# load_nii_as_narray
def load_nii_as_narray(filename):
    img_obj = sitk.ReadImage(filename)
    data_array = sitk.GetArrayFromImage(img_obj)
    spacing = img_obj.GetSpacing()
    print(filename,spacing)
    return data_array,spacing


# save_arry_as_nii
def save_arry_as_nii(data,image_name,pixel_spacing=[1, 1, 1]):
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(pixel_spacing)
    sitk.WriteImage(img, image_name)


def save_array_as_volume(data, filename, transpose=True, pixel_spacing=[1,1,1]):
    """
    save a numpy array as nifty image_back
    inputs:
        data: a numpy array with shape [Channel, Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    if transpose:
        data = data.transpose(2, 1, 0)
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(pixel_spacing)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(img)


# Extract sub-organ and cropping according the coarse segmentation
'''
def crop(coarseg_root,image_root,label_root,coarseg_save,image_save,label_save,label_wanted,pading,
         crop_coarsegs,crop_images,crop_labels):
    dict_crop={}
    anchor = {}
    for file in os.listdir(coarseg_root):  # the file end with  "_0006.nii.gz"
        sub_coarseg = os.path.join(coarseg_root, file)
        sub_coarseg = load_nii_as_narray(sub_coarseg)

        sub_coarseg_size = sub_coarseg.shape
        crop_data = []
        anchor_data = []
        # Extract  label  wanted
        for i in range(len(label_wanted)):
            if file =='img_014.nii.gz':
                if i==5:
                    print(1)
            ncoarseg = np.zeros_like(sub_coarseg)
            ncoarseg[np.where(sub_coarseg == label_wanted[i])] = 1
            sub_coarseg_2 = ncoarseg
            # get the coordinate
            nonzeropoint = np.asarray(np.nonzero(sub_coarseg_2))

            maxpoint = np.max(nonzeropoint, 1).tolist()

            minpoint = np.min(nonzeropoint, 1).tolist()

            for ii in range(len(pading)):
                maxpoint[ii] = min(maxpoint[ii] + pading[ii], sub_coarseg_size[ii])  # sub_coarseg_size
                minpoint[ii] = max(minpoint[ii] - pading[ii], 0)
            crop_point = minpoint + maxpoint
            crop_point.insert(0, label_wanted[i])
            crop_data.append(crop_point)
            anchor_point = [maxpoint[0]-minpoint[0],
                            maxpoint[1]-minpoint[1],
                            maxpoint[2]-minpoint[2]]
            anchor_point.insert(0,label_wanted[i])
            anchor_data.append(anchor_point)
        dict_crop[file] = crop_data
        anchor[file] = anchor_data
        # print(1)

        for i in range(len(label_wanted)):
            # cropping  coarsegTr
            if crop_coarsegs:
                sub_coarseg_3 = sub_coarseg_2[minpoint[0]:maxpoint[0],
                                minpoint[1]:maxpoint[1],
                                minpoint[2]:maxpoint[2]]
                print('coarsegTr:', sub_coarseg_3.shape)
                coarseg_name = os.path.join(coarseg_save, str(label_wanted[i]), file)
                save_array_as_volume(sub_coarseg_3, coarseg_name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
                print(coarseg_name)

            # croppting label
            if crop_labels:
                file_c = file
                sub_label = os.path.join(label_root, file_c)
                sub_label = load_nii_as_narray(sub_label)

                nsub_label = np.zeros_like(sub_label)
                nsub_label[np.where(sub_label == label_wanted[i])] = 1

                sub_label_2 = nsub_label[minpoint[0]:maxpoint[0],
                              minpoint[1]:maxpoint[1],
                              minpoint[2]:maxpoint[2]]
                print('label:', sub_label_2.shape)
                label_name = os.path.join(label_save, str(label_wanted[i]), file_c)
                # label_name = label_save
                save_array_as_volume(sub_label_2, label_name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
                # save_arry_as_nii(sub_label_2, label_name ,pixel_spacing=[1.2, 1.2, 1.2])
                print(label_name)

            # crop ct t1 t2   data set
            if crop_images:
                ct = file
                t1 = file
                t2 = file
                # modes = [ct,t1,t2]
                modes = [ct]
                for mode in modes:
                    sub_image = os.path.join(image_root, mode)
                    sub_image = load_nii_as_narray(sub_image)
                    sub_image_1 = sub_image[minpoint[0]:maxpoint[0],
                                  minpoint[1]:maxpoint[1],
                                  minpoint[2]:maxpoint[2]]
                    print('image_back:', sub_image_1.shape)
                    image_name = os.path.join(image_save, str(label_wanted[i]), mode)
                    save_array_as_volume(sub_image_1, image_name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
                    print(image_name)

    return dict_crop
'''

# 所有数据相同
def crop_data_same(coarseg_root, image_root, label_root, coarseg_save, image_save, label_save, label_wanted, pading,data_int,
              crop_coarsegs, crop_images, crop_labels,crop_sum,single_label):
    dict_crop = {}
    anchor = {}
    position = {}
    crop_start_end_dict = {}
    data_shape = []
    # 找到裁剪图层起始值和裁剪区域
    for file in os.listdir(coarseg_root):
        print(file)# the file end with  "_0006.nii.gz"
        sub_coarseg = os.path.join(coarseg_root, file)
        sub_coarseg,sub_space = load_nii_as_narray(sub_coarseg)

        sub_coarseg_size = sub_coarseg.shape
        crop_data = []
        anchor_data = []
        position_data = []
        data_shape.append(np.array(sub_coarseg_size))
        # Extract  label  wanted
        for i in range(len(label_wanted)):

            ncoarseg = np.zeros_like(sub_coarseg)
            ncoarseg[np.where(sub_coarseg == label_wanted[i])] = 1
            sub_coarseg_2 = ncoarseg.copy()
            # get the coordinate
            nonzeropoint = np.asarray(np.nonzero(sub_coarseg_2))
            # if file == 'img_014.nii.gz':
            #     if i == 5:
            #         print(1)
            if nonzeropoint.size == 0:
                position_data.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
                crop_data.append([0,0,0,0,0,0,0])
                anchor_data.append([0,0,0,0])
                continue

            end_point = np.max(nonzeropoint, 1).tolist()
            start_point = np.min(nonzeropoint, 1).tolist()
            maxpoint = np.max(nonzeropoint, 1).tolist()
            minpoint = np.min(nonzeropoint, 1).tolist()


            for ii in range(len(pading)):
                end_point[ii] = min(maxpoint[ii], sub_coarseg_size[ii])
                start_point[ii] = max(minpoint[ii], 0)
                center_point = [int((start_point[0]+end_point[0])/2),int((start_point[1]+end_point[1])/2),int((start_point[2]+end_point[2])/2)]
                maxpoint[ii] = min(maxpoint[ii] + pading[ii], sub_coarseg_size[ii])  # sub_coarseg_size
                minpoint[ii] = max(minpoint[ii] - pading[ii], 0)
            # 加一是因为栽树问题
            posi_point = [(end_point[0]-start_point[0])+1,
                          (end_point[1]-start_point[1])+1,
                          (end_point[2]-start_point[2])+1]
            position_point = start_point + end_point + center_point + posi_point
            crop_point = minpoint + maxpoint
            position_point.insert(0, label_wanted[i])
            crop_point.insert(0, label_wanted[i])
            position_data.append(position_point)
            crop_data.append(crop_point)
            anchor_point = [(maxpoint[0] - minpoint[0])+1,
                            (maxpoint[1] - minpoint[1])+1,
                            (maxpoint[2] - minpoint[2])+1]
            anchor_data.append(anchor_point)

        dict_crop[file] = crop_data
        anchor[file] = anchor_data
        position[file] = position_data
        # if file == 'img_014.nii.gz':
        #     break
        # print(1)
    # 找到平均值
    position_crop_data = []
    dict_crop_data = []
    anchor_data = []
    for file in os.listdir(coarseg_root):
        position_crop_data_f = np.array(position[file])
        dict_crop_data_f = np.array(dict_crop[file])
        position_crop_data.append(position_crop_data_f)
        dict_crop_data.append(dict_crop_data_f)
        anchor_data.append(anchor_f)
        # if file == 'img_014.nii.gz':
        #     break
    # 求平均起始位置和边界框
    position_crop_data_np = np.array(position_crop_data)
    dict_crop_data_np = np.array(dict_crop_data)
    anchor_data_np = np.array(anchor_data).reshape((len(anchor),4*len(label_wanted)))
    position_mean = []
    dict_mean = []
    anchor_mean = []
    for i in range(len(label_wanted)):
        if i==5:
            print(1)
        posi_i = position_crop_data_np[:,i,:]
        dict_i = dict_crop_data_np[:,i,:]
        label = dict_crop_data_np[:,i,:][:,0]
        index_zero = np.argwhere(label==0)
        if index_zero.size != 0:
            posi_i = np.delete(posi_i,index_zero,axis=0)
            dict_i = np.delete(dict_i,index_zero,axis=0)
        # if dict_i[i][0] == 0:
        #     continue
        posi_mean = np.mean(posi_i,axis=0)
        label_mean = np.mean(dict_i,axis=0)
        posi_mean_f = posi_mean.astype(np.int32)
        label_mean_f = label_mean.astype(np.int32)
        dict_mean.append(label_mean.astype(np.int32))
        anchor_mean_f = [i+1,label_mean_f[4]-label_mean_f[1],
                         label_mean_f[5]-label_mean_f[2],
                         label_mean_f[6]-label_mean_f[3]]
        anchor_mean.append(anchor_mean_f)
    # 如果没有分割结果，则指定平均像素起始值和裁剪框
    for file in os.listdir(coarseg_root):
        for i in range(len(label_wanted)):
            crop_f = dict_crop[file][i]
            if crop_f == [0,0,0,0,0,0,0]:
                position[file][i] = list(position_mean[i])
                dict_crop[file][i] = list(dict_mean[i])
                anchor[file][i] = anchor_mean[i]
        # if file == 'img_014.nii.gz':
        #     break
    # print(1)
    # for file in os.listdir(coarseg_root):
    #     for i in range(len(label_wanted)):
    #         if
    # 找到最大的anchor
    # anchor_max = []
    # anchor_data = []
    # for file in os.listdir(coarseg_root):
    #     anchor_f = np.array(anchor[file])
    #     anchor_data.append(anchor_f)
    # anchor_data_np = np.array(anchor_data).reshape((len(anchor),4*len(label_wanted)))
    # anchor_label_t中是所有数据anchor的最大值
    anchor_label_t = []
    for i in range(len(label_wanted)):
        max_z = np.max(anchor_data_np[:,(i*4)+1])
        max_x = np.max(anchor_data_np[:, (i * 4) + 2])
        max_y = np.max(anchor_data_np[:, (i * 4) + 3])
        anchor_label_t.append([max_z,max_x,max_y])
    anchor_label = []
    for i in range(len(anchor_mean)):
        anchor_z = data_int[0][data_int[0]>=anchor_mean[i][1]][0]
        anchor_x_t = data_int[1][data_int[1]>=anchor_mean[i][2]][0]
        anchor_y_t = data_int[2][data_int[2] >= anchor_mean[i][3]][0]
        anchor_x = max(anchor_x_t,anchor_y_t)
        anchor_y = max(anchor_x_t,anchor_y_t)
        # anchor_label.append([anchor_z,anchor_x,anchor_y])
        anchor_label.append([anchor_z, 256, 256])
    pkl_name = os.path.join(image_out,'position.pkl')
    save_pickle(position,pkl_name)
    # anchor_label = [[16, 160, 160]]

    # 最终的裁剪尺寸确定
    # data_shape_np = np.array(data_shape)
    # min_shape_z = np.min(data_shape_np[:,0])
    # min_shape_x = np.min(data_shape_np[:,1])
    # min_shape_y = np.min(data_shape_np[:,2])
    print(1)

    # 最终裁剪
    for file in os.listdir(coarseg_root):

        # if file == 'img_125.nii.gz':
        #     print(1)
        crop_s_e_dict = {}
        for i in range(len(label_wanted)):
            # cropping  coarsegTr
            crop_start_end = []
            if crop_coarsegs:
                sub_coarseg_origin = os.path.join(coarseg_root, file)
                sub_coarseg_origin,sub_spacing = load_nii_as_narray(sub_coarseg_origin)

                # 可能存在边界超出现象
                # z_start0 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1), 0)
                # x_start0 = max(int(position[file][i][8] - (anchor_label[i][1] / 2) + 1), 0)
                # y_start0 = max(int(position[file][i][9] - (anchor_label[i][2] / 2) + 1), 0)
                # z_end0 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1), sub_coarseg_origin.shape[0])
                # x_end0 = min(int(position[file][i][8]+(anchor_label[i][1]/2)+1),sub_coarseg_origin.shape[1])
                # y_end0 = min(int(position[file][i][9]+(anchor_label[i][2]/2)+1),sub_coarseg_origin.shape[2])
                # crop_s_e0 = [z_start0,x_start0,y_start0,z_end0,x_end0,y_end0]
                #
                # z_start1 = max(int(position[file][i][7] - (anchor_label[i][0] / 4) + 1), 0)
                # x_start1 = max(int(position[file][i][8] - (anchor_label[i][1] / 4) + 1), 0)
                # y_start1 = max(int(position[file][i][9] - (anchor_label[i][2] / 4) + 1), 0)
                # z_end1 = min(int(position[file][i][7] + (anchor_label[i][0]*3 / 4) + 1), sub_coarseg_origin.shape[0])
                # x_end1 = min(int(position[file][i][8] + (anchor_label[i][1]*3 / 4) + 1),sub_coarseg_origin.shape[1])
                # y_end1 = min(int(position[file][i][9] + (anchor_label[i][2]*3 / 4) + 1),sub_coarseg_origin.shape[2])
                # crop_s_e1 = [z_start1,x_start1,y_start1,z_end1,x_end1,y_end1]
                #
                # z_start2 = max(int(position[file][i][7] - (anchor_label[i][0]*3 / 4) + 1), 0)
                # x_start2 = max(int(position[file][i][8] - (anchor_label[i][1]*3 / 4) + 1), 0)
                # y_start2 = max(int(position[file][i][9] - (anchor_label[i][2]*3 / 4) + 1), 0)
                # z_end2 = min(int(position[file][i][7] + (anchor_label[i][0] / 4) + 1), sub_coarseg_origin.shape[0])
                # x_end2 = min(int(position[file][i][8] + (anchor_label[i][1] / 4) + 1),sub_coarseg_origin.shape[1])
                # y_end2 = min(int(position[file][i][9] + (anchor_label[i][2] / 4) + 1),sub_coarseg_origin.shape[2])
                # crop_s_e2 = [z_start2,x_start2,y_start2,z_end2,x_end2,y_end2]
                #
                # z_start3 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1), 0)
                # x_start3 = max(int(position[file][i][8] - (anchor_label[i][1] / 4) + 1), 0)
                # y_start3 = max(int(position[file][i][9] - (anchor_label[i][2] / 4) + 1), 0)
                # z_end3 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1), sub_coarseg_origin.shape[0])
                # x_end3 = min(int(position[file][i][8] + (anchor_label[i][1]*3 / 4) + 1),sub_coarseg_origin.shape[1])
                # y_end3 = min(int(position[file][i][9] + (anchor_label[i][2]*3 / 4) + 1),sub_coarseg_origin.shape[2])
                # crop_s_e3 = [z_start3,x_start3,y_start3,z_end3,x_end3,y_end3]
                #
                # z_start4 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1), 0)
                # x_start4 = max(int(position[file][i][8] - (anchor_label[i][1]*3 / 4) + 1), 0)
                # y_start4 = max(int(position[file][i][9] - (anchor_label[i][2]*3 / 4) + 1), 0)
                # z_end4 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1), sub_coarseg_origin.shape[0])
                # x_end4 = min(int(position[file][i][8] + (anchor_label[i][1] / 4) + 1),sub_coarseg_origin.shape[1])
                # y_end4 = min(int(position[file][i][9] + (anchor_label[i][2] / 4) + 1),sub_coarseg_origin.shape[2])
                # crop_s_e4 = [z_start4,x_start4,y_start4,z_end4,x_end4,y_end4]
                # z_start0 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1), 0)
                # x_start0 = max(int(position[file][i][8] - (anchor_label[i][1] / 2) + 1), 0)
                # y_start0 = max(int(position[file][i][9] - (anchor_label[i][2] / 2) + 1), 0)
                # z_end0 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1), sub_coarseg_origin.shape[0])
                # x_end0 = min(int(position[file][i][8]+(anchor_label[i][1]/2)+1),sub_coarseg_origin.shape[1])
                # y_end0 = min(int(position[file][i][9]+(anchor_label[i][2]/2)+1),sub_coarseg_origin.shape[2])
                # crop_s_e0 = [z_start0,x_start0,y_start0,z_end0,x_end0,y_end0]

                z_start0 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1), 0)
                x_start0 = max(int(position[file][i][8] - (anchor_label[i][1] / 2) + 1), 0)
                y_start0 = max(int(position[file][i][9] - (anchor_label[i][2] / 2) + 1), 0)
                z_end0 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1), sub_coarseg_origin.shape[0])
                x_end0 = min(int(position[file][i][8]+(anchor_label[i][1]/2)+1),sub_coarseg_origin.shape[1])
                y_end0 = min(int(position[file][i][9]+(anchor_label[i][2]/2)+1),sub_coarseg_origin.shape[2])
                crop_s_e0 = [z_start0,x_start0,y_start0,z_end0,x_end0,y_end0]
                # crop_s_e0 = [z_start1, x_start1, y_start1, z_end1, x_end1, y_end1]
                crop_start_end.append(crop_s_e0)
                if crop_sum != 1:
                    z_start1 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1+5), 0)
                    x_start1 = max(int(position[file][i][8] - (anchor_label[i][1] / 2) + 1+10), 0)
                    y_start1 = max(int(position[file][i][9] - (anchor_label[i][2] / 2) + 1+10), 0)
                    z_end1 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1+5), sub_coarseg_origin.shape[0])
                    x_end1 = min(int(position[file][i][8] + (anchor_label[i][1] / 2) + 1+10),sub_coarseg_origin.shape[1])
                    y_end1 = min(int(position[file][i][9] + (anchor_label[i][2] / 2) + 1+10),sub_coarseg_origin.shape[2])
                    crop_s_e1 = [z_start1,x_start1,y_start1,z_end1,x_end1,y_end1]

                    z_start2 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1-5), 0)
                    x_start2 = max(int(position[file][i][8] - (anchor_label[i][1] / 2) + 1-10), 0)
                    y_start2 = max(int(position[file][i][9] - (anchor_label[i][2] / 2) + 1-10), 0)
                    z_end2 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1-5), sub_coarseg_origin.shape[0])
                    x_end2 = min(int(position[file][i][8] + (anchor_label[i][1] / 2) + 1-10),sub_coarseg_origin.shape[1])
                    y_end2 = min(int(position[file][i][9] + (anchor_label[i][2] / 2) + 1-10),sub_coarseg_origin.shape[2])
                    crop_s_e2 = [z_start2,x_start2,y_start2,z_end2,x_end2,y_end2]

                    z_start3 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1), 0)
                    x_start3 = max(int(position[file][i][8] - (anchor_label[i][1] / 2) + 1+10), 0)
                    y_start3 = max(int(position[file][i][9] - (anchor_label[i][2] / 2) + 1+10), 0)
                    z_end3 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1), sub_coarseg_origin.shape[0])
                    x_end3 = min(int(position[file][i][8] + (anchor_label[i][1] / 2) + 1+10),sub_coarseg_origin.shape[1])
                    y_end3 = min(int(position[file][i][9] + (anchor_label[i][2] / 2) + 1+10),sub_coarseg_origin.shape[2])
                    crop_s_e3 = [z_start3,x_start3,y_start3,z_end3,x_end3,y_end3]

                    z_start4 = max(int(position[file][i][7] - (anchor_label[i][0] / 2) + 1), 0)
                    x_start4 = max(int(position[file][i][8] - (anchor_label[i][1] / 2) + 1-10), 0)
                    y_start4 = max(int(position[file][i][9] - (anchor_label[i][2] / 2) + 1-10), 0)
                    z_end4 = min(int(position[file][i][7] + (anchor_label[i][0] / 2) + 1), sub_coarseg_origin.shape[0])
                    x_end4 = min(int(position[file][i][8] + (anchor_label[i][1] / 2) + 1-10),sub_coarseg_origin.shape[1])
                    y_end4 = min(int(position[file][i][9] + (anchor_label[i][2] / 2) + 1-10),sub_coarseg_origin.shape[2])
                    crop_s_e4 = [z_start4,x_start4,y_start4,z_end4,x_end4,y_end4]

                    crop_start_end.append(crop_s_e1)
                    crop_start_end.append(crop_s_e2)
                    crop_start_end.append(crop_s_e3)
                    crop_start_end.append(crop_s_e4)
                for j in range(len(crop_start_end)):
                    for k in range(len(crop_start_end[j])):
                        if crop_start_end[j][k] == 0:
                            crop_start_end[j][k+3] = anchor_label[i][k]
                        if crop_start_end[j][k] == sub_coarseg_origin.shape[k-3]:
                            crop_start_end[j][k-3] = sub_coarseg_origin.shape[k-3]-anchor_label[i][k-3]
                            if crop_start_end[j][k-3] < 0:
                                crop_start_end[j][k - 3] = 0

                crop_s_e_dict[i] = crop_start_end
                sub_coarseg_origin_single = np.zeros_like(sub_coarseg_origin)
                sub_coarseg_origin_single[np.where(sub_coarseg_origin == label_wanted[i])] = 1
                # sub_coarseg_3 = sub_coarseg_origin_single[dict_crop[file][i][1]:dict_crop[file][i][1] + anchor_label[i][0],
                #                 dict_crop[file][i][2]:dict_crop[file][i][2] + anchor_label[i][1],
                #                 dict_crop[file][i][3]:dict_crop[file][i][3] + anchor_label[i][2]]
                # sub_coarseg_3 = sub_coarseg_origin_single[dict_crop[file][i][1]:dict_crop[file][i][1] + anchor_mean[i][1],
                #                 dict_crop[file][i][2]:dict_crop[file][i][2] + anchor_mean[i][2],
                #                 dict_crop[file][i][3]:dict_crop[file][i][3] + anchor_mean[i][3]]
                # sub_coarseg_3 = sub_coarseg_origin_single[position[file][i][1]:(position[file][i][1]+position[file][i][10]),
                #                 position[file][i][2]:(position[file][i][2]+position[file][i][11]),
                #                 position[file][i][3]:(position[file][i][3]+position[file][i][13])]
                # sub_coarseg_0 = sub_coarseg_origin_single[
                #                 int(position[file][i][7]-(anchor_label[i][0]/2)+1):int(position[file][i][7]+(anchor_label[i][0]/2)+1),
                #                 int(position[file][i][8]-(anchor_label[i][1]/2)+1):int(position[file][i][8]+(anchor_label[i][1]/2)+1),
                #                 int(position[file][i][9]-(anchor_label[i][2]/2)+1):int(position[file][i][9]+(anchor_label[i][2]/2)+1)]

                sub_coarseg_0 = sub_coarseg_origin_single[
                                crop_start_end[0][0]:crop_start_end[0][3],
                                crop_start_end[0][1]:crop_start_end[0][4],
                                crop_start_end[0][2]:crop_start_end[0][5]]
                # sub_coarseg_0 = sub_coarseg_0[:, ::-1, :]
                print('coarsegTr:', sub_coarseg_0.shape)
                file_name0 = file.split('.')[0] + '_00.nii.gz'
                coarseg_name0 = os.path.join(coarseg_save, str(label_wanted[i]), file_name0)
                save_array_as_volume(sub_coarseg_0, coarseg_name0, transpose=False,pixel_spacing=sub_spacing)
                print(coarseg_name0)
                if crop_sum!=1:
                    sub_coarseg_1 = sub_coarseg_origin_single[
                                    crop_start_end[1][0]:crop_start_end[1][3],
                                    crop_start_end[1][1]:crop_start_end[1][4],
                                    crop_start_end[1][2]:crop_start_end[1][5]]
                    sub_coarseg_2 = sub_coarseg_origin_single[
                                    crop_start_end[2][0]:crop_start_end[2][3],
                                    crop_start_end[2][1]:crop_start_end[2][4],
                                    crop_start_end[2][2]:crop_start_end[2][5]]
                    sub_coarseg_3 = sub_coarseg_origin_single[
                                    crop_start_end[3][0]:crop_start_end[3][3],
                                    crop_start_end[3][1]:crop_start_end[3][4],
                                    crop_start_end[3][2]:crop_start_end[3][5]]
                    sub_coarseg_4 = sub_coarseg_origin_single[
                                    crop_start_end[4][0]:crop_start_end[4][3],
                                    crop_start_end[4][1]:crop_start_end[4][4],
                                    crop_start_end[4][2]:crop_start_end[4][5]]
                    assert (sub_coarseg_0.shape==sub_coarseg_1.shape)and(sub_coarseg_0.shape==sub_coarseg_2.shape)and(
                            sub_coarseg_0.shape == sub_coarseg_3.shape)and(sub_coarseg_0.shape==sub_coarseg_4.shape)

                    file_name1 = file.split('.')[0] + '_01.nii.gz'
                    file_name2 = file.split('.')[0] + '_02.nii.gz'
                    file_name3 = file.split('.')[0] + '_03.nii.gz'
                    file_name4 = file.split('.')[0] + '_04.nii.gz'

                    coarseg_name1 = os.path.join(coarseg_save, str(label_wanted[i]), file_name1)
                    coarseg_name2 = os.path.join(coarseg_save, str(label_wanted[i]), file_name2)
                    coarseg_name3 = os.path.join(coarseg_save, str(label_wanted[i]), file_name3)
                    coarseg_name4 = os.path.join(coarseg_save, str(label_wanted[i]), file_name4)

                    save_array_as_volume(sub_coarseg_1, coarseg_name1, transpose=False,pixel_spacing=sub_spacing)
                    save_array_as_volume(sub_coarseg_2, coarseg_name2, transpose=False,pixel_spacing=sub_spacing)
                    save_array_as_volume(sub_coarseg_3, coarseg_name3, transpose=False,pixel_spacing=sub_spacing)
                    save_array_as_volume(sub_coarseg_4, coarseg_name4, transpose=False,pixel_spacing=sub_spacing)
            print(1)

            # croppting label
            if crop_labels:
                file_c = file
                sub_label = os.path.join(label_root, file_c)
                sub_label,label_spacing = load_nii_as_narray(sub_label)

                if single_label:
                    nsub_label = np.zeros_like(sub_label)
                    nsub_label[np.where(sub_label == label_wanted[i])] = 1
                # 前景裁剪，保留标签值
                else:
                    nsub_label = sub_label

                # sub_label_2 = nsub_label[dict_crop[file][i][1]:dict_crop[file][i][1] + anchor_label[i][0],
                #               dict_crop[file][i][2]:dict_crop[file][i][2] + anchor_label[i][1],
                #               dict_crop[file][i][3]:dict_crop[file][i][3] + anchor_label[i][2]]

                sub_label_0 = nsub_label[crop_start_end[0][0]:crop_start_end[0][3],
                                crop_start_end[0][1]:crop_start_end[0][4],
                                crop_start_end[0][2]:crop_start_end[0][5]]
                # sub_label_0 = sub_label_0[:, ::-1, :]
                print('label:', sub_label_0.shape)
                file_name0 = file_c.split('.')[0] + '_00.nii.gz'
                label_name0 = os.path.join(label_save, str(label_wanted[i]), file_name0)
                save_array_as_volume(sub_label_0, label_name0, transpose=False,pixel_spacing=label_spacing)
                print(label_name0)
                if crop_sum!=1:
                    sub_label_1 = nsub_label[
                                  crop_start_end[1][0]:crop_start_end[1][3],
                                  crop_start_end[1][1]:crop_start_end[1][4],
                                  crop_start_end[1][2]:crop_start_end[1][5]]
                    sub_label_2 = nsub_label[
                                  crop_start_end[2][0]:crop_start_end[2][3],
                                  crop_start_end[2][1]:crop_start_end[2][4],
                                  crop_start_end[2][2]:crop_start_end[2][5]]
                    sub_label_3 = nsub_label[
                                  crop_start_end[3][0]:crop_start_end[3][3],
                                  crop_start_end[3][1]:crop_start_end[3][4],
                                  crop_start_end[3][2]:crop_start_end[3][5]]
                    sub_label_4 = nsub_label[
                                  crop_start_end[4][0]:crop_start_end[4][3],
                                  crop_start_end[4][1]:crop_start_end[4][4],
                                  crop_start_end[4][2]:crop_start_end[4][5]]

                    file_name1 = file_c.split('.')[0] + '_01.nii.gz'
                    file_name2 = file_c.split('.')[0] + '_02.nii.gz'
                    file_name3 = file_c.split('.')[0] + '_03.nii.gz'
                    file_name4 = file_c.split('.')[0] + '_04.nii.gz'

                    label_name1 = os.path.join(label_save, str(label_wanted[i]), file_name1)
                    label_name2 = os.path.join(label_save, str(label_wanted[i]), file_name2)
                    label_name3 = os.path.join(label_save, str(label_wanted[i]), file_name3)
                    label_name4 = os.path.join(label_save, str(label_wanted[i]), file_name4)
                    # label_name = label_save

                    save_array_as_volume(sub_label_1, label_name1, transpose=False,pixel_spacing=label_spacing)
                    save_array_as_volume(sub_label_2, label_name2, transpose=False,pixel_spacing=label_spacing)
                    save_array_as_volume(sub_label_3, label_name3, transpose=False,pixel_spacing=label_spacing)
                    save_array_as_volume(sub_label_4, label_name4, transpose=False,pixel_spacing=label_spacing)
                # save_arry_as_nii(sub_label_2, label_name ,pixel_spacing=[1.2, 1.2, 1.2])
            # crop ct t1 t2   data set
            if crop_images:
                ct = file
                t1 = file
                t2 = file
                # modes = [ct,t1,t2]
                modes = [ct]
                for mode in modes:
                    sub_image = os.path.join(image_root, mode)
                    sub_image,image_spacing = load_nii_as_narray(sub_image)
                    # sub_image_1 = sub_image[dict_crop[file][i][1]:dict_crop[file][i][1] + anchor_label[i][0],
                    #               dict_crop[file][i][2]:dict_crop[file][i][2] + anchor_label[i][1],
                    #               dict_crop[file][i][3]:dict_crop[file][i][3] + anchor_label[i][2]]

                    sub_image_0 = sub_image[crop_start_end[0][0]:crop_start_end[0][3],
                                crop_start_end[0][1]:crop_start_end[0][4],
                                crop_start_end[0][2]:crop_start_end[0][5]]
                    # sub_image_0 = sub_image_0[:, ::-1, :]
                    print('image_back:', sub_image_0.shape)
                    file_name0 = mode.split('.')[0] + '_00.nii.gz'
                    image_name0 = os.path.join(image_save, str(label_wanted[i]), file_name0)
                    # save_arry_as_nii(sub_image_0, image_name0,image_spacing)
                    save_array_as_volume(sub_image_0, image_name0, transpose=False,pixel_spacing=image_spacing)
                    print(image_name0)
                    if crop_sum != 1:
                        sub_image_1 = sub_image[
                                      crop_start_end[1][0]:crop_start_end[1][3],
                                    crop_start_end[1][1]:crop_start_end[1][4],
                                    crop_start_end[1][2]:crop_start_end[1][5]]
                        sub_image_2 = sub_image[
                                      crop_start_end[2][0]:crop_start_end[2][3],
                                      crop_start_end[2][1]:crop_start_end[2][4],
                                      crop_start_end[2][2]:crop_start_end[2][5]]
                        sub_image_3 = sub_image[
                                      crop_start_end[3][0]:crop_start_end[3][3],
                                      crop_start_end[3][1]:crop_start_end[3][4],
                                      crop_start_end[3][2]:crop_start_end[3][5]]
                        sub_image_4 = sub_image[
                                      crop_start_end[4][0]:crop_start_end[4][3],
                                      crop_start_end[4][1]:crop_start_end[4][4],
                                      crop_start_end[4][2]:crop_start_end[4][5]]

                        file_name1 = mode.split('.')[0] + '_01.nii.gz'
                        file_name2 = mode.split('.')[0] + '_02.nii.gz'
                        file_name3 = mode.split('.')[0] + '_03.nii.gz'
                        file_name4 = mode.split('.')[0] + '_04.nii.gz'

                        image_name1 = os.path.join(image_save, str(label_wanted[i]), file_name1)
                        image_name2 = os.path.join(image_save, str(label_wanted[i]), file_name2)
                        image_name3 = os.path.join(image_save, str(label_wanted[i]), file_name3)
                        image_name4 = os.path.join(image_save, str(label_wanted[i]), file_name4)

                        save_array_as_volume(sub_image_1, image_name1, transpose=False,pixel_spacing=image_spacing)
                        save_array_as_volume(sub_image_2, image_name2, transpose=False,pixel_spacing=image_spacing)
                        save_array_as_volume(sub_image_3, image_name3, transpose=False,pixel_spacing=image_spacing)
                        save_array_as_volume(sub_image_4, image_name4, transpose=False,pixel_spacing=image_spacing)
        crop_start_end_dict[file] = crop_s_e_dict
    pkl_name = os.path.join(image_out,'crop_start_end.pkl')
    save_pickle(crop_start_end_dict,pkl_name)
    return dict_crop

# 所有数据和器官大小都相同
'''
def crop_same(coarseg_root,image_root,label_root,coarseg_save,image_save,label_save,label_wanted,pading,
         crop_coarsegs,crop_images,crop_labels):
    dict_crop={}
    anchor = {}
    # 找到裁剪图层起始值和裁剪区域
    for file in os.listdir(coarseg_root):  # the file end with  "_0006.nii.gz"
        sub_coarseg = os.path.join(coarseg_root, file)
        sub_coarseg = load_nii_as_narray(sub_coarseg)

        sub_coarseg_size = sub_coarseg.shape
        crop_data = []
        anchor_data = []
        # Extract  label  wanted
        for i in range(len(label_wanted)):

            ncoarseg = np.zeros_like(sub_coarseg)
            ncoarseg[np.where(sub_coarseg == label_wanted[i])] = 1
            sub_coarseg_2 = ncoarseg.copy()
            # get the coordinate
            nonzeropoint = np.asarray(np.nonzero(sub_coarseg_2))


            maxpoint = np.max(nonzeropoint, 1).tolist()

            minpoint = np.min(nonzeropoint, 1).tolist()

            for ii in range(len(pading)):
                maxpoint[ii] = min(maxpoint[ii] + pading[ii], sub_coarseg_size[ii])  # sub_coarseg_size
                minpoint[ii] = max(minpoint[ii] - pading[ii], 0)
            crop_point = minpoint + maxpoint
            crop_point.insert(0, label_wanted[i])
            crop_data.append(crop_point)
            anchor_point = [maxpoint[0]-minpoint[0],
                            maxpoint[1]-minpoint[1],
                            maxpoint[2]-minpoint[2]]
            anchor_point.insert(0,label_wanted[i])
            anchor_data.append(anchor_point)
        dict_crop[file] = crop_data
        anchor[file] = anchor_data
        # print(1)
    # 找到最大的anchor
    anchor_max =[]
    for file in os.listdir(coarseg_root):
        anchor_array = np.array(anchor[file])
        max_z = np.max(anchor_array[:,1])
        max_x = np.max(anchor_array[:,2])
        max_y = np.max(anchor_array[:,3])
        anchor_max_factor = [max_z,max_x,max_y]
        anchor_max.append(anchor_max_factor)
    anchor_max_array = np.array(anchor_max)
    anchor_max_z = np.max(anchor_max_array[:,0])
    anchor_max_x = np.max(anchor_max_array[:,1])
    anchor_max_y = np.max(anchor_max_array[:,2])
    anchor_final = [anchor_max_z,anchor_max_x,anchor_max_y]
    print(1)

    # 最终裁剪
    for file in os.listdir(coarseg_root):
        for i in range(len(label_wanted)):
            # cropping  coarsegTr

            if crop_coarsegs:
                sub_coarseg_origin = os.path.join(coarseg_root, file)
                sub_coarseg_origin = load_nii_as_narray(sub_coarseg_origin)
                sub_coarseg_origin_single = np.zeros_like(sub_coarseg_origin)
                sub_coarseg_origin_single[np.where(sub_coarseg_origin == label_wanted[i])] = 1
                sub_coarseg_3 = sub_coarseg_origin_single[dict_crop[file][i][1]:dict_crop[file][i][1]+anchor_final[0],
                                dict_crop[file][i][2]:dict_crop[file][i][2]+anchor_final[1],
                                dict_crop[file][i][3]:dict_crop[file][i][3]+anchor_final[2]]
                print('coarsegTr:', sub_coarseg_3.shape)
                coarseg_name = os.path.join(coarseg_save, str(label_wanted[i]), file)
                save_array_as_volume(sub_coarseg_3, coarseg_name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
                print(coarseg_name)

            # croppting label
            if crop_labels:
                file_c = file
                sub_label = os.path.join(label_root, file_c)
                sub_label = load_nii_as_narray(sub_label)

                nsub_label = np.zeros_like(sub_label)
                nsub_label[np.where(sub_label == label_wanted[i])] = 1

                sub_label_2 = nsub_label[dict_crop[file][i][1]:dict_crop[file][i][1]+anchor_final[0],
                                dict_crop[file][i][2]:dict_crop[file][i][2]+anchor_final[1],
                                dict_crop[file][i][3]:dict_crop[file][i][3]+anchor_final[2]]
                print('label:', sub_label_2.shape)
                label_name = os.path.join(label_save, str(label_wanted[i]), file_c)
                # label_name = label_save
                save_array_as_volume(sub_label_2, label_name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
                # save_arry_as_nii(sub_label_2, label_name ,pixel_spacing=[1.2, 1.2, 1.2])
                print(label_name)

            # crop ct t1 t2   data set
            if crop_images:
                ct = file
                t1 = file
                t2 = file
                # modes = [ct,t1,t2]
                modes = [ct]
                for mode in modes:
                    sub_image = os.path.join(image_root, mode)
                    sub_image = load_nii_as_narray(sub_image)
                    sub_image_1 = sub_image[dict_crop[file][i][1]:dict_crop[file][i][1]+anchor_final[0],
                                dict_crop[file][i][2]:dict_crop[file][i][2]+anchor_final[1],
                                dict_crop[file][i][3]:dict_crop[file][i][3]+anchor_final[2]]
                    print('image_back:', sub_image_1.shape)
                    image_name = os.path.join(image_save, str(label_wanted[i]), mode)
                    save_array_as_volume(sub_image_1, image_name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
                    print(image_name)

    return dict_crop

def recoverTs(crop_seg_in,crop_dict,origin_label,final_label):
    for file in os.listdir(origin_label):
        origin_label_path = os.path.join(origin_label, file)
        origin_label_data = load_nii_as_narray(origin_label_path)
        origin_shape = origin_label_data.shape
        seg_label = np.zeros(origin_shape)
        for label in os.listdir(crop_seg_in):
            crop_seg_label = os.path.join(crop_seg_in,label)
            crop_seg_path = os.path.join(crop_seg_label,file)
            crop_seg_data = load_nii_as_narray(crop_seg_path)
            crop_seg_data[crop_seg_data!=0] = int(label)
            crop_infor = crop_dict[file][int(label)-7]
            seg_zero = np.zeros(origin_shape)
            seg_zero[crop_infor[1]:crop_infor[4],
            crop_infor[2]:crop_infor[5],
            crop_infor[3]:crop_infor[6]
            ] = crop_seg_data
            seg_label = np.where(seg_zero!=0,seg_zero,seg_label)
            origin_label_data[origin_label_data==int(label)] = 0
        final_data = np.where(seg_label!=0,seg_label,origin_label_data)
        final_name = os.path.join(final_label,file)
        save_array_as_volume(final_data, final_name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])

def dice_calculate(out_path,target_path):
    out_name = os.listdir(out_path)
    target_name = os.listdir(target_path)
    dice = []
    for i,j in zip(out_name,target_name):
        out_factor = os.path.join(out_path,i)
        target_factor = os.path.join(target_path,j)
        out_data = load_nii_as_narray(out_factor).astype(np.int64)
        target_data = load_nii_as_narray(target_factor).astype(np.int64)
        out_tensor = torch.tensor(out_data)
        target_tensor = torch.tensor(target_data)
        out = F.one_hot(out_tensor)
        target = F.one_hot(target_tensor)
        out = np.array(out)
        target = np.array(target)
        num_class = out.shape[3]
        dice_label = 0
        for i in range(num_class):
            temp1 = out[:,:,:,i]
            temp2 = target[:,:,:,i]
            out_fact = out[:,:,:,i].flatten()
            target_fact = target[:,:,:,i].flatten()
            numerator = (out_fact * target_fact).sum()
            out_sum = out_fact.sum()
            target_sum = target_fact.sum()
            dice_factor = (2*(numerator + 0.0000001)) / (out_sum + target_sum + 0.0000001)
            dice_label+=dice_factor
        dice_f = dice_label/num_class
        dice.append(dice_f)
    dice_sum = sum(dice)
    dice_mean = dice_sum/len(dice)
    return dice_mean,dice
def dice2(out_path,target_path):
    out_name = os.listdir(out_path)
    target_name = os.listdir(target_path)
    dice = []
    for i,j in zip(out_name,target_name):
        out_factor = os.path.join(out_path,i)
        target_factor = os.path.join(target_path,j)
        out_data = load_nii_as_narray(out_factor).astype(np.int64)
        target_data = load_nii_as_narray(target_factor).astype(np.int64)
        out_tensor = torch.tensor(out_data)
        target_tensor = torch.tensor(target_data)
        out_hot = F.one_hot(out_tensor)
        out = out_hot.permute(3,0,1,2)
        dice_factor = DC_and_CE_loss(out,target_tensor)
        print(1)
'''


if __name__ == '__main__':

    # coarseg_in = 'data/nnunet_PDDCA/coarsegTr'
    # image_in = 'data/nnunet_PDDCA/image_back'
    # label_in = 'data/nnunet_PDDCA/label'
    # # coarseg_in = 'data/PDDCA_crop_foreground/coarseg_label_crop/9'
    # # image_in = 'data/PDDCA_crop_foreground/image_crop/9'
    # # label_in = 'data/PDDCA_crop_foreground/label_crop/9'
    # coarseg_out = 'data/PDDCA_crop_sum/PDDCA_crop_16_160_160/coarseg_crop'
    # image_out = 'data/PDDCA_crop_sum/PDDCA_crop_16_160_160/image_crop'
    # label_out = 'data/PDDCA_crop_sum/PDDCA_crop_16_160_160/label_crop'
    # # final_path_in = 'data/temp/final_in'
    # # final_path_out = 'data/temp/final_o'
    # label_wanted = [7]
    # padding = [0, 0, 0]

    coarseg_in = ''
    image_in = ''
    label_in = ''
    coarseg_out = ''
    image_out = ''
    label_out = ''
    # final_path_in = 'data/temp/final_in'
    # final_path_out = 'data/temp/final_o'
    label_wanted = [7,8]
    padding = [0, 0, 0]

    data_int = np.array([])

    crop_coarsegs = True
    crop_images = True
    crop_labels = True
    crop_sum = 1
    single_label = True

    dict_infor = crop_data_same(coarseg_in,image_in,label_in,coarseg_out,image_out,label_out,label_wanted,padding,data_int,
                      crop_coarsegs,crop_images,crop_labels,crop_sum,single_label)
    # recoverTs(final_path_in,dict_infor,coarseg_in,final_path_out)

    # dice_mean,dice = dice_calculate(coarseg_in,label_in)
    print(1)
