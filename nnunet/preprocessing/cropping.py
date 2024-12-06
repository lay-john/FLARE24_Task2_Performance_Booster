#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import SimpleITK as sitk
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict
import time
import nibabel as nib
from scipy import ndimage


def keep_largest_connected_component_region_growing(binary_image: np.ndarray) -> np.ndarray:
    """
    基于区域生长的标记算法，保留二值图像中的最大连通域。

    参数:
        binary_image (np.ndarray): 输入的二值图像（0 和 1）。

    返回:
        np.ndarray: 只包含最大连通域的二值图像。
    """
    # 确保输入是二值图像
    if not np.array_equal(binary_image, binary_image.astype(bool)):
        raise ValueError("输入图像必须是二值图像（仅包含0和1）。")

    # 图像形状
    shape = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)  # 记录是否访问过
    max_size = 0
    max_region = None

    # 定义 6 连通方向（针对 3D 图像）
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    def region_grow(z, y, x):
        """区域生长，返回区域大小和区域体素坐标"""
        stack = [(z, y, x)]
        region = []
        size = 0

        while stack:
            cz, cy, cx = stack.pop()
            if (
                    0 <= cz < shape[0]
                    and 0 <= cy < shape[1]
                    and 0 <= cx < shape[2]
                    and binary_image[cz, cy, cx] == 1
                    and not visited[cz, cy, cx]
            ):
                visited[cz, cy, cx] = True  # 标记为已访问
                region.append((cz, cy, cx))
                size += 1
                # 将相邻体素加入栈
                for dz, dy, dx in directions:
                    stack.append((cz + dz, cy + dy, cx + dx))

        return size, region

    # 遍历图像中的每个体素
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if binary_image[z, y, x] == 1 and not visited[z, y, x]:
                    # 启动区域生长
                    size, region = region_grow(z, y, x)
                    if size > max_size:
                        max_size = size
                        max_region = region

    # 创建结果图像，只保留最大连通域
    result = np.zeros_like(binary_image, dtype=np.uint8)
    if max_region:
        for z, y, x in max_region:
            result[z, y, x] = 1

    return result


def keep_largest_connected_component(binary_image):
    # 1. 计算所有连通域的标签
    labeled_array, num_features = ndimage.label(binary_image)
    labeled_array = labeled_array.astype(np.int32)
    if num_features == 0:
        print("没有找到连通域")
        return binary_image

    # 2. 计算每个连通域的体素数量
    sizes = ndimage.sum(binary_image, labeled_array, range(num_features + 1))
    del binary_image
    # 3. 找到最大连通域的标签
    max_label = np.argmax(sizes)

    # 4. 创建一个新的二值图像，只保留最大连通域
    largest_component = (labeled_array == max_label)

    return largest_component

def get_bbox_from_mask_b(nonzero_mask, outside_value):
    nonzero_mask = nonzero_mask[0]
    mask_voxel_coords = np.where(nonzero_mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    return bbox

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def get_case_identifier(case):
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    return case_identifier


def get_case_identifier_from_npz(case):
    case_identifier = case.split("/")[-1][:-4]
    return case_identifier


def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    start = time.time()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    print('simpleitk time: ', time.time()-start)
    return data_npy.astype(np.float32), seg_npy, properties


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    # nonzero_mask = create_nonzero_mask(data)
    # nonzero_mask = data[0] != 0
    # start = time.time()
    # bbox = get_bbox_from_mask(nonzero_mask, 0)
    # print('create time2_1: ',time.time() - start)
    # start = time.time()
    '''
        if data.shape[1] > 1200:
            bbox = [[300, data.shape[1]],[0, data.shape[2]],[0, data.shape[3]]]
            resizer = (slice(0, 1), slice(300, data.shape[1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
            data = data[resizer]
            return data, bbox
        if data.shape[1] > 1000:
            bbox = [[200, data.shape[1]],[0, data.shape[2]],[0, data.shape[3]]]
            resizer = (slice(0, 1), slice(200, data.shape[1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
            data = data[resizer]
            return data, bbox
        '''
    if data.shape[1] > 1200:
        bbox = [[400, data.shape[1]], [0, data.shape[2]], [0, data.shape[3]]]
        resizer = (slice(0, 1), slice(400, data.shape[1]), slice(0, data.shape[2]), slice(0, data.shape[3]))
        data = data[resizer]
        return data, bbox
    if data.shape[1] > 400:
        #if data.shape[1] > 1200:
         #   resizer1 = (slice(0, 1), slice(0, data.shape[1]), slice(0, data.shape[2]), slice(0, data.shape[3]))
        if data.shape[1] > 1000:
            resizer1 = (slice(0, 1), slice(400, data.shape[1] - 250), slice(0, data.shape[2]), slice(0, data.shape[3]))
        elif data.shape[1] > 800:
            resizer1 = (slice(0, 1), slice(300, data.shape[1] - 200), slice(0, data.shape[2]), slice(0, data.shape[3]))
        elif data.shape[1] > 600:
            resizer1 = (slice(0, 1), slice(200, data.shape[1] - 100), slice(0, data.shape[2]), slice(0, data.shape[3]))
        else:
            resizer1 = (slice(0, 1), slice(150, data.shape[1] - 100), slice(0, data.shape[2]), slice(0, data.shape[3]))
        data1 = data[resizer1]
        pp = data1 > 50
        del data1
        pp = keep_largest_connected_component(pp)
        # pp = keep_largest_connected_component(pp)
        bbox = get_bbox_from_mask_b(pp, outside_value=0)
        del pp
        bbox[0][0] = 0
        bbox[0][1] = data.shape[1]
        resizer = (slice(0, 1), slice(0, data.shape[1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        data = data[resizer]

    else:
        pp = data > 50
        pp = keep_largest_connected_component(pp)
        # pp = keep_largest_connected_component(pp)
        bbox = get_bbox_from_mask_b(pp, outside_value=0)
        del pp
        resizer = (slice(0, 1), slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        data = data[resizer]
    return data, bbox









    #print("##################################", bbox1)
    #print("#############", bbox)
    #bbox = [[0, data.shape[1]], [0, data.shape[2]], [0, data.shape[3]]]  # pre
    # crop array

    # bbox = get_bbox_from_mask_b(nonzero_mask, 0)


    #bbox = [[0, data.shape[1]], [0, data.shape[2]], [0, data.shape[3]]]


    # print('create time2_2: ',time.time() - start)

    # cropped_data = []
    # for c in range(data.shape[0]):
    #     cropped = crop_to_bbox(data[c], bbox)
    #     cropped_data.append(cropped[None])
    # data = np.vstack(cropped_data)

    # if seg is not None:
    #     cropped_seg = []
    #     for c in range(seg.shape[0]):
    #         cropped = crop_to_bbox(seg[c], bbox)
    #         cropped_seg.append(cropped[None])
    #     seg = np.vstack(cropped_seg)

    # nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    # if seg is not None:
    #     seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    # else:
    #     nonzero_mask = nonzero_mask.astype(int)
    #     nonzero_mask[nonzero_mask == 0] = nonzero_label
    #     nonzero_mask[nonzero_mask > 0] = 0
    #     seg = nonzero_mask
    # seg = np.ones(data.shape, dtype=int)
    # return data, seg, bbox
    #return data, bbox1


def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]


class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None):
        """
        This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        In the case of BRaTS and ISLES data this results in a significant reduction in image size
        :param num_threads:
        :param output_folder: whete to store the cropped data
        :param list_of_files:
        """
        self.output_folder = output_folder
        self.num_threads = num_threads

        if self.output_folder is not None:
            maybe_mkdir_p(self.output_folder)

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape
        # data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        data, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        properties["crop_bbox"] = bbox
        # properties['classes'] = np.unique(seg)
        # seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        # return data, seg, properties
        return data, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            print(case_identifier)
            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % case_identifier))):

                data, seg, properties = self.crop_from_list_of_files(case[:-1], case[-1])

                all_data = np.vstack((data, seg))
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % case_identifier), data=all_data)
                with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e

    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [i.split("/")[-1][:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        maybe_mkdir_p(output_folder_gt)
        for j, case in enumerate(list_of_files):
            if case[-1] is not None:
                shutil.copy(case[-1], output_folder_gt)

        list_of_args = []
        for j, case in enumerate(list_of_files):
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)
