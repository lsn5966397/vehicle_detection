import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.insert(1,'E:/VC_project/AdvancedLessons/FifthWeek/ModelCheck_Practice/models/research')
sys.path.insert(1,'E:/VC_project/AdvancedLessons/FifthWeek/ModelCheck_Practice/models/research/slim')

from utils import visualization_utils as vis_util
from utils import label_map_util


NUM_CLASSES = 90


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    # FLAGS, unparsed = parse_args()

    # PATH_TO_CKPT = os.path.join(FLAGS.output_dir, 'frozen_inference_graph.pb')
    PATH_TO_CKPT = 'E:/CSDN人工智能培训/实战项目-车辆检测/pre_ckpt/ssd_mobilenet_v1_coco_2018_01_28' \
                   '/frozen_inference_graph.pb'
    # PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'mscoco_label_map.pbtxt')
    PATH_TO_LABELS = 'E:/CSDN人工智能培训/实战项目-车辆检测/data/mscoco_label_map.pbtxt'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        '''
        将图片转换为ndarray数组的形式
        :param image:
        :return:
        '''
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # test_img_path = os.path.join(FLAGS.dataset_dir, 'test.jpg')
    test_img_path = 'E:/CSDN人工智能培训/实战项目-车辆检测/image_dir/公路汽车/car2.jpg'

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # 获取模型中的tensor

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # boxes用来显示识别结果
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Echo score代表识别出的物体与标签匹配的相似程度，在类型标签后面
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # 打开图片
            image = Image.open(test_img_path)
            image_np = load_image_into_numpy_array(image)

            # 将图片扩展一维，最后进入神经网络的图片格式应该是[1,weight,height,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes1 = np.squeeze(boxes)
            classes1 = np.squeeze(classes).astype(np.int32)
            scores1 = np.squeeze(scores)

            # 可视化
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                min_score_thresh=0.5,
                use_normalized_coordinates=True,
                line_thickness=4)
            # plt.imsave(os.path.join(FLAGS.output_dir, 'output.png'), image_np)
            plt.imsave(os.path.join('E:/CSDN人工智能培训/实战项目-车辆检测/image_dir', 'output.png'), image_np)

