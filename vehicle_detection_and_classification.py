import argparse
import os
import re
import tarfile

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from six.moves import urllib
import cv2

# from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(1,'/opt/AItrain/ssd_inception/vehicle_detector/vehicle_detection')
#sys.path.insert(1,'E:/VC_project/VehicleDetection/slim')

from utils import visualization_utils as vis_util
from utils import label_map_util

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.80
session = tf.Session(config=config)
# 预训练模型基于coco数据集，有90个类别
NUM_CLASSES = 90

FLAGS = tf.app.flags.FLAGS

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--classification_model_file',type=str,
                        default='freezed.pb',
                        help="""\
                        Path to the classification .pb file that contains the frozen weights. \
                        """)
    parser.add_argument('--detection_model_file',type=str,default='frozen_inference_graph.pb',
                        help='Path to the detection .pb file that contains the frozen weights.')
    parser.add_argument('--label_file',type=str,
                        default='freezed.label',
                        help='Absolute path to label file.')
    parser.add_argument('--image_file',type=str,default='',help='Absolute path to image file.')
    parser.add_argument('--num_top_predictions',type=int,default=3,
                        help='Display this many predictions.')
    parser.add_argument('--output_path',type=str,default='',help='Absolute path to output image file')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

def resize_image(image, height, width):
    '''
    按照指定图像大小调整尺寸：
    inception_v4网络的论文中输入尺寸为：299*299
    本次训练的数据集输入图片尺寸为：320*432
    :param image: 输入图片
    :param height: 指定高
    :param width: 指定宽
    :return:
    '''

    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 获取指定尺寸图片的高宽比
    ratio = height/width

    # 计算需要增加多上像素,使其与高宽比例相同
    if h/w < ratio:
        dh = int(w*ratio) - h
        top = dh // 2
        bottom = dh - top
    elif h/w > ratio:
        dw = int(h/ratio) - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # 调整图像大小并返回
    return cv2.resize(constant, (width,height))

def compute_IoU(image,box1,box2):

    # box = [ymin,xmin,ymax,xmax]
    height,weight = image.size

    area = height*weight
    in_h = min(box1[2],box2[2]) - max(box1[0],box2[0])
    in_w = min(box1[3],box2[3]) - max(box1[1],box2[1])

    # 计算交集
    inter = 0 if in_h < 0 or in_w < 0 else in_h*in_w*area
    # 计算并集
    union = (box1[2]-box1[0])*(box1[3]-box1[1])*area + (box2[2]-box2[0])*(box2[3]-box2[1])*area - inter

    iou = inter/union

    return iou

def image_crop(boxes,classes,scores,image):
    '''
    根据检测的结果完成图片的裁剪
    :param boxes: 检测box位置信息
    :param classes: 每个检测框的类别
    :param scores: 检测框属于某类别的概率
    :param image: 需裁剪的图片
    :return: 返回裁剪的图片以及对应位置信息
    '''


    # 仅保存检测到的汽车以及识别率大于75%的区域
    order = []
    for i in range(len(classes)):
        if (classes[i] == 3 or classes[i] == 8 or classes[i] == 6) and scores[i] > 0.7:
            for k in range(len(boxes) - 1):
                if scores[k] > 0:
                    for j in range(len(boxes) - k - 1):
                        iou = compute_IoU(image, boxes[k], boxes[k + j + 1])
                        if iou > 0.5:
                            scores[k + j + 1] = 0
            if scores[i] > 0 :
                order.append(i)

    boxes_recog = np.zeros((len(order), 4))
    classes_recog = np.zeros(len(order))
    scores_recog = np.zeros(len(order))

    for idx in order:
        boxes_recog[order.index(idx)] = boxes[idx]
        classes_recog[order.index(idx)] = classes[idx]
        scores_recog[order.index(idx)] = scores[idx]


    # 按box中的边界裁剪图片
    # box = [ymin,xmin,ymax,xmax]
    # weight, height = image.size
    # crop_box = [weihgt*xmin,height*ymin,weight*xmax,height*ymax]

    images = []
    weight, height = image.size
    # 裁剪图片
    for boxes_idx in boxes_recog:
        [ymin,xmin,ymax,xmax] = boxes_idx
        crop_box = [weight * xmin, height * ymin, weight * xmax, height * ymax]
        img_crop = image.crop(crop_box)


        # 将图片转成cv2读取的格式
        cv_img = cv2.cvtColor(np.asarray(img_crop), cv2.COLOR_RGB2BGR)
        # 调整裁剪图片的大小
        re_img = resize_image(cv_img,320,432)
        # 将图片转成Image读取格式
        pil_image = Image.fromarray(cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB))

        #plt.imshow(re_img)
        #plt.show()

        images.append(pil_image)

    return images,boxes_recog

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""
    def __init__(self,
                 label_path=None):
        if not label_path:
            tf.logging.fatal('please specify the label file.')
            return
        self.node_lookup = self.load(label_path)

    def load(self, label_path):
        """Loads a human readable English name for each softmax node.
        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.
        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(label_path):
            tf.logging.fatal('File does not exist %s', label_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()

        id_to_human = {}
        for line in proto_as_ascii_lines:
            if line.find(':') < 0:
                continue
            _id, human = line.rstrip('\n').split(':')
            id_to_human[int(_id)] = human

        return id_to_human

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def load_label_to_list(label_path):
    '''
     将label.txt中的内容转换成字典类型
    :param label_path:
    :return:
    '''
    f = open(label_path, encoding="utf-8")
    categories = []
    line = f.readline()
    while line:
        line = line.strip('\n')
        line_info = line.split(':')
        line_id = int(line_info[0])
        line_name = line_info[1]
        category_line = {'id': line_id, 'name': line_name}
        categories.append(category_line)
        line = f.readline()
    f.close()
    return categories

def load_image_into_numpy_array(image):
    '''
    将图片转换为np.array数组的形式
        :param image:
        :return:
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_inference_on_image(image, graph):
    """Runs inference on an image.
    使用inception_v4网络，对裁剪的图片进行分类
    Args:
      image: Image file name.
    Returns:
      Nothing
    """
    # 先将图像转换成数组类型，然后再转换成二进制文件
    image_np = load_image_into_numpy_array(image)
    ret, buf = cv2.imencode(".jpg", image_np)
    image_data = Image.fromarray(np.uint8(buf)).tobytes()

    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   764 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            softmax_tensor = sess.graph.get_tensor_by_name('final_probs:0')
            predictions = sess.run(softmax_tensor,
                                   {'input:0': image_data})
            predictions = np.squeeze(predictions)

            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup(FLAGS.label_file)

            top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
            top_names = []
            print("****************************************")
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                top_names.append(human_string)
                score = predictions[node_id]
                print('id:[%d] name:[%s] (score = %.5f)' %
                      (node_id, human_string, score))
    # 返回top-3的类型及概率
    return predictions, top_k, top_names


def detection_and_classifation(image_file,classification_model_file,detection_model_file,output_path):
    # 创建检测模型的Graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(detection_model_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # 创建分类模型的Graph
    classification_graph = tf.Graph()
    with classification_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(classification_model_file, 'rb') as fid1:
            serialized_graph1 = fid1.read()
            graph_def.ParseFromString(serialized_graph1)
            tf.import_graph_def(graph_def, name="")

    # 将车辆类别转车字典型数据
    categories = load_label_to_list(FLAGS.label_file)
    category_index = label_map_util.create_category_index(categories)

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



            # 打开图片及转换成数组形式
            image = Image.open(image_file)
            image_np = load_image_into_numpy_array(image)

            # 将图片扩展一维，最后进入网络的图片格式应该是[1,weight,height,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # 检测图片
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # 去掉size为1的维度
            boxes_squeeze = np.squeeze(boxes)
            classes_squeeze = np.squeeze(classes).astype(np.int32)
            scores_squeeze = np.squeeze(scores)

            # 根据识别的box裁剪图片
            images, boxes_list = image_crop(boxes_squeeze, classes_squeeze, scores_squeeze, image)

            top_predictions_list = []
            top_class_list = []
            top_names_list = []

            predictions_list = []
            class_list = []
            names_list = []

            for i,img in enumerate(images):
                # 对裁剪图片中的车辆进行分类
                predictions, top_k, top_names = run_inference_on_image(img, classification_graph)
                # 取出概率最高的类型
                top_class_list.append(top_k[0])
                top_predictions_list.append(predictions[top_k[0]])
                top_names_list.append(top_names[0])

                if predictions[top_k[0]] > 0.25:
                    predictions_list.append(predictions.tolist())
                    class_list.append(top_k.tolist())
                    names_list.append(top_names)

            # 可视化
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes_list,  # np.squeeze(boxes),
                top_class_list,  # np.squeeze(classes).astype(np.int32),
                top_predictions_list,  # np.squeeze(scores),
                category_index,  # top_names_list
                min_score_thresh=0.25,
                use_normalized_coordinates=True,
                line_thickness=8)
            # plt.imsave(os.path.join(FLAGS.output_dir, 'output.png'), image_np)
            save_path = output_path+os.path.basename(image_file).split('.')[0]+'_prediction'+'.png'

            #plt.imsave(os.path.join('E:/CSDN人工智能培训/实战项目-车辆检测/image_dir', 'output.png'), image_np)
            plt.imsave(save_path,image_np)

            return predictions_list,class_list,names_list,save_path


def main(_):
    # 获取检测图片路径
    # test_img_path = os.path.join(FLAGS.dataset_dir, 'test.jpg')

    detection_and_classifation(FLAGS.image_file,FLAGS.classification_model_file,FLAGS.detection_model_file,FLAGS.output_path)



if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


