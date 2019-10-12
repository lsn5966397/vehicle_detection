#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import uuid

import tensorflow as tf
from flask import Flask, redirect, request, send_from_directory, url_for

from vehicle_detection_and_classification import detection_and_classifation

ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png'])

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '', """Path to graph_def pb, """)
tf.app.flags.DEFINE_string('classification_model_file', 'freezed.pb', '')
tf.app.flags.DEFINE_string('detection_model_file', 'frozen_inference_graph.pb', '')

tf.app.flags.DEFINE_string('label_file', 'freezed.label', '')
tf.app.flags.DEFINE_string('output_path', '', '')
tf.app.flags.DEFINE_integer('num_top_predictions', 3,
                            """Display this many predictions.""")
tf.app.flags.DEFINE_integer('port', '6006',
                            'server with port,if no port, use deault port 80')

tf.app.flags.DEFINE_boolean('debug', False, '')

UPLOAD_FOLDER = FLAGS.output_path
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png'])

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER


def allowed_files(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name

def inference(file_name):

    try:
        predictions, top_k, top_names, image = detection_and_classifation(
            file_name, classification_model_file=FLAGS.classification_model_file,
            detection_model_file=FLAGS.detection_model_file, output_path=FLAGS.output_path)

    except Exception as ex:
        print(ex)
        return ""


    new_url = '/static/%s' % os.path.basename(image)
    print (new_url)
    image_tag = '<img src="%s"></img><p>'
    new_tag = image_tag % new_url
    format_string = ''
    if len(predictions) > 0:
        for i,prediction in enumerate(predictions):
            format_string += '*********** vehicle_orders:_%s ***********<BR>' % (i)
            for node_id, human_name in zip(top_k[i], top_names[i]):
                score = prediction[node_id]
                format_string += '%s (score:%.5f)<BR>' % (human_name, score)
            ret_string = new_tag + format_string + '<BR>'
    else:
        ret_string = new_tag + 'It doesn\'t detector vehicle！！！' + '<BR>'
    return ret_string

@app.route("/", methods=['GET', 'POST'])
def root():
    result = """
    <!doctype html>
    <title>车辆预测系统</title>
    <h1>请输入待预测图片</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value='选择图片'>
         <input type=submit value='开始预测'>
    </form>
　　<h2>预测结果：</h2>
    <p>%s</p>
    """ % "<br>"
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        print('---------old_file_name---------',old_file_name)
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(FLAGS.output_path, old_file_name)
            file.save(file_path)
            # type_name = 'N/A'
            # print('file saved to %s' % file_path)
            out_html = inference(file_path)
            print(out_html)
            return result + out_html
    return result

if __name__ == "__main__":
    print('listening on port %d' % FLAGS.port)
    app.run(host='127.0.0.1', port=FLAGS.port, debug=FLAGS.debug, threaded=True)
