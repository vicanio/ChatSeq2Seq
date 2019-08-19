# encoding: utf-8


import os

import tensorflow as tf
import numpy as np

from SequenceToSequence import Seq2Seq
import DataProcessing
from CONFIG import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config
import random
import json
import time
from flask import Flask, request,render_template


class AIChat(object):
    def __init__(self):
        print("init")
        self.du = DataProcessing.DataUnit(**data_config)
        save_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)
        batch_size = 1
        tf.reset_default_graph()
        self.model = Seq2Seq(batch_size=batch_size,
                        encoder_vocab_size=self.du.vocab_size,
                        decoder_vocab_size=self.du.vocab_size,
                        mode='decode',
                        **model_config)
        # 创建session的时候允许显存增长
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config = tf.ConfigProto(
            device_count={'CPU': 1, 'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False
        )
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.model.load(self.sess, save_path)

    def getAIMsg(self, question):
        if question is None or question.strip() == '':
            return "请输入聊天信息"

        question = question.strip()
        indexs = self.du.transform_sentence(question)
        x = np.asarray(indexs).reshape((1, -1))
        xl = np.asarray(len(indexs)).reshape((1,))

        end = time.time()

        pred = self.model.predict(
            self.sess, np.array(x),
            np.array(xl)
        )
        end1 = time.time()
        print("t4:", end1 - end)
        ll = len(pred)
        num = 0
        if ll > 20:
            num = random.randint(0, round(ll/3 - 1))
        else:
            num = random.randint(0,(ll-1))

        return self.du.transform_indexs(pred[num])

app = Flask(__name__)

RESFMT = '{"question":"%s","answer":"%s"}'
aichat = AIChat()

@app.route('/api/qazwsx',methods=["GET","POST"])
def hello_world():
    if request.method == "GET":
        infos = request.args['infos']
        ret = aichat.getAIMsg(infos)
        result = RESFMT % (infos, ret)
        return "".join(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)