# encoding: utf-8


import os

import tensorflow as tf
import numpy as np

from SequenceToSequence import Seq2Seq
import DataProcessing
from CONFIG import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config
import random
import tornado.web
import tornado.ioloop
from tornado.web import RequestHandler
import json
import time


class AIChat(object):
    def __init__(self):
        print("init111")
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


class BaseHandler(RequestHandler):
    """解决JS跨域请求问题"""

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Content-type', 'application/json')


class IndexHandler(BaseHandler):
    RESFMT = '{"question":"%s","answer":"%s"}'
    aichat = AIChat()

    def writecontent(self, infos):
        if infos is None or len(infos) == 0:
            print("A:", "".join("null"))
            self.write(self.RESFMT % (infos, ""))
        else:
            # 捕捉服务器异常信息
            start = time.time()
            ret = self.aichat.getAIMsg(infos)
            result = self.RESFMT % (infos, ret)
            self.write("".join(result))
            end = time.time()
            print("t___:", end - start)

    def post(self):
        jobj = json.loads(self.request.body)
        infos = jobj['q']
        self.writecontent(infos)

    def get(self):
        # 向响应中，添加数据
        infos = self.get_query_argument("infos")
        start = time.time()
        self.writecontent(infos)
        end = time.time()
        print("t_over:", end - start)


if __name__ == '__main__':
    #test()
    app = tornado.web.Application([(r'/api/qazwsx', IndexHandler)])
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()
