import os
import base64
import logging
import time
import argparse
import pickle
import math

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2

import utils
import networks

logging.basicConfig(
    level=logging.INFO,
    format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
            '%(levelname)s - %(message)s'),
)
logger = logging.getLogger(__name__)


class FBVCWorker:

    def __init__(self):

        self.opts = self._get_opts()
        print(self.opts)

        if self.opts.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without -cuda")

        self._init_model(self.opts)
        self.frame_i1 = None
        self.frame_i2 = None
        self.frame_o1 = None
        self.frame_o2 = None
        self.frame_p2 = None
        self.video_raw = None
        self.video_processed = None
        self.lstm_state = None
        self.H_orig = None
        self.W_orig = None
        self.H_sr = None
        self.W_sr = None
        self.output_dir = './output_dir_tmp'

    def _get_opts(self):
        parser = argparse.ArgumentParser(
            description='Fast Blind Video Temporal Consistency'
        )

        # dataset options
        parser.add_argument(
            '-phase', type=str, default="test", choices=["train", "test"])
        parser.add_argument(
            '-data_dir', type=str, default='data', help='path to data folder')
        parser.add_argument(
            '-list_dir', type=str, default='lists', help='path to list folder')
        parser.add_argument(
            '-redo', action="store_true", help='Re-generate results')

        # other options
        parser.add_argument(
            '-gpu', type=int, default=0, help='gpu device id')

        opts = parser.parse_args()
        opts.cuda = True

        # Inputs to TransformNet need to be divided by 4
        opts.size_multiplier = 2 ** 2

        return opts

    def _init_model(self, opts):
        # load model opts
        opts_filename = os.path.join(
            'pretrained_models', "ECCV18_blind_consistency_opts.pth")
        print("Load %s" % opts_filename)
        with open(opts_filename, 'rb') as f:
            model_opts = pickle.load(f)

        # initialize model
        print('===> Initializing model from %s...' % model_opts.model)
        self.model = networks.__dict__[model_opts.model](
            model_opts, nc_in=12, nc_out=3)

        # load trained model
        model_filename = os.path.join(
            'pretrained_models', "ECCV18_blind_consistency.pth")
        print("Load %s" % model_filename)
        state_dict = torch.load(model_filename)
        self.model.load_state_dict(state_dict['model'])

        # convert to GPU
        self.device = torch.device("cuda" if opts.cuda else "cpu")
        self.model = self.model.to(self.device)

        self.model.eval()

    def _convert_input(self):
        self.frame_i1 = cv2.resize(self.frame_i1, (self.W_sc, self.H_sc))
        self.frame_i2 = cv2.resize(self.frame_i2, (self.W_sc, self.H_sc))
        self.frame_o1 = cv2.resize(self.frame_o1, (self.W_sc, self.H_sc))
        self.frame_p2 = cv2.resize(self.frame_p2, (self.W_sc, self.H_sc))
        # convert to tensor
        self.frame_i1 = utils.img2tensor(self.frame_i1).to(self.device)
        self.frame_i2 = utils.img2tensor(self.frame_i2).to(self.device)
        self.frame_o1 = utils.img2tensor(self.frame_o1).to(self.device)
        self.frame_p2 = utils.img2tensor(self.frame_p2).to(self.device)

        # model input
        inputs = torch.cat(
            (self.frame_p2, self.frame_o1, self.frame_i2, self.frame_o1),
            dim=1
        )
        return inputs

    def infer(self, image_raw, image_processed, frame_count):

        start_time = time.time()

        if frame_count == 0:
            self.frame_i1 = image_raw
            self.frame_o1 = image_processed
            self.H_orig = self.frame_i1.shape[0]
            self.W_orig = self.frame_i1.shape[1]
            self.H_sc = int(math.ceil(float(
                self.H_orig) / self.opts.size_multiplier
            ) * self.opts.size_multiplier)
            self.W_sc = int(math.ceil(float(
                self.W_orig) / self.opts.size_multiplier
            ) * self.opts.size_multiplier)

        else:
            with torch.no_grad():
                self.frame_i2 = image_raw
                self.frame_p2 = image_processed
                inputs = self._convert_input()
                output, self.lstm_state = self.model(inputs, self.lstm_state)
                self.frame_o2 = self.frame_p2 + output
                # create new variable to detach from graph and avoid
                # memory accumulation
                self.lstm_state = utils.repackage_hidden(self.lstm_state)

            # convert to numpy array
            self.frame_o2 = utils.tensor2img(self.frame_o2)

            # resize to original size
            self.frame_o2 = cv2.resize(
                self.frame_o2, (self.W_orig, self.H_orig))

            # Set new i1 and o1
            self.frame_i1 = image_raw
            self.frame_o1 = self.frame_o2

        # return output frame
        buf = cv2.imencode(".jpg", self.frame_o1)[1]
        encoded_string = base64.b64encode(buf)
        encoded_result_image = (
            b'data:image/jpeg;base64,' + encoded_string
        )
        logger.info("Infer time: {}".format(time.time() - start_time))
        return encoded_result_image


app = Flask(__name__)
CORS(app)
fbvc_worker = FBVCWorker()


@app.route('/hi', methods=['GET'])
def hi():
    return jsonify({"message": "Hi!"})


@app.route('/fbvc', methods=['POST'])
def fast_blind_video_consistency():
    """
    try:
        fbvc_worker.infer()
    except Exception as err:
        logger.error(str(err), exc_info=True)
    return

    """

    try:
        image_file_raw = request.files['raw']
        image_file_processed = request.files['processed']
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "has no files['raw']"
        )
    if image_file_raw is None or image_file_processed is None:
        raise InvalidUsage('There is no iamge')
    try:
        image_raw = utils.read_img_from_file_storage(image_file_raw)
        image_processed = utils.read_img_from_file_storage(
            image_file_processed)
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request.files['raw'] {request.files['raw']} "
            "could not be read by opencv"
        )
    try:
        result = fbvc_worker.infer(
            image_raw, image_processed, int(request.values['num'])
        )
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "The server encounters some error to process this image",
            status_code=500
        )
    return jsonify({'result': result.decode('utf-8')})


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
