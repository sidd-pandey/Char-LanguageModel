import numpy as np
import tensorflow as tf

from flask_cors import cross_origin
from flask import Flask
from flask import (
    Blueprint, request, jsonify
)
from keras.models import load_model
from editor.utils import generate_sample

bp = Blueprint('api', __name__, url_prefix='/api')


print("loading model...")
model = load_model("editor/saved_model/best_model-3layer-512-256-256.h5")
graph = tf.get_default_graph()
print("model loaded!")

@bp.route('/text')
def gentext():
    
    seed = request.args.get("seed", default="never judge a book by its ")

    with graph.as_default():
        text = generate_sample(model, "flask model", text=seed, temperatures=[0.6])
    
    text = text[0][3]
    return text