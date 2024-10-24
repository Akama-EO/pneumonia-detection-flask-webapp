import os
import sys
import imghdr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect,  \
    url_for, abort, send_from_directory, jsonify, session

#=====================================

# To find local version of the library
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR) 

# import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

import keras.backend

K = keras.backend.backend()
if K == 'tensorflow':
    keras.backend.set_image_dim_ordering('tf')

#=====================================

# configure the Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.dcm', '.jpg', '.png']
app.config['MODEL_PATH'] = 'model/resnet50_pre-trained.h5'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = "secretkey"  


app.config.update(
DROPZONE_REDIRECT_VIEW ='prediction',  # set redirect view
DROPZONE_MAX_FILES=1,
)

dropzone = Dropzone(app)

#=======================================
# These parameters have been selected for demonstration purposes
# Note: These parameters are not optimal

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 200

# custom configurations for inference
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False

# load the configuration
inference_config = InferenceConfig()

# initialize Mask R-CNN model
model = modellib.MaskRCNN(mode = "inference", 
                          model_dir = "./model", 
                          config = inference_config)

# load model weights globally
model.load_weights(app.config['MODEL_PATH'], by_name=True)
model.keras_model._make_predict_function()

# define class names
class_names = ['BG','Lung Opacity']

#=========================================

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

# make prediction on the uploaded image
def predict_on_image(uploaded_file):
    global model
 
    img = Image.open(uploaded_file)
    img = np.array(img)

    results = model.detect([img], verbose=0)
    fig, ax = plt.subplots(figsize=(16, 16)) 
    r = results[0]
    
    # visualize result and save to file
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], figsize=(16,16), ax=ax)
    fig.savefig('static/prediction.png',bbox_inches='tight') 
    plt.close(fig)  
    
    response = [] 
    for p, scr in zip(results[0]['class_ids'], results[0]['scores']):
        response.append({"class":class_names[p], "score":str(scr)})
    return response

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/', methods=['GET','POST'])
def upload_files():
     if request.method == "POST":
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
           
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                    file_ext != validate_image(uploaded_file.stream):
                return "Invalid image", 400
            
            response = predict_on_image(uploaded_file)
            print(response)
            session["response"] = response
            return render_template("prediction.html", jsonfile = session["response"])
           
     else:
        return render_template('index.html')


@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == "GET":
        return render_template("prediction.html", jsonfile = session["response"])
    else:
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                    file_ext != validate_image(uploaded_file.stream):
                return "Invalid image", 400
            
            response = predict_on_image(uploaded_file)
            session["response"] = response
        
        return render_template("prediction.html", jsonfile = session["response"])

#=========================================

if __name__=='__main__':
    app.run(debug=True)