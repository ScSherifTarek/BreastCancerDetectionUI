from flask import Flask, render_template, request
import numpy as np
# from keras.applications.vgg16 import preprocess_input
import os
import random
import uuid
# from keras.preprocessing import image

app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('index.html')

THRESHOLD = 0.2
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':      
        file = request.files['file']
        if file and allowed_file(file.filename):
            _, file_extension = os.path.splitext(file.filename)
            file_path = os.path.join('static/images', str(uuid.uuid4())+file_extension)
            file.save(file_path)
            img = read_image(file_path)
            notHealthyPercentage = _predict(img)
            isHealthy = True
            if notHealthyPercentage > THRESHOLD:
                isHealthy = False
            return render_template('predict.html', notHealthyPercentage = notHealthyPercentage, isHealthy=isHealthy, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"


#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    return ""
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def _predict(img):
    return random.uniform(0, 1)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)
