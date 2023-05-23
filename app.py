from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import os
import uuid
from PIL import Image, ImageOps


app = Flask(__name__)
model = load_model('model.h5')

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
            prediction =  _predict(img)
            label = np.argmax(prediction)
            prediction = prediction.tolist()[0]
            if label == 0:
                message = "Benign, the cells are not yet cancerous, but they have the potential to become malignant. Consult the doctor"
            elif label == 1:
                message = "Malignant, tumors are cancerous. The cells can grow and spread to other parts of the body. Visit to the doctor as soon as possible"
            elif label == 2 and (prediction[0] >= 0.2 or prediction[1] >= 0.2):
                message = "You look Normal, but there's a small propability of having cancerous cells, let's check the doctor"                
            else:
                message = "YOU are in Normal condition. No need to worry"
            predictionsMessage = """
                P(Bengin) = {}
                P(Malignant) = {}
                P(Normal) = {}
            """.format(prediction[0], prediction[1], prediction[2])
            return render_template('predict.html', message = message, predictionsMessage=predictionsMessage, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"



    

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    data = np.ndarray(shape=(1, 128, 128, 1), dtype=np.float32)
    image = Image.open(filename)
    image_array = np.asarray(image)
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.NONE)
    image = image.convert('L')
    image_array = np.asarray(image)
    image_array = image_array.reshape(128, 128, 1)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    return data

def _predict(img):
    return model.predict(img)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)
