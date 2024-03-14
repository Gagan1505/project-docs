from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
from keras.utils import img_to_array
from flask import Flask, render_template, request


app = Flask(__name__)



@app.route('/',methods=['POST','GET'])
def index():
    model = load_model('kavya_model.h5')
    def check(res):

          p1=["absent","present"]
          path=p1
          pred=model.predict(res)
          res=np.argmax(pred)
          res=path[res]
          return (res)


    def convert_img_to_tensor2(fpath):
            img = cv2.imread(fpath)
            img = cv2.resize(img,(256,256))
            res = img_to_array(img)
            res = np.array(res, dtype=np.float16)/ 255.0
            res = res.reshape(-1,256,256,3)
            res = res.reshape(1,256,256,3)
            return res

    if request.method == 'POST':
        img = request.files['img']
        img.save('static/example.jpg')
        res = convert_img_to_tensor2("static/example.jpg")
        msg = check(res)
        return render_template('kfront.html', res=msg)

    else:
        return render_template('kfront.html')


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")