import glob
import cv2
import numpy as np
import base64
import imutils
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load data template
template_data = []
template_files = glob.glob('template/*.jpg', recursive=True)
print("template loaded:", template_files)
# proses gambar dan masukkan ke dalam array template_data
for template_file in template_files:
    tmp = cv2.imread(template_file)
    tmp = imutils.resize(tmp, width=int(tmp.shape[1]*0.5))  # scalling
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)  # grayscale
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    tmp = cv2.filter2D(tmp, -1, kernel) #sharpening
    tmp = cv2.blur(tmp, (3, 3))  # smoothing
    tmp = cv2.Canny(tmp, 50, 200)  # Edge with Canny
    nominal = template_file.replace('template/', '').replace('.jpg', '')
    template_data.append({"glob":tmp, "nominal":nominal})


#API
@app.route('/', methods=['GET'])
def index():
    return 'Welcome to Uang Matching API!'

@app.route('/process_image', methods=['POST'])
def process_image():
	  # ambil gambar yang di post
    data = request.get_json()
    img_base64 = data['image']
    img_bytes = base64.b64decode(img_base64)
    npimg = np.fromstring(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # proses looping "data template"
    result_data = []
    for template in template_data:
        image_test_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_test_p = cv2.Canny(image_test_p, 50, 200)
        (tmp_height, tmp_width) = template['glob'].shape[:2]
        found = None
        thershold = 0.4
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # scalling uang
            resized = imutils.resize(
                image_test_p, width=int(image_test_p.shape[1] * scale))
            r = image_test_p.shape[1] / float(resized.shape[1])

            if resized.shape[0] < tmp_height or resized.shape[1] < tmp_width:
                break
            # proses template matching antara "data template" dengan data yang "di post(data test)""
            result = cv2.matchTemplate(resized, template['glob'], cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                if maxVal >= thershold:
                    image_result = {}
                    image_result["nominal"] = template['nominal']
                    image_result["match_score"] = maxVal
                    result_data.append(image_result)
    # proses debug untuk menampilkan output pada terminal/command prompt
    print("result:", result_data)

    # output
    return jsonify({"result": result_data})

if __name__ == '__main__':
    app.run(debug=True)
