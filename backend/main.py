from flask import Flask,request,jsonify
from torch_utils import transform_image,get_prediction
app=Flask(__name__)
ALLOWED_EXTENSIONS={'jpg','png','jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower()
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        file=request.files.get('file')
        if file is None or file.filename=="":
            return jsonify({'error':'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error':'file format not supported'})
        try:
            img_bytes=file.read()
            tensor=transform_image(img_bytes)
            print(f"the tensor is :{tensor}")
            prediction=get_prediction(tensor)
            if prediction is None:
                raise ValueError("Prediction returned None.")
            
            return jsonify({'prediction':prediction})
        except Exception as e:
            print(f"Error: {e}")  # Logs the error on the server side
            return jsonify({"error": str(e)}), 500
