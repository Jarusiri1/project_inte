import os
from flask import Flask, render_template, request
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename  # ป้องกันปัญหาชื่อไฟล์
import joblib


app = Flask(__name__)  

# ✅ โหลดโมเดล MNIST
MNIST_MODEL_PATH = "model4.h5"
mnist_model = None

if os.path.exists(MNIST_MODEL_PATH):
    mnist_model = load_model(MNIST_MODEL_PATH, compile=False)
    print("✅ MNIST Model loaded successfully!")
else:
    print("❌ MNIST Model not found! Make sure 'model4.h5' exists.")

# ✅ ฟังก์ชันพยากรณ์ตัวเลขจากภาพ
def predict_mnist(img_path):
    if mnist_model is None:
        return "❌ MNIST Model not loaded"

    # โหลดภาพและแปลงเป็น grayscale
    img = Image.open(img_path).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0  # Normalize 0-1
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape ให้ตรงกับ input model

    prediction = mnist_model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    return f"ผลลัพธ์: {predicted_digit}"

# ✅ อัปโหลดและพยากรณ์ภาพตัวเลข MNIST
UPLOAD_FOLDER = "static/mnist_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/mnist', methods=["GET", "POST"])
def mnist():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            result = predict_mnist(img_path)

    return render_template('mnist.html', result=result, img_path=img_path)

# ✅ โหลดโมเดลพยากรณ์เบาหวาน
MODEL_PATH_3 = "stack_model.pkl"
SCALER_PATH = "scaler.pkl"

stack_model = None
scaler = None

if os.path.exists(MODEL_PATH_3) and os.path.exists(SCALER_PATH):
    stack_model = joblib.load(MODEL_PATH_3)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Diabetes prediction model loaded successfully!")
else:
    print("❌ Model files not found! Make sure 'stack_model.pkl' and 'scaler.pkl' exist.")

# ✅ ฟังก์ชันสำหรับพยากรณ์เบาหวาน
def predict_diabetes(features):
    try:
        features = np.array(features, dtype=np.float32).reshape(1, -1)

        # ตรวจสอบว่า scaler โหลดมาหรือยัง
        if scaler is None or stack_model is None:
            return "❌ Model or Scaler not loaded"

        # ทำการสเกลข้อมูลก่อนส่งเข้าโมเดล
        features_scaled = scaler.transform(features)

        # พยากรณ์
        prediction = stack_model.predict(features_scaled)
        result = "เป็นเบาหวาน" if prediction[0] == 1 else "ไม่เป็นเบาหวาน"

        return result
    except Exception as e:
        print("Error during prediction:", e)
        return f"Error during prediction: {str(e)}"

# เพิ่มหน้าเว็บใหม่สำหรับพยากรณ์เบาหวาน
@app.route('/diabetes', methods=["GET", "POST"])
def diabetes():
    result = None
    if request.method == "POST":
        try:
            # รับค่าจากฟอร์ม
            pregnancies = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['blood_pressure'])
            skin_thickness = float(request.form['skin_thickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = float(request.form['age'])
            bmi_age_ratio = bmi / age  # คำนวณ feature ใหม่

            features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, bmi_age_ratio]
            result = predict_diabetes(features)
        except ValueError:
            return "❌ Error: Invalid input. Please enter valid numbers."

    return render_template('diabetes.html', result=result)

# ✅ ตรวจสอบและโหลดโมเดล
MODEL_PATH = "model.h5"
model = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
else:
    print("❌ Model not found! Make sure to train and save `model.h5`.")

# ✅ โหลดโมเดลทำนายราคาบ้าน
MODEL_PATH_2 = "model2.h5"
model3 = None

if os.path.exists(MODEL_PATH_2):
    model2 = load_model(MODEL_PATH_2, compile=False)
    print("✅ House price prediction model loaded successfully!")
else:
    print("❌ House price prediction model not found! Make sure to train and save `model3.h5`.")

# ✅ ฟังก์ชันสำหรับพยากรณ์รูปภาพ (หมา หรือ แมว)
class_names = ["Cat", "Dog"]

def predict_image(img_path):
    if model is None:
        return "❌ Model not loaded"

    img = Image.open(img_path).resize((160, 160))  # ปรับขนาด
    img_array = np.array(img) / 255.0  # Normalize 0-1
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] > 0.5)  # ใช้ threshold 0.5
    return class_names[predicted_class]

# ฟังก์ชันสำหรับพยากรณ์ราคา
def predict_house_price(features):
    try:
        # ตรวจสอบและแปลงข้อมูล
        features = np.array(features, dtype=np.float32)

        # ตรวจสอบให้แน่ใจว่า shape ของข้อมูลตรงตามที่โมเดลคาดหวัง
        features = features.reshape(1, -1)

        print("Features shape before prediction:", features.shape)

        # ทำการพยากรณ์
        prediction = model2.predict(features)

        # ผลลัพธ์การพยากรณ์
        predicted_price = prediction[0][0]
        print("Predicted house price:", predicted_price)
        return predicted_price
    except Exception as e:
        print("Error during prediction:", e)
        return f"Error during prediction: {str(e)}"


# ✅ กำหนดโฟลเดอร์สำหรับอัปโหลดรูป
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ เส้นทางของแต่ละหน้าเว็บ
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/explaindiabetes')
def explaindiabetes():
    return render_template('explaindiabetes.html')

@app.route('/explainmnist')
def explainmnist():
    return render_template('explainmnist.html')

@app.route('/page2', methods=["GET", "POST"])
def page2():
    predicted_price = None
    if request.method == "POST":
        try:
            house_size = float(request.form['house_size'])
            bedrooms = float(request.form['bedrooms'])
            bathrooms = float(request.form['bathrooms'])
            living_rooms = float(request.form['living_rooms'])
            floors = float(request.form['floors'])
            kitchens = float(request.form['kitchens'])
            garages = float(request.form['garages'])

            # ส่งข้อมูลไปพยากรณ์
            features = [house_size, bedrooms, bathrooms, living_rooms, floors, kitchens, garages, 0]  # garden = 0
            predicted_price = predict_house_price(features)
        except ValueError:
            return "Error: Invalid input. Please enter valid numbers for all fields."

    return render_template('page2.html', predicted_price=predicted_price)


# ✅ อัปโหลดและพยากรณ์รูปภาพ
@app.route('/page3', methods=["GET", "POST"])
def page3():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            result = predict_image(img_path)

    return render_template('page3.html', result=result, img_path=img_path)

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # ใช้ PORT จาก Render หรือ fallback เป็น 10000
    app.run(host='0.0.0.0', port=port)