
# ♻️ Smart Waste Bin – Real-Time Waste Classification Using Deep Learning

A smart AI-powered system that classifies waste in real-time using image recognition. This project uses a Convolutional Neural Network (MobileNetV2) and a user-friendly web interface (Streamlit) to help automate waste segregation into categories such as plastic, paper, metal, glass, cardboard, and general trash.

---

## 📌 Project Objective

This project addresses the growing problem of improper waste disposal by creating a real-time, image-based waste classification system. It aims to:
- Improve waste segregation at the source
- Support sustainable and smart city initiatives
- Provide a low-cost, scalable solution using machine learning

---

## 🧠 Technologies Used

| Component     | Technology         |
|---------------|--------------------|
| Programming   | Python             |
| Deep Learning | TensorFlow, Keras  |
| Model         | MobileNetV2 (Transfer Learning) |
| Interface     | Streamlit          |
| Dataset       | TrashNet (GitHub) or TACO Dataset |

---

## 📂 Project Structure

```
smart-waste-bin/
├── app/
│   ├── main.py           # Streamlit front-end
│   └── utils.py          # Helper functions (prediction)
├── model/
│   └── waste_classifier.h5  # Trained MobileNetV2 model
├── dataset/              # (Optional) Training data
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

---

## 🧪 Dataset

The TrashNet dataset is used in this project. It contains over 2,500 images in six categories:
- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

80% of the images are used for training, and 20% for testing and validation.

---

## ⚙️ How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-waste-bin.git
cd smart-waste-bin
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # For Windows
# OR
source venv/bin/activate  # For macOS/Linux
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Train the Model (optional)
If you want to train your own model:
```bash
python train_model.py
```

### 5. Run the Streamlit App
```bash
python -m pip install streamlit

streamlit run app/main.py
```

---

## 📈 Model Architecture

- Base: MobileNetV2 (pretrained on ImageNet)
- Layers added:
  - GlobalAveragePooling2D
  - Dense(128, ReLU)
  - Dense(6, softmax)

Loss Function: `categorical_crossentropy`  
Optimizer: `Adam`

---

## 📊 Sample Output

When a user uploads an image, the app will display the predicted waste class like this:

```
✅ Prediction: Paper
```

---

## 📌 Future Improvements

- Add webcam/live video classification
- Integrate with Raspberry Pi & sensors
- Expand dataset using TACO or custom images
- Multilingual support & mobile responsiveness
- Deploy online using Streamlit Cloud or Heroku

---

## 📚 References

- [TrashNet Dataset (GitHub)](https://github.com/garythung/trashnet)
- [TACO Dataset](https://tacodataset.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [Keras](https://keras.io/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---


