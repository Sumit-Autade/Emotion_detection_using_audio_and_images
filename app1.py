import streamlit as st 
import cv2
from PIL import Image, ImageEnhance
import numpy as np 
from my_model.model import FacialExpressionModel
import time
# this file is imported in main app it will run when images detection option is selected
# application for emotion detection using images using CNN 

# Importing the cnn model + using the CascadeClassifier to use features at once to check if a window is not a face region

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("my_model/model.json", "my_model/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX 

# Face expression detecting function
def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img, faces, pred

# The main function    
def app():
    
    """Face Expression Detection App"""
    # Setting the app title & sidebar
    activities = ["Home", "Detect your Facial expressions"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Home':
        st.subheader(":smile: :worried: :fearful: :rage: :hushed:")

    if choice == 'Detect your Facial expressions':
        st.title("Face Expression WEB Application :")
        st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        
        # If image is uploaded, display the progress bar + the image
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            st.image(our_image)

            task = ["Faces"]
            feature_choice = st.sidebar.selectbox("Find Features", task)
            if st.button("Process"):
                if feature_choice == 'Faces':
                    # Process bar
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.05)
                        progress.progress(i+1)
                    # End of process bar
                    result_img, result_faces, prediction = detect_faces(our_image)
                    if st.image(result_img):
                        st.success("Found {} faces".format(len(result_faces)))

                        if prediction == 'Happy':
                            st.subheader("YeeY! You are Happy :smile: today, Always Be!")
                        elif prediction == 'Angry':
                            st.subheader("You seem to be angry :rage: today, Take it easy!")
                        elif prediction == 'Disgust':
                            st.subheader("You seem to be Disgust :rage: today!")
                        elif prediction == 'Fear':
                            st.subheader("You seem to be Fearful :fearful: today, Be courageous!")
                        elif prediction == 'Neutral':
                            st.subheader("You seem to be Neutral today, Happy day!")
                        elif prediction == 'Sad':
                            st.subheader("You seem to be Sad :sad: today, Smile and be happy!")
                        elif prediction == 'Surprise':
                            st.subheader("You seem to be surprised today!")
                        else:
                            st.error("Sorry unable to detect")
                    else:
                        st.subheader('Unable to detect')
            if image_file is None:
                st.error("No image uploaded yet")

        # Face Detection
        

    elif choice == 'About':
        st.title("Face Expression WEB Application :")
        st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
        st.subheader("About Face Expression Detection App")

if __name__ == '__app__':
    app()
