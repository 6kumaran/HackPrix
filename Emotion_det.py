import cv2
import pandas as pd

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
from tensorflow.keras.models import load_model

# Load a pre-trained emotion recognition model (this is just an example, you need a model file)
emotion_model = load_model('C:/Users/KUMARAN/OneDrive/Desktop/model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
def detect_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    predicted_emotion=""
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to the size expected by the model

        roi_gray = roi_gray.astype('float32') / 255  # Normalize the image
        roi_gray = roi_gray.reshape(1, 48, 48, 1)

        # Predict the emotion
        predictions = emotion_model.predict(roi_gray)
        max_index = predictions[0].argmax()
        predicted_emotion = emotion_labels[max_index]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame,predicted_emotion
Music_Player=pd.read_csv("C:/Users/KUMARAN/OneDrive/Desktop/Spotify_Music_data_to_identify_the_moods/data_moods.csv")
Music_Player=Music_Player[['name','artist','mood','popularity']]
Music_Player.head()
Music_Player["mood"].value_counts()
Play = Music_Player[Music_Player ['mood'] == 'Calm' ]
Play = Play.sort_values (by="popularity", ascending=False)
Play = Play [:5].reset_index(drop=True)
def Recommend_Songs (pred_class):
    if( pred_class == 'Disgust' ):
        Play = Music_Player [Music_Player ['mood'] == 'Sad' ]
        Play = Play.sort_values (by="popularity", ascending=False)
        Play = Play [:5].reset_index(drop=True)
        print(Play)
    if( pred_class == 'Happy' or pred_class == 'Sad' ):
        Play = Music_Player [Music_Player['mood']== 'Happy' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play =Play [:5].reset_index(drop=True)
        print(Play)
    if( pred_class == 'Fear' or pred_class == 'Angry' ):
        Play = Music_Player [Music_Player ['mood'] == 'Calm' ]
        Play = Play.sort_values (by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        print(Play)
    if( pred_class == 'Surprise' or pred_class == 'Neutral' ):
        Play = Music_Player[Music_Player ['mood'] == 'Energetic' ]
        Play = Play.sort_values (by="popularity", ascending=False)
        Play = Play [:5].reset_index(drop=True)
        print(Play)
def main():
    cap = cv2.VideoCapture(0)  # Capture video from webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame,emot = detect_emotions(frame)
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\nYou Listen to the Following Famous Songs to Lighten up Your Mood\n")
    Recommend_Songs(emot)

if __name__ == "__main__":
    main()
