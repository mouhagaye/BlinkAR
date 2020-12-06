from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np
import os


def create_app():
    app = Flask(__name__)

    # Initialize our ngrok settings into Flask
    app.config.from_mapping(
        BASE_URL="http://localhost:5000",
        USE_NGROK=os.environ.get("USE_NGROK", "False") == "True" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    )

    if app.config.get("ENV") == "development" and app.config["USE_NGROK"]:
        # pyngrok will only be installed, and should only ever be initialized, in a dev environment
        from pyngrok import ngrok

        # Get the dev server port (defaults to 5000 for Flask, can be overridden with `--port`
        # when starting the server
        port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 5000

        # Open a ngrok tunnel to the dev server
        public_url = ngrok.connect(port).public_url
        print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

        # Update any base URLs or webhooks to use the public ngrok URL
        app.config["BASE_URL"] = public_url
        init_webhooks(public_url)

    # ... Initialize Blueprints and the rest of our app

    return app

app = create_app()


video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
abdallah_image = face_recognition.load_image_file("abdallah.jpg")
abdallah_face_encoding = face_recognition.face_encodings(abdallah_image)[0]

# Load a second sample picture and learn how to recognize it.
gaye_image = face_recognition.load_image_file("gaye.jpg")
gaye_face_encoding = face_recognition.face_encodings(gaye_image)[0]

cima_image = face_recognition.load_image_file("cima.jpg")
cima_face_encoding = face_recognition.face_encodings(cima_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    abdallah_face_encoding,
    gaye_face_encoding,
    cima_face_encoding,

]

tagname = 'NOM :'
tagage = 'AGE :'
tagsexe = 'SEXE :'
taggroupe = 'GROUPE :'


known_face_names = [
    'Abdou Lahi Diop',
    'Mouhamadou Gaye',
    "Mamadou Ciss"
]
ages = [
    "23 ans",
    "23 ans",
    "19 ans"
]
sexes = [
    "masculin",
    "masculin",
    "masculin"
]
groupes = [
    'O+',
    'O-',
    'AB-'
]
# Initialize some variables
# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True


def gen_frames():  # generate frame by frame from camera
    age = 0
    groupe = ''
    sexe = ''
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while True:
        # Capture frame-by-frame
        success, frame = video_capture.read()  # read the camera frame
        if not success:
            break
        else:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                for i ,nom in enumerate(known_face_names):
                    if nom == name:
                        age = ages[i]
                        sexe = sexes[i]
                        groupe = groupes[i]


                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, tagname+name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                if name != 'Unknown':
                    cv2.putText(frame, tagage+age, (left + 6, bottom + 50), font, 1.0, (255, 255, 255), 1)
                    cv2.putText(frame, tagsexe+sexe, (left + 6, bottom + 100), font, 1.0, (255, 255, 255), 1)
                    cv2.putText(frame, taggroupe+groupe, (left + 6, bottom + 150), font, 1.0, (255, 255, 255), 1)


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    """Video streaming home page."""
    return render_template('stream.html')


if __name__ == '__main__':
    app.run()