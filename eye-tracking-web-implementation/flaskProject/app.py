import base64
import copy
import io
import os
import time
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
from numpy import random
from werkzeug.utils import secure_filename

import frame_processing

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')
screen_size_in_inches = 0
window = None
number = 0
show_camera = True
uploaded_file_name = ""
UPLOAD_FOLDER = 'C:\\Users\\themi\\Desktop\\Diplomatic\\Repository\\eye-tracking-thesis\\eye-tracking-web-implementation\\flaskProject\\static\\images\\'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
debug_image = []
debug_screen_image = None
debug_started = False
gaze_point = None


# @app.route('/')
# def hello_world():  # put application's code here
#     return render_template("home-page.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    global screen_size_in_inches, window, uploaded_file_name
    if request.method == 'POST':
        inches_input = int(request.form['number'])
        res_input = request.form['Resolution']
        window = [int(res_input.split('×')[0]), int(res_input.split('×')[1])]
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            # return redirect(request.url)
            return jsonify(isError=True,
                           message="No file part",
                           statusCode=400), 400
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            # flash('No selected file')
            # return redirect(request.url)
            return jsonify(isError=True,
                           message="No file part",
                           statusCode=400), 400
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #     return redirect(url_for('download_file', name=filename))
        if isinstance(inches_input, int) and file and allowed_file(file.filename):
            screen_size_in_inches = inches_input
            filename = secure_filename(file.filename)
            uploaded_file_name = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify(isError=False,
                           statusCode=200), 200
        else:
            screen_size_in_inches = 0
            uploaded_file_name = ""
            return jsonify(isError=True,
                           message="Please enter integer value",
                           statusCode=400), 400


@app.route('/calibration', methods=['POST', 'GET'])
def calibration():
    if screen_size_in_inches != 0:
        return render_template('Calibration.html', width=window[0], height=window[1],
                               camera_margin=30 if show_camera else window[0])
    else:
        return redirect(url_for('home'))


@app.route('/recording', methods=['POST', 'GET'])
def recording():
    global number
    frame_processing.state_values.recording_happening = not frame_processing.state_values.recording_happening
    frame_processing.start_recording_to_file()
    number = random.randint(1, 5)
    return render_template('Recording.html', rand_img=uploaded_file_name, width=window[0], height=window[1],
                           camera_margin=30 if show_camera else window[0])


@app.route('/result')
def result():
    frame_processing.state_values.recording_happening = not frame_processing.state_values.recording_happening
    frame_processing.stop_recording_to_file()
    return render_template('Result.html', heatmap=frame_processing.calculated_values.last_file_name + '_heatmap.png',
                           scanpath=frame_processing.calculated_values.last_file_name + '_scanpath.png',
                           fixation_map=frame_processing.calculated_values.last_file_name + '_fixation_map.png',
                           fixation_scan=frame_processing.calculated_values.last_file_name + '_fixation_scan.png',
                           rand_img=uploaded_file_name, width=window[0], height=window[1])


@app.route('/1')
def index():
    return render_template('index.html')


@app.route('/f1')
def f1():
    frame_processing.calibrate_pose_estimation_and_anchor_points()
    return "Nothing"


@app.route('/f2')
def f2():
    frame_processing.calibrate_offsets()
    return "Nothing"


@app.route('/f3')
def f3():
    frame_processing.calibrate_eyes_depth()
    frame_processing.calculate_face_distance_offset()
    return "Nothing"


@app.route('/f4')
def f4():
    frame_processing.calibrate_offsets()
    return "Nothing"


@app.route('/f5')
def f5():
    frame_processing.calculate_eye_correction_height_factor()
    return "Nothing"


@app.route('/f6')
def f6():
    frame_processing.calculate_eye_correction_width_factor()
    return "Nothing"


# def gen(camera):
#     while True:
#         img = camera.get_frame()
#         frame = frame_processing.process_frame(img, [0, 0, window[0], window[1]], screen_size_in_inches)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def moving_average(x):
    return np.mean(x)


@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)


global fps, prev_recv_time, cnt, fps_array
fps = 30
prev_recv_time = 0
cnt = 0
fps_array = [0]


@socketio.on('image')
def image(data_image):
    global fps, cnt, prev_recv_time, fps_array, debug_image, debug_started, gaze_point, debug_screen_image
    recv_time = time.time()
    text = 'FPS: ' + str(fps)
    frame = (readb64(data_image))
    frame = np.asarray(frame)
    # frame = cv2.flip(frame, 1)

    if window is not None:
        frame_in_bytes, frame, gaze_point = frame_processing.process_frame(frame, [0, 0, window[0], window[1]],
                                                                           screen_size_in_inches)
    debug_image = copy.copy(frame)
    if not debug_started:
        debug_started = True
        thread = Thread(target=threaded_function, args=(10,))
        thread.start()
        thread.join()
        print("thread finished...exiting")

    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

    # if debug_screen_image is not None:
    #     screen_encode = cv2.imencode('.jpeg', debug_screen_image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]
    #
    #     # base64 encode
    #     screen_string_data = base64.b64encode(screen_encode).decode('utf-8')
    #     b64_src = 'data:image/jpeg;base64,'
    #     screen_string_data = b64_src + screen_string_data
    #     emit('background_image', screen_string_data)

    fps = 1 / (recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)), 1)
    prev_recv_time = recv_time
    # print(fps_array)
    cnt += 1
    if cnt == 30:
        fps_array = [fps]
        cnt = 0


def threaded_function(arg):
    global debug_screen_image
    while True:
        cv2.namedWindow("debug")
        cv2.imshow('debug', debug_image)
        cv2.namedWindow("debug_screen")
        if window is not None:
            debug_screen_image = np.zeros((window[1], window[0], 3), dtype='uint8')
            debug_screen_image = cv2.rectangle(debug_screen_image, (0, 0), (window[0] - 1, window[1] - 1),
                                               (255, 255, 255), -1)
            debug_screen_image = cv2.circle(debug_screen_image, (int(window[0] / 2), int(window[1] / 2)), 20,
                                            (80, 80, 80), 2)
            debug_screen_image = cv2.circle(debug_screen_image, (int(window[0]), int(window[1] / 2)), 20,
                                            (80, 80, 80), 2)
            debug_screen_image = cv2.circle(debug_screen_image, (int(window[0] / 2), int(window[1])), 20,
                                            (80, 80, 80), 2)
            if gaze_point != (-10, -10) and gaze_point is not None:
                debug_screen_image = cv2.circle(debug_screen_image, gaze_point, 20,
                                                (70, 70, 200), 2)
            cv2.imshow('debug_screen', debug_screen_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
