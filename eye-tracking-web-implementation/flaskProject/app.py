import os

from flask import Flask, request, render_template, Response, jsonify, redirect, url_for, flash
from numpy import random
from werkzeug.utils import secure_filename

import frame_processing
from camera import VideoCamera

app = Flask(__name__)
screen_size_in_inches = 0
window = []
number = 0
show_camera = True
uploaded_file_name = ""
UPLOAD_FOLDER = 'C:\\Users\\themi\\Desktop\\Diplomatic\\Repository\\eye-tracking-thesis\\eye-tracking-web-implementation\\flaskProject\\static\\images\\'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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


@app.route('/calibration')
def calibration():
    if screen_size_in_inches != 0:
        return render_template('Calibration.html', width=window[0], height=window[1],
                               camera_margin=0 if show_camera else window[0])
    else:
        return redirect(url_for('home'))


@app.route('/recording')
def recording():
    global number
    frame_processing.state_values.recording_happening = not frame_processing.state_values.recording_happening
    frame_processing.start_recording_to_file()
    number = random.randint(1, 5)
    return render_template('Recording.html', rand_img=uploaded_file_name, width=window[0], height=window[1],
                           camera_margin=0 if show_camera else window[0])


@app.route('/result')
def result():
    frame_processing.state_values.recording_happening = not frame_processing.state_values.recording_happening
    frame_processing.stop_recording_to_file()
    return render_template('Result.html', heatmap=frame_processing.calculated_values.last_file_name + '.png',
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


def gen(camera):
    while True:
        img = camera.get_frame()
        frame = frame_processing.process_frame(img, [0, 0, window[0], window[1]], screen_size_in_inches)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
