
from flask import Flask, request, render_template, Response, jsonify, redirect, url_for
from numpy import random

import frame_processing
from camera import VideoCamera

app = Flask(__name__)
screen_size_in_inches = 0
window = []
number = 0
show_camera = True


# @app.route('/')
# def hello_world():  # put application's code here
#     return render_template("home-page.html")

@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    global screen_size_in_inches, window
    if request.method == 'POST':
        inches_input = int(request.form['number'])
        res_input = request.form['Resolution']
        window = [int(res_input.split('×')[0]), int(res_input.split('×')[1])]
        if isinstance(inches_input, int):
            screen_size_in_inches = inches_input
            return jsonify(isError=False,
                           statusCode=200), 200
        else:
            screen_size_in_inches = 0
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
    return render_template('Recording.html', rand_img=str(number)+'.jpg', width=window[0], height=window[1],
                           camera_margin=0 if show_camera else window[0])


@app.route('/result')
def result():
    frame_processing.state_values.recording_happening = not frame_processing.state_values.recording_happening
    frame_processing.stop_recording_to_file()
    return render_template('Result.html', heatmap=frame_processing.calculated_values.last_file_name+'.png',
                           rand_img=str(number)+'.jpg', width=window[0], height=window[1])


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
