from flask import Flask, request, render_template, Response, jsonify, redirect, url_for

import frame_processing
from camera import VideoCamera

app = Flask(__name__)
screen_size_in_inches = 0


# @app.route('/')
# def hello_world():  # put application's code here
#     return render_template("home-page.html")

@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    global screen_size_in_inches
    if request.method == 'POST':
        inches_input = int(request.form['number'])
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
        return render_template('Calibration.html')
    else:
        return redirect(url_for('home'))


@app.route('/recording')
def recording():
    return render_template('Recording.html')


@app.route('/result')
def result():
    return render_template('Result.html')


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
        frame = frame_processing.process_frame(img, [0, 0, img.shape[1], img.shape[0]], screen_size_in_inches)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
