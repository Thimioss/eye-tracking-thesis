from flask import Flask, request, render_template, Response

import frame_processing
from camera import VideoCamera

app = Flask(__name__)


# @app.route('/')
# def hello_world():  # put application's code here
#     return render_template("home-page.html")

@app.route('/')
def index():
    return render_template('index1.html')


def gen(camera):
    while True:
        img = camera.get_frame()
        frame = frame_processing.process_frame(img, [0, 0, img.shape[1], img.shape[0]])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
