# Eye tracking system using a simple recording device

These projects are part of a diploma thesis and allow for **webcam based eye tracking** with no additional equipment. This repository includes two projects. The first is a python project that uses our alorithm for **local eye tracking** and provides calibration, evaluation, output recording, and visualization capabilities. The second project is a flask python project that deploys the first project into a host and converts it into a web service for **remote eye tracking** through the user's webcam.


## Local application

For the local application it is required to have a camera in your system. The application begins with the calibration stage. There are numbered instructions and buttons on the screen to make the proccess clear and easy. When calibration is completed you can press the green button on the left (Calibration complete). If there was a fault in the calibration you can press the purple button to repeat the proccess (Restart calibration). To show or hide the diagnostic gaze points on the screen you can press the orange button on the left (Show/Hide diagnostics). To begin the evalutaion process you can press the yellow button (Start evaluation). After the evalutaion proccess the results are presented on the screen in numerical values. Lastly, to record the system's output you can press the red button (Start recording). A fullscreen image appears on the screen and your gaze upon it is recorded. When you click anywhere on the screen the recording proccess is finished and the results are saved in the project directory.


## Web service

In summary the web service project allows for the same functionality as the local application (except for the evaluation process and the diagnostic graphics) but through a browser. That way, it allows users to access our eye tracking system and record its output while observing their own uploaded images in their own browser, using their own webcam. Note that in its current form the service makes use of a local host and is not actually deployed in a server.
