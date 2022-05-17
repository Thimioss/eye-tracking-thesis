import collections
import fractions

import cv2
import copy
import math
import mediapipe as mp
import numpy as np
import time
import pyautogui as gui

from calibration_values import CalibrationValues
from constants import Constants
from evaluation_data import EvaluationData

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

evaluation_data = EvaluationData()
constants = Constants()
calibration_values = CalibrationValues()

start_time = time.time()
display_time = 2
frames_counter = 0
fps = 0
angles = [0, 0, 0]

left_edge_norm = (0, 0, 0)
right_edge_norm = (0, 0, 0)
left_eye_norm = (0, 0, 0)
right_eye_norm = (0, 0, 0)
left_anchor_calibration_point_offset_norm = (0, 0, 0)
right_anchor_calibration_point_offset_norm = (0, 0, 0)
face_anchor_initial_points_2d = ((0, 0), (0, 0), (0, 0), (0, 0))
face_anchor_initial_points_3d = []
eyes_anchor_initial_points = ((0, 0), (0, 0))
forehead_chin_landmark_distance = 0
face_center_screen = (0, 0)
face_tilt_width_offset = 0
face_tilt_height_offset = 0
window = [0, 0, 1, 1]
left_eye_width_offset = 0
left_eye_height_offset = 0
right_eye_width_offset = 0
right_eye_height_offset = 0
face_distance = 60
x_angle, y_angle, z_angle = 0, 0, 0
x_cal, y_cal, z_cal = 0, 0, 0
scaled_face_vector = [1, 1]


def nothing(x):
    pass


def get_contour_from_landmark_indexes(indexes, img):
    i = 0
    cont = np.zeros((len(indexes), 2), dtype=int)
    for landmark_index in indexes:
        cont[i][0] = int(face_landmarks.landmark[landmark_index].x * img.shape[1])
        cont[i][1] = int(face_landmarks.landmark[landmark_index].y * img.shape[0])
        i = i + 1
    return cont


def mouse_event(event, x, y, flags, param):
    global calibration_values, evaluation_data
    if event is cv2.EVENT_LBUTTONDOWN and 20 <= x < 40 and int(window[3] / 2) - 10 <= y < int(window[3] / 2) + 10:
        if calibration_values.evaluation_happening:
            gui.alert("You cannot calibrate while evaluation is happening", "Error")
        else:
            calibration_values.calibration_completed = False
            reset_calibrations()
    elif event is cv2.EVENT_LBUTTONDOWN and 20 <= x < 40 and int(window[3] / 2) + 30 <= y < int(window[3] / 2) + 50:
        if calibration_values.evaluation_happening:
            gui.alert("Evaluation is happening", "Error")
        else:
            calibration_values.calibration_completed = True
    elif event is cv2.EVENT_LBUTTONDOWN and 20 <= x < 40 and int(window[3] / 2) + 70 <= y < int(window[3] / 2) + 90:
        if calibration_values.calibration_completed is False:
            gui.alert("You cannot start evaluation without completing calibration", "Error")
        elif calibration_values.evaluation_happening is True:
            pass
        else:
            calibration_values.evaluation_happening = True

    if calibration_values.calibration_completed is False:
        if event is cv2.EVENT_LBUTTONDOWN and int(2 * window[2] / 3) <= x < int(
                2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) and int(2 * window[3] / 3) <= y < int(
                2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)):
            calibrate_pose_estimation_and_anchor_points()
        elif event is cv2.EVENT_LBUTTONDOWN and int(
                2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) <= x < window[2] and int(
                2 * window[3] / 3) <= y < int(
                2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)):
            calibrate_eyes_depth()
            calculate_face_distance_offset()
        elif event is cv2.EVENT_MBUTTONDOWN:
            calibrate_offsets()
        elif event is cv2.EVENT_LBUTTONDOWN and int(2 * window[2] / 3) <= x < int(
                2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) and int(
                2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) <= y < window[3]:
            calculate_eye_correction_height_factor()
            # calculate_eyes_distance_offset()
        elif event is cv2.EVENT_LBUTTONDOWN and int(
                2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) <= x < window[2] and int(
                2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) <= y < window[3]:
            calculate_eye_correction_width_factor()
    else:
        pass


def reset_calibrations():
    global calibration_values
    # global face_height_on_60cm_away, face_position_correction_width, face_position_correction_height, \
    #     face_tilt_correction_width, face_tilt_correction_height, face_tilt_factor_width, face_tilt_factor_height, \
    #     left_eye_tilt_correction_width, left_eye_tilt_correction_height, right_eye_tilt_correction_width, \
    #     right_eye_tilt_correction_height, left_eye_tilt_factor_height, left_eye_tilt_factor_width, \
    #     right_eye_tilt_factor_height, right_eye_tilt_factor_width, left_eye_calibration_point_offset, \
    #     right_eye_calibration_point_offset, eyes_depth_offset, scaled_face_vector, x_off, y_off, z_off, \
    #     left_gaze_point_offset, right_gaze_point_offset, left_gaze_point_factor, right_gaze_point_factor, \
    #     face_anchor_initial_points_2d, eyes_anchor_initial_points, face_anchor_initial_points_3d, rvec_init, \
    #     tvec_init, face_point_correction, face_distance_offset, left_eye_point_correction, right_eye_point_correction, \
    #     left_eye_distance_offset, right_eye_distance_offset, calibration_completed
    calibration_values = CalibrationValues()


def get_calibrated_eye_depth_error(anchor_initial_points, keypoint, depth_offset, rec, tec, cmat, dmat):
    eyes_anchor_initial_points_cal_temp = np.float32([[anchor_initial_points[0][0],
                                                       anchor_initial_points[0][1],
                                                       anchor_initial_points[0][2] +
                                                       depth_offset],
                                                      [anchor_initial_points[1][0],
                                                       anchor_initial_points[1][1],
                                                       anchor_initial_points[1][2] +
                                                       depth_offset]])
    eyes_anchor_points_cal_temp = cv2.projectPoints(eyes_anchor_initial_points_cal_temp, rec, tec,
                                                    cmat, dmat)[0]
    return abs(keypoint[0] - eyes_anchor_points_cal_temp[0][0][0])


def calibrate_eyes_depth():
    global calibration_values, x_cal, y_cal, scaled_face_vector, eyes_anchor_initial_points, rot_vec, trans_vec, \
        cam_matrix, dist_matrix
    # eyes_depth_offset = 0
    # error_prev = 10000
    # error = get_calibrated_eye_depth_error(eyes_anchor_initial_points, keypoint_left, eyes_depth_offset, rot_vec,
    #                                        trans_vec, cam_matrix, dist_matrix)
    # while error < error_prev:
    #     eyes_depth_offset += 1
    #     error_prev = error
    #     error = get_calibrated_eye_depth_error(eyes_anchor_initial_points, keypoint_left, eyes_depth_offset, rot_vec,
    #                                            trans_vec, cam_matrix, dist_matrix)
    #
    # eyes_depth_offset -= 1
    calibration_values.eyes_depth_offset = (keypoint_left[0] - eyes_anchor_points[0][0]) / scaled_face_vector[0]


def calibrate_pose_estimation_and_anchor_points():
    global face_anchor_initial_points_2d, face_anchor_initial_points_3d, eyes_anchor_initial_points, nose_landmark
    face_anchor_initial_points_2d = face_2d
    face_anchor_initial_points_3d = face_3d
    face_anchor_initial_points_3d = move_origin_point(face_anchor_initial_points_3d, nose_landmark)
    eyes_anchor_initial_points = [(int(keypoint_left[0]), int(keypoint_left[1]), int(keypoint_left[2])),
                                  (int(keypoint_right[0]), int(keypoint_right[1]), int(keypoint_right[2]))]
    # eyes_anchor_initial_points = move_origin_point(eyes_anchor_initial_points, nose_landmark)


def move_origin_point(points, new_origin_point):
    for j, point_ in enumerate(points, start=0):
        points[j] = [point_[0] - new_origin_point[0], point_[1] - new_origin_point[1], point_[2] - new_origin_point[2]]
    return points


def calibrate_offsets():
    global calibration_values
    # global face_detected, face_height_on_60cm_away, face_position_correction_width, face_position_correction_height, \
    #     forehead_chin_landmark_distance, face_center_screen, window, x_angle, y_angle, x_off, y_off, z_off, z_angle, \
    #     face_point, face_center_screen_cal, face_point_correction, left_eye_point_correction, left_eye_point, \
    #     right_eye_point_correction, right_eye_point
    if face_detected:
        calibration_values.face_height_on_60cm_away = forehead_chin_landmark_distance
        calibration_values.face_position_correction_width = face_center_screen[0] - (window[2] / 2)
        calibration_values.face_position_correction_height = face_center_screen[1] - (window[3] / 2)
        calibration_values.face_point_correction = [screen.shape[1] / 2 - face_point[0],
                                                    screen.shape[0] / 2 - face_point[1]]
        # left_eye_point_correction = [screen.shape[1] / 2 - left_eye_point[0],
        #                              screen.shape[0] / 2 - left_eye_point[1]]
        # right_eye_point_correction = [screen.shape[1] / 2 - right_eye_point[0],
        #                               screen.shape[0] / 2 - right_eye_point[1]]
        calculate_eye_correction_offsets()
        calibration_values.x_off = -x_angle
        calibration_values.y_off = -y_angle
        calibration_values.z_off = -z_angle


def calculate_eye_correction_offsets():
    global calibration_values, left_gaze_point, right_gaze_point
    calibration_values.left_gaze_point_offset = [-left_gaze_point[0], -left_gaze_point[1]]
    calibration_values.right_gaze_point_offset = [-right_gaze_point[0], -right_gaze_point[1]]


def calculate_eye_correction_width_factor():
    # global left_gaze_point_factor, right_gaze_point_factor, left_gaze_point_cal, right_gaze_point_cal
    calibration_values.left_gaze_point_factor[0] = (screen.shape[1] / 2) / left_gaze_point_cal[0]
    calibration_values.right_gaze_point_factor[0] = (screen.shape[1] / 2) / right_gaze_point_cal[0]


def calculate_eye_correction_height_factor():
    # global left_gaze_point_factor, right_gaze_point_factor, left_gaze_point_cal, right_gaze_point_cal
    calibration_values.left_gaze_point_factor[1] = (screen.shape[0] / 2) / left_gaze_point_cal[1]
    calibration_values.right_gaze_point_factor[1] = (screen.shape[0] / 2) / right_gaze_point_cal[1]


def calculate_face_distance_offset():
    # global face_distance_offset, face_vector, face_distance
    calibration_values.face_distance_offset = ((screen.shape[1] - face_center_screen_cal[0] -
                                                calibration_values.face_point_correction[0]) * face_vector[2] /
                                               face_vector[0]) - face_distance


def calculate_eyes_distance_offset():
    # global left_eye_distance_offset, left_eye_vector, face_distance, left_eye_center_screen_cal, \
    #     right_eye_distance_offset, right_eye_vector, right_eye_center_screen_cal
    global keypoint_right, keypoint_left, face_distance, left_eye_center_screen_cal, right_eye_center_screen_cal, \
        left_eye_vector, right_eye_vector
    calibration_values.left_eye_distance_offset = ((screen.shape[1] - left_eye_center_screen_cal[0] -
                                                    calibration_values.left_eye_point_correction[0]) *
                                                   left_eye_vector[2] /
                                                   left_eye_vector[0]) - face_distance - keypoint_left[2]
    calibration_values.right_eye_distance_offset = ((screen.shape[1] - right_eye_center_screen_cal[0]
                                                     - calibration_values.right_eye_point_correction[0]) *
                                                    right_eye_vector[2] /
                                                    right_eye_vector[0]) - face_distance - keypoint_right[2]


def calculate_eyes_vectors(o, k_l, k_r, edo):
    global face_distance
    if edo == 0:
        return [[0, 0, 1], [0, 0, 1]]
    else:
        x_left = (k_l[0] - o[0][0]) / (edo * 60 / face_distance)
        y_left = (k_l[1] - o[0][1]) / (edo * 60 / face_distance)

        if x_left > 1:
            x_left = 1
        if y_left > 1:
            y_left = 1
        if x_left < -1:
            x_left = -1
        if y_left < -1:
            y_left = -1

        angle_x_l = math.asin(x_left)

        if math.cos(angle_x_l) == 0:
            angle_x_l += 0.01

        temp_left = y_left / math.cos(angle_x_l)
        if temp_left > 1:
            temp_left = 1
        if temp_left < -1:
            temp_left = -1

        angle_y_l = math.asin(temp_left)

        z_left = math.cos(angle_y_l) * math.cos(angle_x_l)

        x_right = (k_r[0] - o[1][0]) / (edo * 60 / face_distance)
        y_right = (k_r[1] - o[1][1]) / (edo * 60 / face_distance)

        if x_right > 1:
            x_right = 1
        if y_right > 1:
            y_right = 1
        if x_right < -1:
            x_right = -1
        if y_right < -1:
            y_right = -1

        angle_x_r = math.asin(x_right)

        if math.cos(angle_x_r) == 0:
            angle_x_r += 0.01

        temp_right = y_right / math.cos(angle_x_r)
        if temp_right > 1:
            temp_right = 1
        if temp_right < -1:
            temp_right = -1

        angle_y_r = math.asin(temp_right)

        z_right = math.cos(angle_y_r) * math.cos(angle_x_r)

        return [[-x_left, -y_left, z_left], [-x_right, -y_right, z_right]]


def show_text(img, string, x_, y_):
    img = cv2.putText(img, string, (x_, y_),
                      cv2.FONT_HERSHEY_PLAIN, 1,
                      (255, 255, 255), 0)


def show_calibration_values(img):
    pass
    # show_text(img, "face_detected: " + str(face_detected), 10, 30)
    # show_text(img, "calibration_completed: " + str(calibration_completed), 10, 70)
    # show_text(img, "face_height: " + str(face_height_on_60cm_away), 10, 90)
    # show_text(img, "face_position_correction_width: " + str(face_position_correction_width), 10, 110)
    # show_text(img, "face_position_correction_height: " + str(face_position_correction_height), 10, 130)
    # show_text(img, "face_tilt_correction_width: " + str(face_tilt_correction_width), 10, 150)
    # show_text(img, "face_tilt_correction_height: " + str(face_tilt_correction_height), 10, 170)
    # show_text(img, "face_tilt_factor_width: " + str(face_tilt_factor_width), 10, 190)
    # show_text(img, "face_tilt_factor_height: " + str(face_tilt_factor_height), 10, 210)
    # show_text(img, "left_eye_tilt_correction_width: " + str(left_eye_tilt_correction_width), 10, 230)
    # show_text(img, "left_eye_tilt_correction_height: " + str(left_eye_tilt_correction_height), 10, 250)
    # show_text(img, "left_eye_tilt_factor_height: " + str(left_eye_tilt_factor_height), 10, 270)
    # show_text(img, "left_eye_tilt_factor_width: " + str(left_eye_tilt_factor_width), 10, 290)
    # show_text(img, "right_eye_tilt_correction_width: " + str(right_eye_tilt_correction_width), 460, 230)
    # show_text(img, "right_eye_tilt_correction_height: " + str(right_eye_tilt_correction_height), 460, 250)
    # show_text(img, "right_eye_tilt_factor_height: " + str(right_eye_tilt_factor_height), 460, 270)
    # show_text(img, "right_eye_tilt_factor_width: " + str(right_eye_tilt_factor_width), 460, 290)
    # show_text(img, "forehead_chin_landmark_distance: " + str(forehead_chin_landmark_distance), 10, 310)
    # show_text(img, "face_center_screen: " + str(face_center_screen), 10, 330)
    # show_text(img, "face_tilt_width_offset: " + str(face_tilt_width_offset), 10, 350)
    # show_text(img, "face_tilt_height_offset: " + str(face_tilt_height_offset), 10, 370)
    # show_text(img, "window: " + str(window), 10, 390)
    # show_text(img, "left_eye_width_offset: " + str(left_eye_width_offset), 10, 410)
    # show_text(img, "left_eye_height_offset: " + str(left_eye_height_offset), 10, 430)
    # show_text(img, "right_eye_width_offset: " + str(right_eye_width_offset), 460, 410)
    # show_text(img, "right_eye_height_offset: " + str(right_eye_height_offset), 460, 430)
    # show_text(img, "face_distance: " + str(face_distance), 10, 450)
    #
    # show_text(img, "face_tilt_factor_height: " + str(face_tilt_factor_height), int(2 * window[2] / 3) + 50,
    #           int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) - 50)
    # show_text(img, "face_tilt_factor_width: " + str(face_tilt_factor_width),
    #           int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
    #           int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) - 50)
    # show_text(img, "left_eye_tilt_factor_height: " + str(left_eye_tilt_factor_height), int(2 * window[2] / 3) + 50,
    #           int(window[3]) - 50)
    # show_text(img, "left_eye_tilt_factor_width: " + str(left_eye_tilt_factor_width),
    #           int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
    #           int(window[3]) - 50)
    # show_text(img, "right_eye_tilt_factor_height: " + str(right_eye_tilt_factor_height), int(2 * window[2] / 3) + 50,
    #           int(window[3]) - 25)
    # show_text(img, "right_eye_tilt_factor_width: " + str(right_eye_tilt_factor_width),
    #           int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
    #           int(window[3]) - 25)


def show_ui(img):
    global calibration_values
    show_text(img, "Restart calibration", 50, int(window[3] / 2))
    img = cv2.rectangle(img, (20, int(window[3] / 2) - 10), (40, int(window[3] / 2) + 10),
                        (0, 0, 200), -1)
    show_text(img, "Calibration complete", 50, int(window[3] / 2) + 40)
    img = cv2.rectangle(img, (20, int(window[3] / 2) + 30), (40, int(window[3] / 2) + 50),
                        (0, 200, 0), -1)

    show_text(img, "Start evaluation", 50, int(window[3] / 2) + 80)
    img = cv2.rectangle(img, (20, int(window[3] / 2) + 70), (40, int(window[3] / 2) + 90),
                        (0, 200, 200), -1)

    # show_text(img, "x: " + str(angles[0]), 20, window[3] - 190)
    # show_text(img, "y: " + str(angles[1]), 20, window[3] - 160)
    # show_text(img, "z: " + str(angles[2]), 20, window[3] - 130)
    #
    # show_text(img, "x_cal: " + str(fractions.Fraction(x_cal / math.pi).limit_denominator(10)) + "pi", 20,
    #           window[3] - 100)
    # show_text(img, "y_cal: " + str(fractions.Fraction(y_cal / math.pi).limit_denominator(10)) + "pi", 20,
    #           window[3] - 70)
    # show_text(img, "z_cal: " + str(fractions.Fraction(z_cal / math.pi).limit_denominator(10)) + "pi", 20,
    #           window[3] - 40)

    if calibration_values.calibration_completed is False:
        show_calibration_ui(img)

    if calibration_values.evaluation_happening:
        if evaluation_data.get_completed_stages_count() == 0:
            show_ideal_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 1:
            show_edge_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 2:
            show_dark_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 3:
            show_turn_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 4:
            calibration_values.evaluation_happening = False


def show_dark_evaluation_ui(img):
    show_text(img, "Dark conditions evaluation", int(window[2]/2)-100, 20)

    img = cv2.circle(img, (int(window[2] / 2) + 400, int(window[3] / 2)), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2) + 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) + 400, int(window[3] / 2) + 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) - 400, int(window[3] / 2)), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2) - 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) - 400, int(window[3] / 2) - 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) - 400, int(window[3] / 2) + 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) + 400, int(window[3] / 2) - 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2)), 20,
                     (200, 200, 200), 2)


def show_turn_evaluation_ui(img):
    show_text(img, "Various head pose conditions evaluation", int(window[2] / 2) - 100, 20)

    img = cv2.circle(img, (int(window[2] / 2) + 400, int(window[3] / 2)), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2) + 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) + 400, int(window[3] / 2) + 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) - 400, int(window[3] / 2)), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2) - 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) - 400, int(window[3] / 2) - 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) - 400, int(window[3] / 2) + 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2) + 400, int(window[3] / 2) - 400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2)), 20,
                     (200, 200, 200), 2)


def show_ideal_evaluation_ui(img):
    show_text(img, "Ideal conditions evaluation", int(window[2] / 2) - 100, 20)

    img = cv2.circle(img, (int(window[2]/2)+400, int(window[3] / 2)), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2)+400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2)+400, int(window[3] / 2)+400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2)-400, int(window[3] / 2)), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2)-400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2)-400, int(window[3] / 2)-400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2)-400, int(window[3] / 2)+400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2)+400, int(window[3] / 2)-400), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2)), 20,
                     (200, 200, 200), 2)


def show_edge_evaluation_ui(img):
    show_text(img, "Edge conditions evaluation", int(window[2] / 2) - 100, 20)

    img = cv2.circle(img, (100, 100), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2])-100, int(window[3]) - 100), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (100, int(window[3]) - 100), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2]) - 100, 100), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (300, 300), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2]) - 300, int(window[3]) - 300), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (300, int(window[3]) - 300), 20,
                     (200, 200, 200), 2)
    img = cv2.circle(img, (int(window[2]) - 300, 300), 20,
                     (200, 200, 200), 2)


def show_calibration_ui(img):
    img = cv2.circle(img, (window[2], int(window[3] / 2)), 20,
                     (70, 200, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), window[3]), 20,
                     (200, 70, 200), 2)
    img = cv2.circle(img, (int(window[2] / 2), int(window[3] / 2)), 20,
                     (200, 200, 200), 2)
    img = cv2.putText(img, "Stand 60cm away from screen and keep",
                      (int(img.shape[1] / 2) - 250, int(img.shape[0] / 2) - 200),
                      cv2.FONT_HERSHEY_PLAIN, 1.5,
                      (150, 150, 150), 2)
    img = cv2.putText(img, "your head still for calibration",
                      (int(img.shape[1] / 2) - 200, int(img.shape[0] / 2 + 30) - 200),
                      cv2.FONT_HERSHEY_PLAIN, 1.5,
                      (150, 150, 150), 2)
    img = cv2.rectangle(img, (int(2 * window[2] / 3), int(2 * window[3] / 3)), (int(window[2]), int(window[3])),
                        (200, 100, 70), -1)

    img = cv2.line(img, (int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)), int(2 * window[3] / 3)),
                   (int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)), int(window[3])),
                   (70, 70, 70), 2)
    img = cv2.line(img, (int(2 * window[2] / 3), int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2))),
                   (int(window[2]), int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2))),
                   (70, 70, 70), 2)

    show_text(img, "3",
              int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
              int(2 * window[3] / 3) + 30)
    show_text(img, "Face the right edge of the screen,",
              int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
              int(2 * window[3] / 3) + 50)
    show_text(img, "look into the camera and left click here",
              int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
              int(2 * window[3] / 3) + 70)

    show_text(img, "1", int(2 * window[2] / 3) + 50,
              int(2 * window[3] / 3) + 30)
    show_text(img, "Face and look into the camera", int(2 * window[2] / 3) + 50,
              int(2 * window[3] / 3) + 50)
    show_text(img, "and left click here",
              int(2 * window[2] / 3) + 50,
              int(2 * window[3] / 3) + 70)

    show_text(img, "5", int(2 * window[2] / 3) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 30)
    show_text(img, "Face the center of the", int(2 * window[2] / 3) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 50)
    show_text(img, "screen (white point),",
              int(2 * window[2] / 3) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 70)
    show_text(img, "look at the bottom edge (pink point)",
              int(2 * window[2] / 3) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 90)
    show_text(img, "and left click here",
              int(2 * window[2] / 3) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 110)

    show_text(img, "6", int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 30)
    show_text(img, "Face the center of the", int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 50)
    show_text(img, "screen (white point),",
              int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 70)
    show_text(img, "look at the right edge (yellow point)",
              int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 90)
    show_text(img, "and left click here",
              int(2 * window[2] / 3 + ((window[2] - 2 * window[2] / 3) / 2)) + 50,
              int(2 * window[3] / 3 + ((window[3] - 2 * window[3] / 3) / 2)) + 110)

    show_text(img, "2, 4", int(window[2] / 2) - 170, int(window[3] / 2) + 50)
    show_text(img, "Face and look at the middle of the", int(window[2] / 2) - 170, int(window[3] / 2) + 70)
    show_text(img, "screen (white point) and middle click here", int(window[2] / 2) - 200, int(window[3] / 2) + 90)


def show_fps(img):
    global frames_counter, fps, start_time, display_time
    frames_counter += 1
    time_duration = time.time() - start_time
    if time_duration >= display_time:
        fps = frames_counter / time_duration
        frames_counter = 0
        start_time = time.time()
    cv2.putText(img, "FPS: " + str(fps), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# initiate screen interface
cv2.namedWindow('screen', cv2.WINDOW_FREERATIO)
cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# initiate webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow('image')
cv2.createTrackbar('smoothing_past_values_count', 'image', 0, 15, nothing)
cv2.createTrackbar('smoothing_landmarks_count', 'image', 0, 15, nothing)

# calibration with mouse event
cv2.setMouseCallback('screen', mouse_event)

# average smoothing arrays
offset_history = np.array(
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
     [0, 0]])

landmarks_history = np.zeros([10, 4, 3])


def calculate_iris_points(f_l, img):
    left_iris_landmark = f_l.landmark[constants.left_iris_index]
    k_l = [left_iris_landmark.x * img.shape[1],
           left_iris_landmark.y * img.shape[0],
           left_iris_landmark.z * img.shape[1]]
    img = cv2.circle(img, (int(k_l[0]), int(k_l[1])), 1, (0, 0, 255), 1)

    right_iris_landmark = f_l.landmark[constants.right_iris_index]
    k_r = [right_iris_landmark.x * img.shape[1],
           right_iris_landmark.y * img.shape[0],
           right_iris_landmark.z * img.shape[1]]
    img = cv2.circle(img, (int(k_r[0]), int(k_r[1])), 1, (0, 255, 0), 1)
    return k_l, k_r


def calculate_face_distance(f_l, img):
    global forehead_chin_landmark_distance
    chin_landmark = (f_l.landmark[constants.chin_landmark_index].x * img.shape[1],
                     f_l.landmark[constants.chin_landmark_index].y * img.shape[0],
                     f_l.landmark[constants.chin_landmark_index].z * img.shape[1])
    forehead_landmark = (f_l.landmark[constants.forehead_landmark_index].x * img.shape[1],
                         f_l.landmark[constants.forehead_landmark_index].y * img.shape[0],
                         f_l.landmark[constants.forehead_landmark_index].z * img.shape[1])
    forehead_chin_landmark_distance = np.sqrt(
        pow(chin_landmark[0] - forehead_landmark[0], 2) +
        pow(chin_landmark[1] - forehead_landmark[1], 2) +
        pow(chin_landmark[2] - forehead_landmark[2], 2))

    f_d = calibration_values.face_height_on_60cm_away * 60 / forehead_chin_landmark_distance
    return f_d


def calculate_face_center_screen_cal(img, wndw):
    face_cont = get_contour_from_landmark_indexes(constants.face_edge_landmarks_indexes, img)
    face_moment = cv2.moments(face_cont)
    face_center_image = (
        int(face_moment["m10"] / face_moment["m00"]), int(face_moment["m01"] / face_moment["m00"]))
    img = cv2.circle(img, face_center_image, 3, (200, 200, 200), 1)

    # face_center_image_offset = (face_center_image[0] - img.shape[1] / 2,
    #                             face_center_image[1] - img.shape[0] / 2)
    f_c_s = (wndw[2] * face_center_image[0] / img.shape[1],
             wndw[3] * face_center_image[1] / img.shape[0])
    f_c_s_cal = (f_c_s[0] - calibration_values.face_position_correction_width,
                 f_c_s[1] - calibration_values.face_position_correction_height)
    return f_c_s_cal


with mp_face_mesh.FaceMesh(max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.99) as face_mesh:
    while cap.isOpened():
        window = cv2.getWindowImageRect('screen')
        screen = np.zeros((window[3], window[2], 3), dtype='uint8')
        screen = cv2.rectangle(screen, (0, 0), (window[2] - 1, window[3] - 1), (85, 80, 78), -1)

        success, image = cap.read()
        pure_image = copy.copy(image)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        smoothing_past_values_count = cv2.getTrackbarPos('smoothing_past_values_count', 'image')
        smoothing_landmarks_count = cv2.getTrackbarPos('smoothing_landmarks_count', 'image')
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # get face mesh results
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        show_ui(screen)

        show_fps(image)

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:

                # show all indexes
                # i = 0
                # for landmark in face_landmarks.landmark:
                #     image = cv2.putText(image, str(i),
                #                         (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])),
                #                         cv2.FONT_HERSHEY_PLAIN, 0.7,
                #                         (0, 255, 255), 0)
                #     i += 1

                # iris points
                keypoint_left, keypoint_right = calculate_iris_points(face_landmarks, image)

                # face distance
                face_distance = calculate_face_distance(face_landmarks, image)

                # face position
                face_center_screen_cal = calculate_face_center_screen_cal(image, window)

                # get face orientation
                nose_landmark = (int(face_landmarks.landmark[constants.nose_landmark_index].x * image.shape[1]),
                                 int(face_landmarks.landmark[constants.nose_landmark_index].y * image.shape[0]),
                                 int(face_landmarks.landmark[constants.nose_landmark_index].z * image.shape[1]))
                face_anchors_3d = []
                face_anchors_2d = []
                for face_anchors_landmarks_index in constants.face_anchors_landmarks_indexes:
                    x, y, z = int(face_landmarks.landmark[face_anchors_landmarks_index].x * image.shape[1]), \
                              int(face_landmarks.landmark[face_anchors_landmarks_index].y * image.shape[0]), \
                              int(face_landmarks.landmark[face_anchors_landmarks_index].z * image.shape[1])
                    face_anchors_3d.append([x, y, z])
                    face_anchors_2d.append([x, y])

                face_3d = np.array(face_anchors_3d, dtype=np.float64)
                face_2d = np.array(face_anchors_2d, dtype=np.float64)

                for point in face_2d:
                    image = cv2.circle(image, (int(point[0]), int(point[1])),
                                       1, (255, 255, 255), 1)

                focal_length = 1 * image.shape[1]

                cam_matrix = np.array([[focal_length, 0, image.shape[0] / 2],
                                       [0, focal_length, image.shape[1] / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                face_anchor_initial_points_3d = np.array(face_anchor_initial_points_3d, np.float32)
                if np.any(face_anchor_initial_points_3d):
                    # Solve PnP
                    if np.any(calibration_values.rvec_init) and np.any(calibration_values.tvec_init):
                        success, rot_vec, trans_vec = cv2.solvePnP(face_anchor_initial_points_3d, face_2d, cam_matrix,
                                                                   dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE,
                                                                   useExtrinsicGuess=True,
                                                                   rvec=calibration_values.rvec_init,
                                                                   tvec=calibration_values.tvec_init)
                    else:
                        success, rot_vec, trans_vec = cv2.solvePnP(face_anchor_initial_points_3d, face_2d, cam_matrix,
                                                                   dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
                        if trans_vec[2] > 0:
                            calibration_values.rvec_init = rot_vec
                            calibration_values.tvec_init = trans_vec

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Convert to radians
                    x_angle = angles[0] * np.pi / 180
                    y_angle = angles[1] * np.pi / 180
                    z_angle = angles[2] * np.pi / 180

                    # + x_off and so on
                    x_cal = x_angle + calibration_values.x_off
                    y_cal = y_angle + calibration_values.y_off
                    z_cal = z_angle + calibration_values.z_off

                    face_vector = [
                        (math.cos(z_cal) * math.sin(y_cal) * math.cos(x_cal) + math.sin(z_cal) * math.sin(
                            x_cal)),
                        (math.sin(z_cal) * math.sin(y_cal) * math.cos(x_cal) - math.cos(z_cal) * math.sin(
                            x_cal)),
                        (math.cos(y_cal) * math.cos(x_cal))]

                    scaled_face_vector = [
                        (math.cos(z_cal) * math.sin(y_cal) * math.cos(x_cal) + math.sin(z_cal) * math.sin(
                            x_cal)) * 60 / face_distance,
                        (math.sin(z_cal) * math.sin(y_cal) * math.cos(x_cal) - math.cos(z_cal) * math.sin(
                            x_cal)) * 60 / face_distance]

                    axisBoxes = np.float32([[0, 0, 0],
                                            [50, 0, 0],
                                            [0, 50, 0],
                                            [0, 0, 50]])

                    axis_3d_projection = cv2.projectPoints(axisBoxes, rot_vec, trans_vec,
                                                           cam_matrix, dist_matrix)[0]

                    # p11 = (int(nose_landmark[0]), int(nose_landmark[1]))
                    # p22 = (int(nose_landmark[0] + face_vector[0] * 50 * 60 / face_distance),
                    #        int(nose_landmark[1] + face_vector[1] * 50 * 60 / face_distance))
                    #
                    # cv2.line(image, p11, p22, (255, 255, 255), 3)

                    if axis_3d_projection is not None and math.isnan(axis_3d_projection[0][0][0]) is False:
                        p1 = (int(axis_3d_projection[0][0][0]), int(axis_3d_projection[0][0][1]))
                        p2 = (int(axis_3d_projection[1][0][0]), int(axis_3d_projection[1][0][1]))
                        p3 = (int(axis_3d_projection[2][0][0]), int(axis_3d_projection[2][0][1]))
                        p4 = (int(axis_3d_projection[3][0][0]), int(axis_3d_projection[3][0][1]))

                        cv2.line(image, p1, p4, (255, 0, 0), 3)
                        cv2.line(image, p1, p2, (0, 0, 255), 3)
                        cv2.line(image, p1, p3, (0, 255, 0), 3)

                        show_text(image, "x", p2[0], p2[1])
                        show_text(image, "y", p3[0], p3[1])
                        show_text(image, "z", p4[0], p4[1])

                    face_reprojection = cv2.projectPoints(face_anchor_initial_points_3d, rot_vec, trans_vec,
                                                          cam_matrix, dist_matrix)[0]
                    for point in face_reprojection:
                        image = cv2.circle(image, (int(point[0][0]), int(point[0][1])), 1, (0, 255, 255))

                    face_direction_offset = [((face_distance + calibration_values.face_distance_offset) * face_vector[0]) / face_vector[2],
                                             ((face_distance + calibration_values.face_distance_offset) * face_vector[1]) / face_vector[2]]

                    face_point = [face_center_screen_cal[0] + face_direction_offset[0],
                                  face_center_screen_cal[1] + face_direction_offset[1]]

                    face_point_cal = [face_point[0] + calibration_values.face_point_correction[0],
                                      face_point[1] + calibration_values.face_point_correction[1]]

                    # eye anchor points
                    # face_anchor_points = face_2d
                    # face_anchor_initial_points_2d = np.array(face_anchor_initial_points_2d, np.float32)
                    # eyes_anchor_initial_points = np.array(eyes_anchor_initial_points, np.float32)
                    # eyes_anchor_initial_points_cal = np.float32([[eyes_anchor_initial_points[0][0],
                    #                                               eyes_anchor_initial_points[0][1],
                    #                                               eyes_anchor_initial_points[0][2] +
                    #                                               eyes_depth_offset],
                    #                                             [eyes_anchor_initial_points[1][0],
                    #                                              eyes_anchor_initial_points[1][1],
                    #                                              eyes_anchor_initial_points[1][2] +
                    #                                              eyes_depth_offset]])
                    #
                    # eyes_anchor_points = cv2.projectPoints(eyes_anchor_initial_points, rot_vec, trans_vec,
                    #                                        cam_matrix, dist_matrix)[0]
                    #
                    # eyes_anchor_points_cal = cv2.projectPoints(eyes_anchor_initial_points_cal, rot_vec, trans_vec,
                    #                                            cam_matrix, dist_matrix)[0]

                    # image = cv2.circle(image, (int(eyes_anchor_points[0][0][0]), int(eyes_anchor_points[0][0][1])), 2,
                    #                    (0, 255, 255), 1)
                    # image = cv2.circle(image, (int(eyes_anchor_points[1][0][0]), int(eyes_anchor_points[1][0][1])), 2,
                    #                    (0, 255, 255), 1)
                    # image = cv2.circle(image, (int(eyes_anchor_points_cal[0][0][0]), int(eyes_anchor_points_cal[0][0][1])), 2,
                    #                    (0, 255, 255), 1)
                    # image = cv2.circle(image, (int(eyes_anchor_points_cal[1][0][0]), int(eyes_anchor_points_cal[1][0][1])), 2,
                    #                    (0, 255, 255), 1)

                    face_anchor_initial_points_2d = np.array(face_anchor_initial_points_2d, np.float32)
                    face_anchor_points = np.array(face_2d, np.float32)
                    h, status = cv2.findHomography(face_anchor_initial_points_2d[0:4], face_anchor_points[0:4],
                                                   method=cv2.RANSAC,
                                                   ransacReprojThreshold=1, mask=None, maxIters=1, confidence=1)
                    if h is not None:
                        eyes_anchor_initial_points = np.array(eyes_anchor_initial_points, np.float32)
                        eyes_anchor_points = [np.dot(h, [eyes_anchor_initial_points[0][0],
                                                         eyes_anchor_initial_points[0][1], 1]),
                                              np.dot(h, [eyes_anchor_initial_points[1][0],
                                                         eyes_anchor_initial_points[1][1], 1])]
                        eyes_anchor_points[0] /= eyes_anchor_points[0][2]
                        eyes_anchor_points[1] /= eyes_anchor_points[1][2]

                        eyes_anchor_points_cal = [[0, 0], [0, 0]]
                        eyes_anchor_points_cal[0][0] = eyes_anchor_points[0][0] + scaled_face_vector[
                            0] * calibration_values.eyes_depth_offset
                        eyes_anchor_points_cal[0][1] = eyes_anchor_points[0][1] + scaled_face_vector[
                            1] * calibration_values.eyes_depth_offset
                        eyes_anchor_points_cal[1][0] = eyes_anchor_points[1][0] + scaled_face_vector[
                            0] * calibration_values.eyes_depth_offset
                        eyes_anchor_points_cal[1][1] = eyes_anchor_points[1][1] + scaled_face_vector[
                            1] * calibration_values.eyes_depth_offset

                        image = cv2.circle(image, (int(eyes_anchor_points_cal[0][0]),
                                                   int(eyes_anchor_points_cal[0][1])), 3, (200, 200, 200), 1)
                        image = cv2.circle(image, (int(eyes_anchor_points_cal[1][0]),
                                                   int(eyes_anchor_points_cal[1][1])), 3, (200, 200, 200), 1)

                    # # eye tilt
                    # eyes_vectors = calculate_eyes_vectors(eyes_anchor_points_cal, keypoint_left, keypoint_right,
                    #                                       eyes_depth_offset)
                    #
                    # # todo make for eyes
                    #
                    # left_eye_direction_offset = [((face_distance + left_eye_distance_offset + keypoint_left[2]) *
                    #                               eyes_vectors[0][0]) / eyes_vectors[0][2],
                    #                              ((face_distance + face_distance_offset + keypoint_left[2]) *
                    #                               eyes_vectors[0][1]) / eyes_vectors[0][2]]
                    #
                    # left_eye_center_screen = (window[2] * eyes_anchor_points_cal[0][0] / image.shape[1],
                    #                           window[3] * eyes_anchor_points_cal[0][1] / image.shape[0])
                    # left_eye_center_screen_cal = (left_eye_center_screen[0] - face_position_correction_width,
                    #                               left_eye_center_screen[1] - face_position_correction_height)
                    #
                    # left_eye_point = [left_eye_center_screen_cal[0] + left_eye_direction_offset[0],
                    #                   left_eye_center_screen_cal[1] + left_eye_direction_offset[1]]
                    #
                    # left_eye_point_cal = [left_eye_point[0] + left_eye_point_correction[0],
                    #                       left_eye_point[1] + left_eye_point_correction[1]]
                    #
                    # right_eye_direction_offset = [((face_distance + face_distance_offset + keypoint_right[2]) *
                    #                                eyes_vectors[0][0]) / eyes_vectors[0][2],
                    #                               ((face_distance + face_distance_offset + keypoint_right[2]) *
                    #                                eyes_vectors[0][1]) / eyes_vectors[0][2]]
                    #
                    # right_eye_center_screen = (window[2] * eyes_anchor_points_cal[0][0] / image.shape[1],
                    #                            window[3] * eyes_anchor_points_cal[0][1] / image.shape[0])
                    # right_eye_center_screen_cal = (right_eye_center_screen[0] - face_position_correction_width,
                    #                                right_eye_center_screen[1] - face_position_correction_height)
                    #
                    # right_eye_point = [right_eye_center_screen_cal[0] + right_eye_direction_offset[0],
                    #                    right_eye_center_screen_cal[1] + right_eye_direction_offset[1]]
                    #
                    # right_eye_point_cal = [right_eye_point[0] + right_eye_point_correction[0],
                    #                        right_eye_point[1] + right_eye_point_correction[1]]

                    left_gaze_point = [(keypoint_left[0] - eyes_anchor_points_cal[0][0]),
                                       (keypoint_left[1] - eyes_anchor_points_cal[0][1])]

                    left_gaze_point_cal = [left_gaze_point[0] + calibration_values.left_gaze_point_offset[0],
                                           left_gaze_point[1] + calibration_values.left_gaze_point_offset[1]]

                    left_gaze_point_fin = [left_gaze_point_cal[0] * calibration_values.left_gaze_point_factor[0],
                                           left_gaze_point_cal[1] * calibration_values.left_gaze_point_factor[1]]

                    right_gaze_point = [(keypoint_right[0] - eyes_anchor_points_cal[1][0]),
                                        (keypoint_right[1] - eyes_anchor_points_cal[1][1])]

                    right_gaze_point_cal = [right_gaze_point[0] + calibration_values.right_gaze_point_offset[0],
                                            right_gaze_point[1] + calibration_values.right_gaze_point_offset[1]]

                    right_gaze_point_fin = [right_gaze_point_cal[0] * calibration_values.right_gaze_point_factor[0],
                                            right_gaze_point_cal[1] * calibration_values.right_gaze_point_factor[1]]

                    total_gaze_point = [(left_gaze_point_fin[0] + right_gaze_point_fin[0]) / 2,
                                        (left_gaze_point_fin[1] + right_gaze_point_fin[1]) / 2]



                    # distance scaling
                    # total_gaze_point = (
                    #     int(total_gaze_point[0] * face_distance / 60), int(total_gaze_point[1] * face_distance / 60))

                    # draw gaze point

                    # face position
                    # total_offset = (
                    #     int(face_center_screen_cal[0]),
                    #     int(face_center_screen_cal[1]))
                    # face tilt
                    # total_offset = (
                    #     int(total_gaze_point[0]),
                    #     int(total_gaze_point[1]))
                    # eye tilt
                    total_offset = (
                        int(total_gaze_point[0] + window[2] / 2),
                        int(total_gaze_point[1] + window[3] / 2))
                    # all together
                    # total_offset = (face_center_screen_cal[0] + total_gaze_point[0],
                    #                 face_center_screen_cal[1] + total_gaze_point[1])

                    current_point = total_offset

                    # past value smoothing
                    for i in range(len(offset_history[:, 0]) - 2, -1, -1):
                        offset_history[i + 1] = offset_history[i]
                    offset_history[0] = [current_point[0], current_point[1]]

                    if smoothing_past_values_count == 0:
                        smooth_point = current_point
                    else:
                        sum_x = sum(offset_history[0:smoothing_past_values_count, 0])
                        sum_y = sum(offset_history[0:smoothing_past_values_count, 1])
                        smooth_point = (int(sum_x / smoothing_past_values_count),
                                        int(sum_y / smoothing_past_values_count))

                    # show point
                    screen = cv2.circle(screen, smooth_point, 20,
                                        (70, 200, 200), 2)
                    screen = cv2.circle(screen, (int(face_point_cal[0]), int(face_point_cal[1])), 20,
                                        (200, 200, 70), 2)
                    screen = cv2.circle(screen, (int(right_gaze_point_fin[0] + window[2] / 2),
                                                 int(right_gaze_point_fin[1] + window[3] / 2)), 20,
                                        (70, 200, 70), 2)
                    screen = cv2.circle(screen, (int(left_gaze_point_fin[0] + window[2] / 2),
                                                 int(left_gaze_point_fin[1] + window[3] / 2)), 20,
                                        (70, 70, 200), 2)


                # show whole mesh
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #         .get_default_face_mesh_tesselation_style())
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #         .get_default_face_mesh_contours_style())
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #         .get_default_face_mesh_iris_connections_style())

        else:
            face_detected = False
            left_eye_detected = False
        # show_calibration_values(screen)
        cv2.imshow('image', image)
        cv2.imshow('screen', screen)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
