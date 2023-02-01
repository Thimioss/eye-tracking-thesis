import collections
import fileinput
import fractions
import os

import cv2
import copy
import math
import mediapipe as mp
import numpy as np
import time
import pyautogui as gui
import csv
import json
import jsonpickle
import matplotlib.pyplot as plt
from textwrap import wrap

import heat_map_generator
from calculated_values import CalculatedValues
from calibration_values import CalibrationValues
from constants import Constants
from evaluation_data import EvaluationData
from state_values import StateValues

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

evaluation_data = EvaluationData()
constants = Constants()
calibration_values = CalibrationValues()
calculated_values = CalculatedValues()
state_values = StateValues()

start_time = time.time()
display_time = 2
frames_counter = 0
fps = 0
angles = [0, 0, 0]
f = None
writer = None
video_writer = None


def show_clustered_column_chart():
    plt.rcParams.update({'font.size': 8})
    labels = ['Ιδανικές συνθήκες', 'Ακραία σημεία οθόνης', 'Χαμηλός φωτισμός', 'Έντονος φωτισμός',
              'Μετατόπιση κεφαλιού (άξονας Χ)', 'Μετατόπιση κεφαλιού (άξονας Υ)', 'Μετατόπιση κεφαλιού (άξονας Ζ)']
    labels = ['\n'.join(wrap(l, 13)) for l in labels]
    # accuracy = [0.447, 0.397, 0.552, 0.476, 0.378, 0.421, 0.415]
    # precision = [0.288, 0.569, 0.19, 0.316, 0.49, 0.346, 0.35]
    # std_precision = [0.032, 0.032, 0.031, 0.031, 0.034, 0.04, 0.04]
    accuracy = [0.440, 0.421, 0.507, 0.453, 0.439, 0.334, 0.366]
    precision = [0.302, 0.548, 0.249, 0.276, 0.494, 0.425, 0.337]
    std_precision = [0.039, 0.045, 0.038, 0.052, 0.055, 0.05, 0.05]

    x__ = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    blue = '#4E8DE8'
    red = '#E84E4E'
    green = '#66D062'

    fig, ax = plt.subplots()
    rects1 = ax.bar(x__ - width, accuracy, width, label='Ορθότητα', color=blue)
    rects2 = ax.bar(x__, precision, width, label='Ακρίβεια', color=red)
    rects3 = ax.bar(x__ + width, std_precision, width, label='Ακρίβεια τυπικής απόκλισης', color=green)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Μέγεθος μετρικής')
    ax.set_title('Διοφθαλμικές μετρικές αξιολόγισης συστήματος σε διάφορες συνθήκες λειτουργίας')
    ax.set_xticks(x__, labels)
    ax.legend()

    ax.margins(0.1, 0.4)

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.savefig('clustered_column_chart.png', dpi=1000)
    plt.show()


def show_clustered_column_chart_comparison(my_system_monocular, pro_system_monocular, my_system_binocular,
                                           pro_system_binocular, label):
    plt.rcParams.update({'font.size': 8})
    labels = ['Ορθότητα', 'Ακρίβεια', 'Ακρίβεια τυπικής απόκλισης']
    labels = ['\n'.join(wrap(l, 13)) for l in labels]

    x__ = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    yellow = '#f5f242'
    purple = '#6e00db'

    fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True)
    rects1 = ax[0].bar(x__ - width / 2, my_system_monocular, width, label='Σύστημα που αναπτύχθηκε', color=yellow)
    rects2 = ax[0].bar(x__ + width / 2, pro_system_monocular, width, label='Επαγγελματικό σύστημα', color=purple)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].set_ylabel('Μέγεθος μετρικής')
    ax[0].set_title('Μονοφθαλμικά αποτελέσματα')
    ax[0].set_xticks(x__, labels)
    ax[0].legend()
    ax[0].margins(0.1, 0.4)
    ax[0].bar_label(rects1, padding=3)
    ax[0].bar_label(rects2, padding=3)

    rects1 = ax[1].bar(x__ - width / 2, my_system_binocular, width, label='Σύστημα που αναπτύχθηκε', color=yellow)
    rects2 = ax[1].bar(x__ + width / 2, pro_system_binocular, width, label='Επαγγελματικό σύστημα', color=purple)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1].set_ylabel('Μέγεθος μετρικής')
    ax[1].set_title('Διοφθαλμικά αποτελέσματα')
    ax[1].set_xticks(x__, labels)
    ax[1].legend()
    ax[1].margins(0.1, 0.4)
    ax[1].bar_label(rects1, padding=3)
    ax[1].bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.savefig('compare_column_chart' + label + '.png', dpi=1000)
    plt.show()


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


def make_nd_arrays_lists(calc, calib):
    calc.eyes_anchor_initial_points = calc.eyes_anchor_initial_points.tolist()
    calc.face_anchor_initial_points_3d = calc.face_anchor_initial_points_3d.tolist()
    calc.face_anchor_initial_points_2d = calc.face_anchor_initial_points_2d.tolist()
    calib.rvec_init = calib.rvec_init.tolist()
    calib.tvec_init = calib.tvec_init.tolist()


def save_values_to_json(name, dir):
    global calibration_values, calculated_values
    calc_val = copy.copy(calculated_values)
    calib_val = copy.copy(calibration_values)
    make_nd_arrays_lists(calc_val, calib_val)

    calc_val_json = json.dumps(calc_val.__dict__)
    calib_val_json = json.dumps(calib_val.__dict__)
    with open(name + '_calc_val.json', 'w') as outfile:
        json.dump(calc_val_json, outfile)
    with open(name + '_calib_val.json', 'w') as outfile:
        json.dump(calib_val_json, outfile)


def make_lists_nd_arrays(calc, calib):
    calc.eyes_anchor_initial_points = np.array(calc.eyes_anchor_initial_points)
    calc.face_anchor_initial_points_3d = np.array(calc.face_anchor_initial_points_3d)
    calc.face_anchor_initial_points_2d = np.array(calc.face_anchor_initial_points_2d)
    calib.rvec_init = np.array(calib.rvec_init)
    calib.tvec_init = np.array(calib.tvec_init)


def load_values_from_json(name, dir):
    global calibration_values, calculated_values
    with open(name + '_calc_val.json') as json_file:
        calc_val_json = json.load(json_file)
    with open(name + '_calib_val.json') as json_file:
        calib_val_json = json.load(json_file)

    calc_val_dict = jsonpickle.decode(calc_val_json)
    calib_val_dict = jsonpickle.decode(calib_val_json)

    calc_val = CalculatedValues()
    calc_val.set_values_from_dictionary(calc_val_dict)
    calib_val = CalibrationValues()
    calib_val.set_values_from_dictionary(calib_val_dict)

    make_lists_nd_arrays(calc_val, calib_val)

    calibration_values = calib_val
    calculated_values = calc_val


def show_image_for_heatmap(img):
    cv2.namedWindow('heatmap', cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback('heatmap', heatmap_mouse_event)
    cv2.setWindowProperty('heatmap', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('heatmap', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('heatmap', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('heatmap', img)


def start_recording_to_file():
    global f, writer, video_writer, calculated_values, state_values
    state_values.recording_happening = True
    file_name = gui.prompt("Enter the name of the recording", "Input info", "")
    calculated_values.last_file_name = file_name
    root_dir = os.path.dirname(os.path.abspath(__file__))
    save_values_to_json(file_name, root_dir)
    f = open(root_dir + '//' + file_name + '.csv', 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(
        ['Left_Gaze_Point_On_Display_Area_X', 'Right_Gaze_Point_On_Display_Area_X',
         'Left_Gaze_Point_On_Display_Area_Y', 'Right_Gaze_Point_On_Display_Area_Y', 'Date_time'])
    # record video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(file_name + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
    if not constants.is_for_experiment:
        bg_image = cv2.imread('bg_image.jpg')
        show_text(bg_image, 'Press anywhere to stop recording', 20, 20)
        show_image_for_heatmap(bg_image)


def show_wait_message_heatmap():
    heatmap = cv2.imread('bg_image.jpg')
    show_text(heatmap, 'Please wait...', 20, 20)
    cv2.imshow('heatmap', heatmap)


def show_heat_map():
    temp_right_eye_xs = []
    temp_left_eye_xs = []
    temp_right_eye_ys = []
    temp_left_eye_ys = []
    temp_xs = []
    temp_ys = []
    with open(calculated_values.last_file_name + '.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row_ in csv_reader:
            if line_count == 0:
                line_count += 1
            if 0 <= int((int(row_["Left_Gaze_Point_On_Display_Area_X"]) + int(
                    row_["Right_Gaze_Point_On_Display_Area_X"])) / 2) <= calculated_values.window[2] and \
                    0 <= int((int(row_["Right_Gaze_Point_On_Display_Area_Y"]) + int(
                row_["Left_Gaze_Point_On_Display_Area_Y"])) / 2) <= calculated_values.window[3]:
                temp_left_eye_xs.append(int(row_["Left_Gaze_Point_On_Display_Area_X"]))
                temp_right_eye_xs.append(int(row_["Right_Gaze_Point_On_Display_Area_X"]))
                temp_right_eye_ys.append(int(row_["Right_Gaze_Point_On_Display_Area_Y"]))
                temp_left_eye_ys.append(int(row_["Left_Gaze_Point_On_Display_Area_Y"]))
                temp_xs.append(int((int(row_["Left_Gaze_Point_On_Display_Area_X"]) + int(
                    row_["Right_Gaze_Point_On_Display_Area_X"])) / 2))
                temp_ys.append(calculated_values.window[3] -
                               int((int(row_["Right_Gaze_Point_On_Display_Area_Y"]) + int(
                                   row_["Left_Gaze_Point_On_Display_Area_Y"])) / 2))
            line_count += 1
    # temp_xs.append(calculated_values.window[2])
    # temp_ys.append(calculated_values.window[3])
    heatmap_image = heat_map_generator.generate_heat_map(np.array(temp_xs), np.array(temp_ys), calculated_values)
    show_text(heatmap_image, 'Press anywhere to close window', 20, 20)
    show_image_for_heatmap(heatmap_image)


def hide_heatmap_window():
    cv2.destroyWindow("heatmap")


def stop_recording_to_file():
    global f, writer, video_writer, state_values
    state_values.recording_happening = False
    if constants.is_for_experiment:
        gui.alert("Recording complete", "Success")
    else:
        show_wait_message_heatmap()
    writer = None
    video_writer = None
    f.close()
    f = None
    if not constants.is_for_experiment:
        show_heat_map()


def heatmap_mouse_event(event, x_, y_, flags, param):
    if event is cv2.EVENT_LBUTTONDOWN:
        if state_values.recording_happening:
            stop_recording_to_file()
        else:
            hide_heatmap_window()


def window_mouse_event(event, x_, y_, flags, param):
    global state_values, evaluation_data, calculated_values
    if event is cv2.EVENT_LBUTTONDOWN and 20 <= x_ < 40 and int(calculated_values.window[3] / 2) - 10 <= y_ < int(
            calculated_values.window[3] / 2) + 10:
        if state_values.evaluation_happening:
            gui.alert("You cannot calibrate while evaluation is happening", "Error")
        else:
            state_values.calibration_completed = False
            reset_calibrations()
    elif event is cv2.EVENT_LBUTTONDOWN and 20 <= x_ < 40 and int(calculated_values.window[3] / 2) + 30 <= y_ < int(
            calculated_values.window[3] / 2) + 50:
        if state_values.evaluation_happening:
            gui.alert("Evaluation is happening", "Error")
        else:
            state_values.calibration_completed = True
    elif event is cv2.EVENT_LBUTTONDOWN and 20 <= x_ < 40 and int(calculated_values.window[3] / 2) + 70 <= y_ < int(
            calculated_values.window[3] / 2) + 90:
        if state_values.calibration_completed is False:
            gui.alert("You cannot start evaluation without completing calibration", "Error")
        elif state_values.evaluation_happening is True:
            pass
        else:
            state_values.evaluation_happening = True
    elif event is cv2.EVENT_LBUTTONDOWN and 20 <= x_ < 40 and int(calculated_values.window[3] / 2) + 110 <= y_ < int(
            calculated_values.window[3] / 2) + 130:
        if state_values.calibration_completed is False:
            gui.alert("You cannot start recording without completing calibration", "Error")
        else:
            if not state_values.recording_happening:
                start_recording_to_file()
            else:
                stop_recording_to_file()
    elif event is cv2.EVENT_LBUTTONDOWN and 20 <= x_ < 40 and int(calculated_values.window[3] / 2) + 150 <= y_ < int(
            calculated_values.window[3] / 2) + 170:
        state_values.show_diagnostics = not state_values.show_diagnostics

    if state_values.calibration_completed is False:
        if event is cv2.EVENT_LBUTTONDOWN and int(2 * calculated_values.window[2] / 3) <= x_ < int(
                2 * calculated_values.window[2] / 3 + (
                        (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) and int(
            2 * calculated_values.window[3] / 3) <= y_ < int(
            2 * calculated_values.window[3] / 3 + (
                    (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)):
            calibrate_pose_estimation_and_anchor_points()
        elif event is cv2.EVENT_LBUTTONDOWN and int(
                2 * calculated_values.window[2] / 3 + (
                        (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) <= x_ < \
                calculated_values.window[2] and int(
            2 * calculated_values.window[3] / 3) <= y_ < int(
            2 * calculated_values.window[3] / 3 + (
                    (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)):
            calibrate_eyes_depth()
            calculate_face_distance_offset()
        elif event is cv2.EVENT_MBUTTONDOWN:
            calibrate_offsets()
        elif event is cv2.EVENT_LBUTTONDOWN and int(2 * calculated_values.window[2] / 3) <= x_ < int(
                2 * calculated_values.window[2] / 3 + (
                        (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) and int(
            2 * calculated_values.window[3] / 3 + (
                    (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) <= y_ < \
                calculated_values.window[3]:
            calculate_eye_correction_height_factor()
            # calculate_eyes_distance_offset()
        elif event is cv2.EVENT_LBUTTONDOWN and int(
                2 * calculated_values.window[2] / 3 + (
                        (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) <= x_ < \
                calculated_values.window[2] and int(
            2 * calculated_values.window[3] / 3 + (
                    (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) <= y_ < \
                calculated_values.window[3]:
            calculate_eye_correction_width_factor()
    else:
        pass

    if state_values.evaluation_happening:
        if event is cv2.EVENT_LBUTTONDOWN and int(calculated_values.window[2] / 2) - 150 <= x_ < int(
                calculated_values.window[2] / 2) + 150 \
                and int(calculated_values.window[3]) - 200 <= y_ < int(calculated_values.window[3]) - 100:
            state_values.evaluation_measuring_points = True
        elif event is cv2.EVENT_LBUTTONUP:
            state_values.evaluation_measuring_points = False


def reset_calibrations():
    global calibration_values, state_values, calculated_values
    calibration_values.reset()
    calculated_values.reset()
    state_values.reset()


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
    global calculated_values, calibration_values, rot_vec, trans_vec, \
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
    calibration_values.eyes_depth_offset = (keypoint_left[0] - eyes_anchor_points[0][0]) / \
                                           calculated_values.scaled_face_vector[0]


def calibrate_pose_estimation_and_anchor_points():
    global calculated_values, nose_landmark
    calculated_values.face_anchor_initial_points_2d = face_2d
    calculated_values.face_anchor_initial_points_3d = face_3d
    calculated_values.face_anchor_initial_points_3d = move_origin_point(calculated_values.face_anchor_initial_points_3d,
                                                                        nose_landmark)
    calculated_values.eyes_anchor_initial_points = [
        (int(keypoint_left[0]), int(keypoint_left[1]), int(keypoint_left[2])),
        (int(keypoint_right[0]), int(keypoint_right[1]), int(keypoint_right[2]))]
    # eyes_anchor_initial_points = move_origin_point(eyes_anchor_initial_points, nose_landmark)


def move_origin_point(points, new_origin_point):
    for j, point_ in enumerate(points, start=0):
        points[j] = [point_[0] - new_origin_point[0], point_[1] - new_origin_point[1], point_[2] - new_origin_point[2]]
    return points


def calibrate_offsets():
    global calibration_values, calculated_values
    if face_detected:
        calibration_values.face_height_on_60cm_away = calculated_values.forehead_chin_landmark_distance
        calibration_values.face_position_correction_width = calculated_values.face_center_screen[0] - (
                calculated_values.window[2] / 2)
        calibration_values.face_position_correction_height = calculated_values.face_center_screen[1] - (
                calculated_values.window[3] / 2)
        # calibration_values.face_point_correction = [calculated_values.window[2] / 2 - face_point[0],
        #                                             calculated_values.window[3] / 2 - face_point[1]]
        calculate_eye_correction_offsets()
        calibration_values.x_off = -calculated_values.x_angle
        calibration_values.y_off = -calculated_values.y_angle
        calibration_values.z_off = -calculated_values.z_angle


def calculate_eye_correction_offsets():
    global calibration_values, left_gaze_point, right_gaze_point
    calibration_values.left_gaze_point_offset = [-left_gaze_point[0], -left_gaze_point[1]]
    calibration_values.right_gaze_point_offset = [-right_gaze_point[0], -right_gaze_point[1]]


def calculate_eye_correction_width_factor():
    global calibration_values
    calibration_values.left_gaze_point_factor[0] = (screen.shape[1] / 2) / left_gaze_point_cal[0]
    calibration_values.right_gaze_point_factor[0] = (screen.shape[1] / 2) / right_gaze_point_cal[0]


def calculate_eye_correction_height_factor():
    global calibration_values
    calibration_values.left_gaze_point_factor[1] = (screen.shape[0] / 2) / left_gaze_point_cal[1]
    calibration_values.right_gaze_point_factor[1] = (screen.shape[0] / 2) / right_gaze_point_cal[1]


def calculate_face_distance_offset():
    global calibration_values, calculated_values
    calibration_values.face_distance_offset = ((screen.shape[1] - face_center_screen_cal[0] -
                                                calibration_values.face_point_correction[0]) * face_vector[2] /
                                               face_vector[0]) - calculated_values.face_distance


def calculate_eyes_distance_offset():
    global keypoint_right, keypoint_left, left_eye_center_screen_cal, right_eye_center_screen_cal, \
        left_eye_vector, right_eye_vector, calculated_values
    calibration_values.left_eye_distance_offset = ((screen.shape[1] - left_eye_center_screen_cal[0] -
                                                    calibration_values.left_eye_point_correction[0]) *
                                                   left_eye_vector[2] /
                                                   left_eye_vector[0]) - calculated_values.face_distance - \
                                                  keypoint_left[2]
    calibration_values.right_eye_distance_offset = ((screen.shape[1] - right_eye_center_screen_cal[0]
                                                     - calibration_values.right_eye_point_correction[0]) *
                                                    right_eye_vector[2] /
                                                    right_eye_vector[0]) - calculated_values.face_distance - \
                                                   keypoint_right[2]


def calculate_eyes_vectors(o, k_l, k_r, edo):
    global calculated_values
    if edo == 0:
        return [[0, 0, 1], [0, 0, 1]]
    else:
        x_left = (k_l[0] - o[0][0]) / (edo * 60 / calculated_values.face_distance)
        y_left = (k_l[1] - o[0][1]) / (edo * 60 / calculated_values.face_distance)

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

        x_right = (k_r[0] - o[1][0]) / (edo * 60 / calculated_values.face_distance)
        y_right = (k_r[1] - o[1][1]) / (edo * 60 / calculated_values.face_distance)

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


def show_measure_points_button(img):
    img = cv2.rectangle(img, (int(calculated_values.window[2] / 2) - 150, int(calculated_values.window[3]) - 200),
                        (int(calculated_values.window[2] / 2) + 150, int(calculated_values.window[3]) - 100),
                        (70, 200, 10), -1)
    show_text(img, "Hold to measure", int(calculated_values.window[2] / 2) - 70, int(calculated_values.window[3]) - 150)


def show_metrics(img, metrics, x_, y_, metric_name):
    show_text(img, metric_name, x_, y_)
    show_text(img, "left_pixel_accuracy " + str(metrics.left_pixel_accuracy), x_, y_ + 20)
    show_text(img, "right_pixel_accuracy " + str(metrics.right_pixel_accuracy), x_, y_ + 40)
    show_text(img, "binocular_pixel_accuracy " + str(metrics.binocular_pixel_accuracy), x_, y_ + 60)
    show_text(img, "left_pixel_precision " + str(metrics.left_pixel_precision), x_, y_ + 80)
    show_text(img, "right_pixel_precision " + str(metrics.right_pixel_precision), x_, y_ + 100)
    show_text(img, "binocular_pixel_precision " + str(metrics.binocular_pixel_precision), x_, y_ + 120)
    show_text(img, "pixel_sd_precision " + str(metrics.pixel_sd_precision), x_, y_ + 140)
    show_text(img, "left_angle_accuracy " + str(metrics.left_angle_accuracy), x_, y_ + 160)
    show_text(img, "right_angle_accuracy " + str(metrics.right_angle_accuracy), x_, y_ + 180)
    show_text(img, "binocular_angle_accuracy " + str(metrics.binocular_angle_accuracy), x_, y_ + 200)
    show_text(img, "left_angle_precision " + str(metrics.left_angle_precision), x_, y_ + 220)
    show_text(img, "right_angle_precision " + str(metrics.right_angle_precision), x_, y_ + 240)
    show_text(img, "binocular_angle_precision " + str(metrics.binocular_angle_precision), x_, y_ + 260)
    show_text(img, "angle_sd_precision " + str(metrics.angle_sd_precision), x_, y_ + 280)


def show_evaluation_metrics(img):
    global evaluation_data
    ideal_metrics = evaluation_data.ideal_stage.get_stage_metrics(calculated_values)
    edge_metrics = evaluation_data.edge_stage.get_stage_metrics(calculated_values)
    dark_metrics = evaluation_data.dark_stage.get_stage_metrics(calculated_values)
    bright_metrics = evaluation_data.bright_stage.get_stage_metrics(calculated_values)
    head_left_metrics = evaluation_data.head_left_stage.get_stage_metrics(calculated_values)
    head_right_metrics = evaluation_data.head_right_stage.get_stage_metrics(calculated_values)
    head_up_metrics = evaluation_data.head_up_stage.get_stage_metrics(calculated_values)
    head_down_metrics = evaluation_data.head_down_stage.get_stage_metrics(calculated_values)
    head_close_metrics = evaluation_data.head_close_stage.get_stage_metrics(calculated_values)
    head_far_metrics = evaluation_data.head_far_stage.get_stage_metrics(calculated_values)

    show_text(img, "Evaluation completed", int(calculated_values.window[2] / 2) - 100, 20)

    show_metrics(img, ideal_metrics, 400, 100, "Ideal metrics:")
    show_metrics(img, edge_metrics, 900, 100, "Edge metrics:")
    show_metrics(img, dark_metrics, 400, 430, "Dark metrics:")
    show_metrics(img, bright_metrics, 900, 430, "Bright metrics:")
    show_metrics(img, head_left_metrics, 400, 760, "Left head metrics:")
    show_metrics(img, head_right_metrics, 900, 760, "Right head metrics:")
    show_metrics(img, head_up_metrics, 1400, 100, "Up head metrics:")
    show_metrics(img, head_down_metrics, 1900, 100, "Down head metrics:")
    show_metrics(img, head_close_metrics, 1400, 430, "Close head metrics:")
    show_metrics(img, head_far_metrics, 1900, 430, "Far head metrics:")


def show_diagnostics(img):
    global calculated_values
    show_text(img, "Face distance: " + str(calculated_values.face_distance), 50,
              int(calculated_values.window[3] / 2) - 50)


def show_ui(img):
    global state_values
    show_text(img, "Restart calibration", 50, int(calculated_values.window[3] / 2))
    img = cv2.rectangle(img, (20, int(calculated_values.window[3] / 2) - 10),
                        (40, int(calculated_values.window[3] / 2) + 10),
                        (200, 100, 100), -1)
    show_text(img, "Calibration complete", 50, int(calculated_values.window[3] / 2) + 40)
    img = cv2.rectangle(img, (20, int(calculated_values.window[3] / 2) + 30),
                        (40, int(calculated_values.window[3] / 2) + 50),
                        (0, 200, 0), -1)

    show_text(img, "Start evaluation", 50, int(calculated_values.window[3] / 2) + 80)
    img = cv2.rectangle(img, (20, int(calculated_values.window[3] / 2) + 70),
                        (40, int(calculated_values.window[3] / 2) + 90),
                        (0, 200, 200), -1)

    if not state_values.recording_happening:
        show_text(img, "Start recording", 50, int(calculated_values.window[3] / 2) + 120)
        img = cv2.rectangle(img, (20, int(calculated_values.window[3] / 2) + 110),
                            (40, int(calculated_values.window[3] / 2) + 130),
                            (0, 0, 200), -1)
    else:
        img = cv2.circle(img, (calculated_values.window[2] - 200, 50), 20, (0, 0, 200), -1)
        show_text(img, "Recording...", calculated_values.window[2] - 170, 50)
        show_text(img, "Stop recording", 50, int(calculated_values.window[3] / 2) + 120)
        img = cv2.rectangle(img, (20, int(calculated_values.window[3] / 2) + 110),
                            (40, int(calculated_values.window[3] / 2) + 130),
                            (0, 0, 200), -1)

    if not state_values.show_diagnostics:
        show_text(img, "Show diagnostics", 50, int(calculated_values.window[3] / 2) + 160)
        img = cv2.rectangle(img, (20, int(calculated_values.window[3] / 2) + 150),
                            (40, int(calculated_values.window[3] / 2) + 170),
                            (0, 150, 200), -1)
    else:
        show_text(img, "Hide diagnostics", 50, int(calculated_values.window[3] / 2) + 160)
        img = cv2.rectangle(img, (20, int(calculated_values.window[3] / 2) + 150),
                            (40, int(calculated_values.window[3] / 2) + 170),
                            (0, 150, 200), -1)
        show_diagnostics(img)

    # show evaluation metrics if evaluation completed
    if evaluation_data.get_completed_stages_count() == 10:
        show_evaluation_metrics(img)

    if state_values.calibration_completed is False:
        show_calibration_ui(img)

    if state_values.evaluation_happening:
        show_measure_points_button(img)
        if evaluation_data.get_completed_stages_count() == 0:
            show_ideal_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 1:
            show_edge_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 2:
            show_dark_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 3:
            show_bright_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 4:
            show_head_left_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 5:
            show_head_right_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 6:
            show_head_up_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 7:
            show_head_down_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 8:
            show_head_close_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 9:
            show_head_far_evaluation_ui(img)
        elif evaluation_data.get_completed_stages_count() == 10:
            state_values.evaluation_happening = False


def show_head_far_evaluation_ui(img):
    img = cv2.putText(img, "Head far conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.head_far_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_head_close_evaluation_ui(img):
    img = cv2.putText(img, "Head close conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.head_close_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_head_down_evaluation_ui(img):
    img = cv2.putText(img, "Head down conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.head_down_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_head_up_evaluation_ui(img):
    img = cv2.putText(img, "Head up conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.head_up_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_head_left_evaluation_ui(img):
    img = cv2.putText(img, "Head left conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.head_left_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_head_right_evaluation_ui(img):
    img = cv2.putText(img, "Head right conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.head_right_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_bright_evaluation_ui(img):
    img = cv2.putText(img, "Bright conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.bright_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_dark_evaluation_ui(img):
    img = cv2.putText(img, "Dark conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.dark_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_head_left_evaluation_ui(img):
    img = cv2.putText(img, "Various head pose conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.head_left_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_ideal_evaluation_ui(img):
    img = cv2.putText(img, "Ideal conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.central_evaluation_points_offsets[
        evaluation_data.ideal_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_edge_evaluation_ui(img):
    img = cv2.putText(img, "Edge conditions evaluation", (int(calculated_values.window[2] / 2) - 100, 40),
                      cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), 2)

    img = cv2.circle(img, calculated_values.edge_evaluation_points_offsets[
        evaluation_data.edge_stage.get_completed_evaluation_points_count()], 20,
                     (200, 200, 200), 2)


def show_calibration_ui(img):
    global calculated_values
    img = cv2.circle(img, (calculated_values.window[2], int(calculated_values.window[3] / 2)), 20,
                     (70, 200, 200), 2)
    img = cv2.circle(img, (int(calculated_values.window[2] / 2), calculated_values.window[3]), 20,
                     (200, 70, 200), 2)
    img = cv2.circle(img, (int(calculated_values.window[2] / 2), int(calculated_values.window[3] / 2)), 20,
                     (200, 200, 200), 2)
    img = cv2.putText(img, "Stand 60cm away from screen and keep",
                      (int(img.shape[1] / 2) - 250, int(img.shape[0] / 2) - 200),
                      cv2.FONT_HERSHEY_PLAIN, 1.5,
                      (150, 150, 150), 2)
    img = cv2.putText(img, "your head still for calibration",
                      (int(img.shape[1] / 2) - 200, int(img.shape[0] / 2 + 30) - 200),
                      cv2.FONT_HERSHEY_PLAIN, 1.5,
                      (150, 150, 150), 2)
    img = cv2.rectangle(img, (int(2 * calculated_values.window[2] / 3), int(2 * calculated_values.window[3] / 3)),
                        (int(calculated_values.window[2]), int(calculated_values.window[3])),
                        (200, 100, 70), -1)

    img = cv2.line(img, (int(2 * calculated_values.window[2] / 3 + (
            (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)),
                         int(2 * calculated_values.window[3] / 3)),
                   (int(2 * calculated_values.window[2] / 3 + (
                           (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)),
                    int(calculated_values.window[3])),
                   (70, 70, 70), 2)
    img = cv2.line(img, (int(2 * calculated_values.window[2] / 3), int(2 * calculated_values.window[3] / 3 + (
            (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2))),
                   (int(calculated_values.window[2]), int(2 * calculated_values.window[3] / 3 + (
                           (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2))),
                   (70, 70, 70), 2)

    show_text(img, "3",
              int(2 * calculated_values.window[2] / 3 + (
                      (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) + 50,
              int(2 * calculated_values.window[3] / 3) + 30)
    show_text(img, "Face the right edge of the screen,",
              int(2 * calculated_values.window[2] / 3 + (
                      (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) + 50,
              int(2 * calculated_values.window[3] / 3) + 50)
    show_text(img, "look into the camera and left click here",
              int(2 * calculated_values.window[2] / 3 + (
                      (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) + 50,
              int(2 * calculated_values.window[3] / 3) + 70)

    show_text(img, "1", int(2 * calculated_values.window[2] / 3) + 50,
              int(2 * calculated_values.window[3] / 3) + 30)
    show_text(img, "Face and look into the camera", int(2 * calculated_values.window[2] / 3) + 50,
              int(2 * calculated_values.window[3] / 3) + 50)
    show_text(img, "and left click here",
              int(2 * calculated_values.window[2] / 3) + 50,
              int(2 * calculated_values.window[3] / 3) + 70)

    show_text(img, "5", int(2 * calculated_values.window[2] / 3) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 30)
    show_text(img, "Face the center of the", int(2 * calculated_values.window[2] / 3) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 50)
    show_text(img, "screen (white point),",
              int(2 * calculated_values.window[2] / 3) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 70)
    show_text(img, "look at the bottom edge (pink point)",
              int(2 * calculated_values.window[2] / 3) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 90)
    show_text(img, "and left click here",
              int(2 * calculated_values.window[2] / 3) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 110)

    show_text(img, "6", int(2 * calculated_values.window[2] / 3 + (
            (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 30)
    show_text(img, "Face the center of the", int(2 * calculated_values.window[2] / 3 + (
            (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 50)
    show_text(img, "screen (white point),",
              int(2 * calculated_values.window[2] / 3 + (
                      (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 70)
    show_text(img, "look at the right edge (yellow point)",
              int(2 * calculated_values.window[2] / 3 + (
                      (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 90)
    show_text(img, "and left click here",
              int(2 * calculated_values.window[2] / 3 + (
                      (calculated_values.window[2] - 2 * calculated_values.window[2] / 3) / 2)) + 50,
              int(2 * calculated_values.window[3] / 3 + (
                      (calculated_values.window[3] - 2 * calculated_values.window[3] / 3) / 2)) + 110)

    show_text(img, "2, 4", int(calculated_values.window[2] / 2) - 170, int(calculated_values.window[3] / 2) + 50)
    show_text(img, "Face and look at the middle of the", int(calculated_values.window[2] / 2) - 170,
              int(calculated_values.window[3] / 2) + 70)
    show_text(img, "screen (white point) and middle click here", int(calculated_values.window[2] / 2) - 200,
              int(calculated_values.window[3] / 2) + 90)


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
    global calculated_values
    chin_landmark = (f_l.landmark[constants.chin_landmark_index].x * img.shape[1],
                     f_l.landmark[constants.chin_landmark_index].y * img.shape[0],
                     f_l.landmark[constants.chin_landmark_index].z * img.shape[1])
    forehead_landmark = (f_l.landmark[constants.forehead_landmark_index].x * img.shape[1],
                         f_l.landmark[constants.forehead_landmark_index].y * img.shape[0],
                         f_l.landmark[constants.forehead_landmark_index].z * img.shape[1])
    calculated_values.forehead_chin_landmark_distance = np.sqrt(
        pow(chin_landmark[0] - forehead_landmark[0], 2) +
        pow(chin_landmark[1] - forehead_landmark[1], 2) +
        pow(chin_landmark[2] - forehead_landmark[2], 2))

    f_d = calibration_values.face_height_on_60cm_away * 60 / calculated_values.forehead_chin_landmark_distance
    return f_d


def calculate_face_center_screen_cal(img, wndw):
    face_cont = get_contour_from_landmark_indexes(constants.face_edge_landmarks_indexes, img)
    face_moment = cv2.moments(face_cont)
    face_center_image = (
        int(face_moment["m10"] / face_moment["m00"]), int(face_moment["m01"] / face_moment["m00"]))
    img = cv2.circle(img, face_center_image, 3, (200, 200, 200), 1)

    # face_center_image_offset = (face_center_image[0] - img.shape[1] / 2,
    #                             face_center_image[1] - img.shape[0] / 2)
    calculated_values.face_center_screen = (wndw[2] * face_center_image[0] / img.shape[1],
                                            wndw[3] * face_center_image[1] / img.shape[0])
    f_c_s_cal = (calculated_values.face_center_screen[0] - calibration_values.face_position_correction_width,
                 calculated_values.face_center_screen[1] - calibration_values.face_position_correction_height)
    return f_c_s_cal


def show_all_indexes(f_l, img):
    i_ = 0
    for landmark in f_l.landmark:
        img = cv2.putText(img, str(i_),
                          (int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])),
                          cv2.FONT_HERSHEY_PLAIN, 0.7,
                          (0, 255, 255), 0)
        i_ += 1


def show_whole_mesh(f_l, mp_f_m, mp_d_s, img):
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=f_l,
        connections=mp_f_m.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_d_s.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=f_l,
        connections=mp_f_m.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_d_s.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=f_l,
        connections=mp_f_m.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_d_s.get_default_face_mesh_iris_connections_style())


# show_clustered_column_chart()
# show_clustered_column_chart_comparison([0.401, 0.349, 0], [0.3, 0.14, 0], [0.44, 0.302, 0.039],
#                                        [0.2, 0.1, 0.05], 'ideal')
# show_clustered_column_chart_comparison([0.383, 0.571, 0], [0.4, 0.13, 0], [0.421, 0.548, 0.045],
#                                        [0.3, 0.09, 0.03], 'edges')
# show_clustered_column_chart_comparison([0.492, 0.266, 0], [1, 0.13, 0], [0.507, 0.249, 0.038],
#                                        [0.8, 0.09, 0.06], 'dark')
# show_clustered_column_chart_comparison([0.407, 0.292, 0], [0.6, 0.17, 0], [0.453, 0.276, 0.052],
#                                        [0.5, 0.11, 0.02], 'bright')
# show_clustered_column_chart_comparison([0.519, 0.867, 0], [0.4, 0.13, 0], [0.524, 0.685, 0.049],
#                                        [0.3, 0.1, 0.02], 'left')
# show_clustered_column_chart_comparison([0.289, 0.578, 0], [0.4, 0.2, 0], [0.354, 0.303, 0.061],
#                                        [0.5, 0.14, 0.07], 'right')
# show_clustered_column_chart_comparison([0.489, 0.166, 0], [0.5, 0.15, 0], [0.496, 0.189, 0.039],
#                                        [0.4, 0.1, 0.04], 'up')
# show_clustered_column_chart_comparison([0.165, 0.632, 0], [0.8, 0.16, 0], [0.173, 0.668, 0.062],
#                                        [0.8, 0.11, 0.03], 'down')
# show_clustered_column_chart_comparison([0.477, 0.194, 0], [0.8, 0.29, 0], [0.47, 0.181, 0.021],
#                                        [0.6, 0.19, 0.07], 'close')
# show_clustered_column_chart_comparison([0.212, 0.664, 0], [0.6, 0.34, 0], [0.292, 0.493, 0.08],
#                                        [0.5, 0.21, 0.14], 'far')
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
cv2.setMouseCallback('screen', window_mouse_event)

# average smoothing arrays
offset_history = np.array(
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
     [0, 0]])

landmarks_history = np.zeros([10, 4, 3])

screen_diagonal_in_inches = gui.prompt("Enter screen diagonal size in inches", "Input info", "24")
calculated_values.screen_diagonal_in_cm = int(screen_diagonal_in_inches) * 2.54
with mp_face_mesh.FaceMesh(max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.99) as face_mesh:
    while cap.isOpened():
        calculated_values.window = cv2.getWindowImageRect('screen')

        calculated_values.set_evaluation_points()

        screen = np.zeros((calculated_values.window[3], calculated_values.window[2], 3), dtype='uint8')
        screen = cv2.rectangle(screen, (0, 0), (calculated_values.window[2] - 1, calculated_values.window[3] - 1),
                               (85, 80, 78), -1)

        success, image = cap.read()
        pure_image = copy.copy(image)
        if state_values.recording_happening:
            video_writer.write(image)

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
                # show_all_indexes(face_landmarks, image)

                # iris points
                keypoint_left, keypoint_right = calculate_iris_points(face_landmarks, image)

                # face distance
                calculated_values.face_distance = calculate_face_distance(face_landmarks, image)

                # face position
                face_center_screen_cal = calculate_face_center_screen_cal(image, calculated_values.window)

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

                # for point in face_2d:
                #     image = cv2.circle(image, (int(point[0]), int(point[1])),
                #                        1, (255, 255, 255), 1)

                focal_length = 1 * image.shape[1]

                cam_matrix = np.array([[focal_length, 0, image.shape[0] / 2],
                                       [0, focal_length, image.shape[1] / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                calculated_values.face_anchor_initial_points_3d = np.array(
                    calculated_values.face_anchor_initial_points_3d, np.float32)
                if np.any(calculated_values.face_anchor_initial_points_3d):
                    # Solve PnP
                    if np.any(calibration_values.rvec_init) and np.any(calibration_values.tvec_init):
                        success, rot_vec, trans_vec = cv2.solvePnP(calculated_values.face_anchor_initial_points_3d,
                                                                   face_2d, cam_matrix,
                                                                   dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE,
                                                                   useExtrinsicGuess=True,
                                                                   rvec=calibration_values.rvec_init,
                                                                   tvec=calibration_values.tvec_init)
                    else:
                        success, rot_vec, trans_vec = cv2.solvePnP(calculated_values.face_anchor_initial_points_3d,
                                                                   face_2d, cam_matrix,
                                                                   dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
                        if trans_vec[2] > 0:
                            calibration_values.rvec_init = rot_vec
                            calibration_values.tvec_init = trans_vec

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Convert to radians
                    calculated_values.x_angle = angles[0] * np.pi / 180
                    calculated_values.y_angle = angles[1] * np.pi / 180
                    calculated_values.z_angle = angles[2] * np.pi / 180

                    # + x_off and so on
                    calculated_values.x_cal = calculated_values.x_angle + calibration_values.x_off
                    calculated_values.y_cal = calculated_values.y_angle + calibration_values.y_off
                    calculated_values.z_cal = calculated_values.z_angle + calibration_values.z_off

                    face_vector = [
                        (math.cos(calculated_values.z_cal) * math.sin(calculated_values.y_cal) * math.cos(
                            calculated_values.x_cal) + math.sin(calculated_values.z_cal) * math.sin(
                            calculated_values.x_cal)),
                        (math.sin(calculated_values.z_cal) * math.sin(calculated_values.y_cal) * math.cos(
                            calculated_values.x_cal) - math.cos(calculated_values.z_cal) * math.sin(
                            calculated_values.x_cal)),
                        (math.cos(calculated_values.y_cal) * math.cos(calculated_values.x_cal))]

                    calculated_values.scaled_face_vector = [
                        (math.cos(calculated_values.z_cal) * math.sin(calculated_values.y_cal) * math.cos(
                            calculated_values.x_cal) + math.sin(calculated_values.z_cal) * math.sin(
                            calculated_values.x_cal)) * 60 / calculated_values.face_distance,
                        (math.sin(calculated_values.z_cal) * math.sin(calculated_values.y_cal) * math.cos(
                            calculated_values.x_cal) - math.cos(calculated_values.z_cal) * math.sin(
                            calculated_values.x_cal)) * 60 / calculated_values.face_distance]

                    axisBoxes = np.float32([[0, 0, 0],
                                            [50, 0, 0],
                                            [0, 50, 0],
                                            [0, 0, 50]])

                    axis_3d_projection = cv2.projectPoints(axisBoxes, rot_vec, trans_vec,
                                                           cam_matrix, dist_matrix)[0]

                    # p11 = (int(nose_landmark[0]), int(nose_landmark[1]))
                    # p22 = (int(nose_landmark[0] + face_vector[0] * 50 * 60 / calculated_values.face_distance),
                    #        int(nose_landmark[1] + face_vector[1] * 50 * 60 / calculated_values.face_distance))
                    #
                    # cv2.line(image, p11, p22, (255, 255, 255), 3)

                    if axis_3d_projection is not None and math.isnan(axis_3d_projection[0][0][0]) is False:
                        p1 = (int(axis_3d_projection[0][0][0]), int(axis_3d_projection[0][0][1]))
                        p2 = (int(axis_3d_projection[1][0][0]), int(axis_3d_projection[1][0][1]))
                        p3 = (int(axis_3d_projection[2][0][0]), int(axis_3d_projection[2][0][1]))
                        p4 = (int(axis_3d_projection[3][0][0]), int(axis_3d_projection[3][0][1]))

                        # show face axis
                        # cv2.line(image, p1, p4, (255, 0, 0), 3)
                        # cv2.line(image, p1, p2, (0, 0, 255), 3)
                        # cv2.line(image, p1, p3, (0, 255, 0), 3)

                        # show_text(image, "x", p2[0], p2[1])
                        # show_text(image, "y", p3[0], p3[1])
                        # show_text(image, "z", p4[0], p4[1])

                    face_reprojection = \
                        cv2.projectPoints(calculated_values.face_anchor_initial_points_3d, rot_vec, trans_vec,
                                          cam_matrix, dist_matrix)[0]
                    # for point in face_reprojection:
                    #     image = cv2.circle(image, (int(point[0][0]), int(point[0][1])), 1, (0, 255, 255))

                    face_direction_offset = [
                        ((calculated_values.face_distance + calibration_values.face_distance_offset) * face_vector[0]) /
                        face_vector[2],
                        ((calculated_values.face_distance + calibration_values.face_distance_offset) * face_vector[1]) /
                        face_vector[2]]

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

                    calculated_values.face_anchor_initial_points_2d = np.array(
                        calculated_values.face_anchor_initial_points_2d, np.float32)
                    face_anchor_points = np.array(face_2d, np.float32)
                    for point in face_2d[0:4]:
                        image = cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 255))
                    h, status = cv2.findHomography(calculated_values.face_anchor_initial_points_2d[0:4],
                                                   face_anchor_points[0:4],
                                                   method=cv2.RANSAC,
                                                   ransacReprojThreshold=1, mask=None, maxIters=1, confidence=1)
                    if h is not None:
                        calculated_values.eyes_anchor_initial_points = np.array(
                            calculated_values.eyes_anchor_initial_points, np.float32)
                        eyes_anchor_points = [np.dot(h, [calculated_values.eyes_anchor_initial_points[0][0],
                                                         calculated_values.eyes_anchor_initial_points[0][1], 1]),
                                              np.dot(h, [calculated_values.eyes_anchor_initial_points[1][0],
                                                         calculated_values.eyes_anchor_initial_points[1][1], 1])]
                        eyes_anchor_points[0] /= eyes_anchor_points[0][2]
                        eyes_anchor_points[1] /= eyes_anchor_points[1][2]

                        eyes_anchor_points_cal = [[0, 0], [0, 0]]
                        eyes_anchor_points_cal[0][0] = eyes_anchor_points[0][0] + calculated_values.scaled_face_vector[
                            0] * calibration_values.eyes_depth_offset
                        eyes_anchor_points_cal[0][1] = eyes_anchor_points[0][1] + calculated_values.scaled_face_vector[
                            1] * calibration_values.eyes_depth_offset
                        eyes_anchor_points_cal[1][0] = eyes_anchor_points[1][0] + calculated_values.scaled_face_vector[
                            0] * calibration_values.eyes_depth_offset
                        eyes_anchor_points_cal[1][1] = eyes_anchor_points[1][1] + calculated_values.scaled_face_vector[
                            1] * calibration_values.eyes_depth_offset

                        image = cv2.circle(image, (int(eyes_anchor_points_cal[0][0]),
                                                   int(eyes_anchor_points_cal[0][1])), 3, (0, 255, 255), 1)
                        image = cv2.circle(image, (int(eyes_anchor_points_cal[1][0]),
                                                   int(eyes_anchor_points_cal[1][1])), 3, (0, 255, 255), 1)

                    # # eye tilt
                    # eyes_vectors = calculate_eyes_vectors(eyes_anchor_points_cal, keypoint_left, keypoint_right,
                    #                                       eyes_depth_offset)
                    #
                    # # todo make for eyes
                    #
                    # left_eye_direction_offset = [((calculated_values.face_distance + left_eye_distance_offset + keypoint_left[2]) *
                    #                               eyes_vectors[0][0]) / eyes_vectors[0][2],
                    #                              ((calculated_values.face_distance + face_distance_offset + keypoint_left[2]) *
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
                    # right_eye_direction_offset = [((calculated_values.face_distance + face_distance_offset + keypoint_right[2]) *
                    #                                eyes_vectors[0][0]) / eyes_vectors[0][2],
                    #                               ((calculated_values.face_distance + face_distance_offset + keypoint_right[2]) *
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

                    left_gaze_point_fin = [left_gaze_point_fin[0] * np.square(calculated_values.face_distance / 60),
                                           left_gaze_point_fin[1] * np.square(calculated_values.face_distance / 60)]

                    right_gaze_point = [(keypoint_right[0] - eyes_anchor_points_cal[1][0]),
                                        (keypoint_right[1] - eyes_anchor_points_cal[1][1])]

                    right_gaze_point_cal = [right_gaze_point[0] + calibration_values.right_gaze_point_offset[0],
                                            right_gaze_point[1] + calibration_values.right_gaze_point_offset[1]]

                    right_gaze_point_fin = [right_gaze_point_cal[0] * calibration_values.right_gaze_point_factor[0],
                                            right_gaze_point_cal[1] * calibration_values.right_gaze_point_factor[1]]

                    right_gaze_point_fin = [right_gaze_point_fin[0] * np.square(calculated_values.face_distance / 60),
                                            right_gaze_point_fin[1] * np.square(calculated_values.face_distance / 60)]

                    total_gaze_point = [(left_gaze_point_fin[0] + right_gaze_point_fin[0]) / 2,
                                        (left_gaze_point_fin[1] + right_gaze_point_fin[1]) / 2]

                    # distance scaling
                    # total_gaze_point = (
                    #     int(total_gaze_point[0] * calculated_values.face_distance / 60), int(total_gaze_point[1] * calculated_values.face_distance / 60))

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
                        int(total_gaze_point[0] + calculated_values.window[2] / 2),
                        int(total_gaze_point[1] + calculated_values.window[3] / 2))
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
                    if state_values.show_diagnostics:
                        screen = cv2.circle(screen, smooth_point, 20,
                                            (70, 200, 200), 2)
                        screen = cv2.circle(screen, (int(face_point_cal[0]), int(face_point_cal[1])), 20,
                                            (200, 200, 70), 2)
                        screen = cv2.circle(screen, (int(right_gaze_point_fin[0] + calculated_values.window[2] / 2),
                                                     int(right_gaze_point_fin[1] + calculated_values.window[3] / 2)),
                                            20,
                                            (70, 200, 70), 2)
                        screen = cv2.circle(screen, (int(left_gaze_point_fin[0] + calculated_values.window[2] / 2),
                                                     int(left_gaze_point_fin[1] + calculated_values.window[3] / 2)), 20,
                                            (70, 70, 200), 2)
                        screen = cv2.circle(screen, (int(face_center_screen_cal[0]),
                                                     int(face_center_screen_cal[1])), 20,
                                            (200, 200, 200), 2)

                    # record values to file
                    if state_values.recording_happening:
                        if writer is not None:
                            row = [int(left_gaze_point_fin[0] + calculated_values.window[2] / 2),
                                   int(right_gaze_point_fin[0] + calculated_values.window[2] / 2),
                                   int(left_gaze_point_fin[1] + calculated_values.window[3] / 2),
                                   int(right_gaze_point_fin[1] + calculated_values.window[3] / 2),
                                   time.time()]
                            writer.writerow(row)

                    # measure values for evaluation
                    if state_values.evaluation_happening:
                        if evaluation_data.get_completed_stages_count() == 10:
                            state_values.evaluation_happening = False
                            state_values.evaluation_measuring_points = False
                        if state_values.evaluation_measuring_points:
                            if evaluation_data.get_active_stage() is not None:
                                temp = evaluation_data.get_active_stage().get_completed_evaluation_points_count()
                                evaluation_data.add_points(current_point,
                                                           (int(left_gaze_point_fin[0] + calculated_values.window[
                                                               2] / 2),
                                                            int(left_gaze_point_fin[1] + calculated_values.window[
                                                                3] / 2)),
                                                           (int(right_gaze_point_fin[0] + calculated_values.window[
                                                               2] / 2),
                                                            int(right_gaze_point_fin[1] + calculated_values.window[
                                                                3] / 2)))
                                if evaluation_data.get_active_stage() is not None:
                                    if evaluation_data.get_active_stage().get_completed_evaluation_points_count() is not \
                                            temp:
                                        state_values.evaluation_measuring_points = False
                                else:
                                    state_values.evaluation_measuring_points = False

                # show whole mesh
                # show_whole_mesh(face_landmarks, mp_face_mesh, mp_drawing_styles, image)

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
