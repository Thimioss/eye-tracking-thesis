import imageio.v2 as imageio
import cv2
import math

import numpy as np

from flaskProject.image_edit import remove_black_background


def fixation_scan_im(x_list, y_list, time_list, c_v):
    """
    Draws Fixation-Scanpath on image with saved data
    :param x_list: list with x coordinates of fixations
    :param y_list: list with y coordinates of fixations
    :param time_list: list with time of fixations
    :param imagen: image shown
    """
    image = np.zeros(shape=(c_v.window[3], c_v.window[2], 3), dtype=np.uint8)
    overlay = image.copy()
    # For circle
    alpha = 0.5
    radius = []
    # For Arrow
    color = (0, 0, 255)
    thickness = 5
    if len(x_list) == 0:
        overlay = remove_black_background(overlay)
        cv2.imwrite(c_v.last_file_name + '_fixation_scan.png', overlay)
        return overlay
        # name = uniquify(str(home_directory) + '/' + 'Fixation_Map.png')
        # cv2.imwrite(name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        min_time = min(time_list)
        max_time = max(time_list)
        differece_times = math.ceil(max_time - min_time)
        for i in range(len(x_list)):
            if differece_times != 0:
                radius.append(25 + int(50 * (time_list[i] - min_time) / differece_times))
            else:
                radius.append(25)
        for i in range(len(x_list)):
            cv2.circle(overlay, (x_list[i], y_list[i]), radius[i], (0, 255, 0), -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius[i], (10, 10, 10), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if i < 10:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 5, y_list[i] + 5), font, 0.5, (10, 10, 10), 2, cv2.LINE_AA)
            elif 9 < i < 100:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 10, y_list[i] + 5), font, 0.5, (10, 10, 10), 2, cv2.LINE_AA)
            elif i >= 100:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 15, y_list[i] + 5), font, 0.5, (10, 10, 10), 2, cv2.LINE_AA)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            if i < len(x_list) - 1:
                if x_list[i + 1] == x_list[i]:
                    theta = 90
                    if y_list[i + 1] > y_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] + radius[i])),
                                                    (int(x_list[i]), int(y_list[i + 1] - radius[i + 1])),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] < y_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] - radius[i])),
                                                    (int(x_list[i + 1]), int(y_list[i + 1] + radius[i + 1])),
                                                    color, thickness, tipLength=0.05)
                elif y_list[i + 1] == y_list[i]:
                    theta = 0
                    if x_list[i] < x_list[i + 1]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i] + radius[i]), int(y_list[i])),
                                                    (int(x_list[i + 1] - radius[i + 1]), int(y_list[i + 1])),
                                                    color, thickness, tipLength=0.05)
                    elif x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i] + radius[i]), int(y_list[i])),
                                                    (int(x_list[i + 1] - radius[i + 1]), int(y_list[i + 1])),
                                                    color, thickness, tipLength=0.05)
                else:
                    theta = math.atan(abs((y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])))
                    if y_list[i + 1] < y_list[i] and x_list[i + 1] > x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] + radius[i] * math.cos(theta))),
                                                     int(y_list[i] - radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] - radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] + radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] > y_list[i] and x_list[i + 1] > x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] + radius[i] * math.cos(theta))),
                                                     int(y_list[i] + radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] - radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] - radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] > y_list[i] and x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] - radius[i] * math.cos(theta))),
                                                     int(y_list[i] + radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] + radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] - radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] < y_list[i] and x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] - radius[i] * math.cos(theta))),
                                                     int(y_list[i] - radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] + radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] + radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)

        image_new = remove_black_background(image_new)
        cv2.imwrite(c_v.last_file_name + '_fixation_scan.png', image_new)
        return image_new
        # name = uniquify(str(home_directory) + '/' + 'Fixation_Map.png')
        # cv2.imwrite(name, cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
