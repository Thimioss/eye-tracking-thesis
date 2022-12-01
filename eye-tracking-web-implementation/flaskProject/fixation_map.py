import imageio.v2 as imageio
import cv2
import math

import numpy as np

from flaskProject.image_edit import remove_black_background


def fixation_map_im(x_list, y_list, time_list, c_v):
    image = np.zeros(shape=(c_v.window[3], c_v.window[2], 3), dtype=np.uint8)
    overlay = image.copy()
    alpha = 0.5

    if len(x_list) == 0:
        overlay = remove_black_background(overlay)
        cv2.imwrite(c_v.last_file_name + '_fixation_map.png', overlay)
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
                radius = 25 + int(50 * (time_list[i] - min_time) / differece_times)
            else:
                radius = 25
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, (0, 255, 0), -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, (10, 10, 10), 2)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        image_new = remove_black_background(image_new)
        cv2.imwrite(c_v.last_file_name + '_fixation_map.png', image_new)
        return image_new
        # name = uniquify(str(home_directory) + '/' + 'Fixation_Map.png')
        # cv2.imwrite(name, cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()