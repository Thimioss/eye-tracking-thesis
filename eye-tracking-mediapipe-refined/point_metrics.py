import numpy as np


class PointMetrics:
    def __init__(self):
        self.left_pixel_accuracy = -1
        self.right_pixel_accuracy = -1
        self.binocular_pixel_accuracy = -1
        self.left_pixel_precision = -1
        self.right_pixel_precision = -1
        self.binocular_pixel_precision = -1
        self.pixel_sd_precision = -1
        self.left_angle_accuracy = -1
        self.right_angle_accuracy = -1
        self.binocular_angle_accuracy = -1
        self.left_angle_precision = -1
        self.right_angle_precision = -1
        self.binocular_angle_precision = -1
        self.angle_sd_precision = -1

    def set_metrics_from_points_lists(self, evaluated_point, measured_both_points, measured_left_points,
                                      measured_right_points, calculated_values):
        both_distance_mm = 60*10
        left_distance_mm = 60*10
        right_distance_mm = 60*10

        left_mean = ((sum(measured_left_points[:][0]) / len(measured_left_points[:][0])),
                     (sum(measured_left_points[:][1]) / len(measured_left_points[:][1])))
        right_mean = ((sum(measured_right_points[:][0]) / len(measured_right_points[:][0])),
                      (sum(measured_right_points[:][1]) / len(measured_right_points[:][1])))
        both_mean = ((sum(measured_both_points[:][0]) / len(measured_both_points[:][0])),
                     (sum(measured_both_points[:][1]) / len(measured_both_points[:][1])))

        self.left_pixel_accuracy = np.sqrt(np.power(evaluated_point[0] - left_mean[0], 2) +
                                           np.power(evaluated_point[1] - left_mean[1], 2))
        self.right_pixel_accuracy = np.sqrt(np.power(evaluated_point[0] - right_mean[0], 2) +
                                            np.power(evaluated_point[1] - right_mean[1], 2))
        self.binocular_pixel_accuracy = np.sqrt(np.power(evaluated_point[0] - both_mean[0], 2) +
                                                np.power(evaluated_point[1] - both_mean[1], 2))

        f = np.arctan(calculated_values.window[2] / calculated_values.window[3])
        h_cm = np.cos(f) * calculated_values.screen_diagonal_in_cm
        pixel_size_cm = h_cm / calculated_values.window[3]
        pixel_size_mm = pixel_size_cm * 10
        left_on_screen_distance = pixel_size_mm * np.sqrt(np.power((left_mean[0] -
                                                                    (calculated_values.window[2] / 2)), 2) +
                                                          np.power(left_mean[1] +
                                                                   calculated_values.camera_height_offset / pixel_size_mm,
                                                                   2))
        right_on_screen_distance = pixel_size_mm * np.sqrt(np.power((right_mean[0] -
                                                                     (calculated_values.window[2] / 2)), 2) +
                                                           np.power(right_mean[1] +
                                                                    calculated_values.camera_height_offset / pixel_size_mm,
                                                                    2))
        both_on_screen_distance = pixel_size_mm * np.sqrt(np.power((both_mean[0] -
                                                                    (calculated_values.window[2] / 2)), 2) +
                                                          np.power(both_mean[1] +
                                                                   calculated_values.camera_height_offset / pixel_size_mm,
                                                                   2))

        left_angle = np.arctan(left_on_screen_distance / left_distance_mm)
        right_angle = np.arctan(right_on_screen_distance / right_distance_mm)
        both_angle = np.arctan(both_on_screen_distance / both_distance_mm)

        self.left_angle_accuracy = (pixel_size_mm * self.left_pixel_accuracy * np.power(np.cos(left_angle),
                                                                                        2)) / left_distance_mm
        self.right_angle_accuracy = (pixel_size_mm * self.right_pixel_accuracy * np.power(np.cos(right_angle),
                                                                                          2)) / right_distance_mm
        self.binocular_angle_accuracy = (pixel_size_mm * self.binocular_pixel_accuracy * np.power(np.cos(both_angle),
                                                                                                  2)) / both_distance_mm

        left_dx = np.diff(measured_left_points[:][0])
        left_dy = np.diff(measured_left_points[:][1])
        right_dx = np.diff(measured_right_points[:][0])
        right_dy = np.diff(measured_right_points[:][1])
        both_dx = np.diff(measured_both_points[:][0])
        both_dy = np.diff(measured_both_points[:][1])

        left_d = []
        right_d = []
        both_d = []
        for i in range(len(left_dy)):
            left_d.append(np.sqrt(np.power(left_dx[i], 2) + np.power(left_dy[i], 2)))
            right_d.append(np.sqrt(np.power(right_dx[i], 2) + np.power(right_dy[i], 2)))
            both_d.append(np.sqrt(np.power(both_dx[i], 2) + np.power(both_dy[i], 2)))

        self.left_pixel_precision = np.sqrt(np.power(sum(left_d)/len(left_d), 2))
        self.right_pixel_precision = np.sqrt(np.power(sum(right_d)/len(right_d), 2))
        self.binocular_pixel_precision = np.sqrt(np.power(sum(both_d)/len(both_d), 2))

        self.left_angle_precision = (pixel_size_mm*self.left_pixel_precision*np.power(np.cos(left_angle), 2))/left_distance_mm
        self.right_angle_precision = (pixel_size_mm*self.right_pixel_precision*np.power(np.cos(right_angle), 2))/right_distance_mm
        self.binocular_angle_precision = (pixel_size_mm*self.binocular_pixel_precision*np.power(np.cos(both_angle), 2))/both_distance_mm

        temp_x = []
        temp_y = []
        for i in range(len(measured_both_points)):
            temp_x.append((measured_left_points[i][0]+measured_right_points[i][0])/2)
            temp_y.append((measured_left_points[i][1]+measured_right_points[i][1])/2)

        self.pixel_sd_precision = np.sqrt(np.power(np.std(temp_x), 2)+np.power(np.std(temp_y), 2))
        self.angle_sd_precision = (pixel_size_mm*self.pixel_sd_precision*np.power(np.cos(both_angle), 2))/both_distance_mm

    def set_metrics_from_sub_metrics(self, sub_metrics_list):
        l_p_a_sum = 0
        r_p_a_sum = 0
        b_p_a_sum = 0
        l_p_p_sum = 0
        r_p_p_sum = 0
        b_p_p_sum = 0
        p_s_p_sum = 0
        l_a_a_sum = 0
        r_a_a_sum = 0
        b_a_a_sum = 0
        l_a_p_sum = 0
        r_a_p_sum = 0
        b_a_p_sum = 0
        a_s_p_sum = 0
        for sub_metrics in sub_metrics_list:
            l_p_a_sum += sub_metrics.left_pixel_accuracy
            r_p_a_sum += sub_metrics.right_pixel_accuracy
            b_p_a_sum += sub_metrics.binocular_pixel_accuracy
            l_p_p_sum += sub_metrics.left_pixel_precision
            r_p_p_sum += sub_metrics.right_pixel_precision
            b_p_p_sum += sub_metrics.binocular_pixel_precision
            p_s_p_sum += sub_metrics.pixel_sd_precision
            l_a_a_sum += sub_metrics.left_angle_accuracy
            r_a_a_sum += sub_metrics.right_angle_accuracy
            b_a_a_sum += sub_metrics.binocular_angle_accuracy
            l_a_p_sum += sub_metrics.left_angle_precision
            r_a_p_sum += sub_metrics.right_angle_precision
            b_a_p_sum += sub_metrics.binocular_angle_precision
            a_s_p_sum += sub_metrics.angle_sd_precision
        self.left_pixel_accuracy = l_p_a_sum / len(sub_metrics_list)
        self.right_pixel_accuracy = r_p_a_sum / len(sub_metrics_list)
        self.binocular_pixel_accuracy = b_p_a_sum / len(sub_metrics_list)
        self.left_pixel_precision = l_p_p_sum / len(sub_metrics_list)
        self.right_pixel_precision = r_p_p_sum / len(sub_metrics_list)
        self.binocular_pixel_precision = b_p_p_sum / len(sub_metrics_list)
        self.pixel_sd_precision = p_s_p_sum / len(sub_metrics_list)
        self.left_angle_accuracy = l_a_a_sum / len(sub_metrics_list)
        self.right_angle_accuracy = r_a_a_sum / len(sub_metrics_list)
        self.binocular_angle_accuracy = b_a_a_sum / len(sub_metrics_list)
        self.left_angle_precision = l_a_p_sum / len(sub_metrics_list)
        self.right_angle_precision = r_a_p_sum / len(sub_metrics_list)
        self.binocular_angle_precision = b_a_p_sum / len(sub_metrics_list)
        self.angle_sd_precision = a_s_p_sum / len(sub_metrics_list)
