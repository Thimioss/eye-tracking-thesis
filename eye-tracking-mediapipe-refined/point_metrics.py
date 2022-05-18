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
        #todo
        self.left_pixel_accuracy = 0
        self.right_pixel_accuracy = 1
        self.binocular_pixel_accuracy = 2
        self.left_pixel_precision = 3
        self.right_pixel_precision = 4
        self.binocular_pixel_precision = 5
        self.pixel_sd_precision = 6
        self.left_angle_accuracy = 7
        self.right_angle_accuracy = 8
        self.binocular_angle_accuracy = 9
        self.left_angle_precision = 10
        self.right_angle_precision = 11
        self.binocular_angle_precision = 12
        self.angle_sd_precision = 13

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