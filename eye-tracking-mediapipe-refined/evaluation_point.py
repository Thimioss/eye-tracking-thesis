import numpy as np

from point_metrics import PointMetrics


class EvaluationPoint:
    def __init__(self, measured_points_count):
        self.measured_points_count = measured_points_count
        self.evaluated_point = (-1, -1)
        self.measured_both_points = []
        self.measured_left_points = []
        self.measured_right_points = []
        self.__point_metrics = PointMetrics()

    def get_measured_points_count(self):
        return len(self.measured_both_points)

    def add_measured_point(self, both_point, left_point, right_point):
        if not self.are_measured_points_filled():
            self.measured_both_points.append(both_point)
            self.measured_left_points.append(left_point)
            self.measured_right_points.append(right_point)

    def set_evaluated_point(self, evaluated_point):
        self.evaluated_point = evaluated_point

    def are_measured_points_filled(self):
        return len(self.measured_both_points) == self.measured_points_count

    def get_point_metrics(self, calculated_values):
        if len(self.measured_both_points) < self.measured_points_count:
            return None
        else:
            self.__point_metrics.set_metrics_from_points_lists(self.evaluated_point, self.measured_both_points, self.measured_left_points,
                                                               self.measured_right_points, calculated_values)
            return self.__point_metrics
