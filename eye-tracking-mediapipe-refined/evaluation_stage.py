from constants import Constants
from evaluation_point import EvaluationPoint
from point_metrics import PointMetrics


class EvaluationStage:
    def __init__(self, points_count):
        self.__stage_metrics = PointMetrics()
        self.evaluation_points_list = []
        for i in range(0, points_count):
            self.evaluation_points_list.append(EvaluationPoint(Constants().measured_points_per_metric))
        self.points_count = points_count

    def get_stage_metrics(self, calculated_values):
        if self.__stage_metrics.pixel_sd_precision == -1:
            sub_metrics = []
            for e_point in self.evaluation_points_list:
                sub_metrics.append(e_point.get_point_metrics(calculated_values))
            self.__stage_metrics.set_metrics_from_sub_metrics(sub_metrics)
        return self.__stage_metrics

    def add_points(self, both_point, left_point, right_point):
        self.evaluation_points_list[self.get_completed_evaluation_points_count()]\
            .add_measured_point(both_point, left_point, right_point)

    def is_stage_complete(self):
        if self.evaluation_points_list:
            for point_metrics in self.evaluation_points_list:
                if not point_metrics.are_measured_points_filled():
                    return False
            return True
        else:
            return False

    def get_completed_evaluation_points_count(self):
        count = 0
        for e_point in self.evaluation_points_list:
            if e_point.are_measured_points_filled():
                count += 1
        return count
