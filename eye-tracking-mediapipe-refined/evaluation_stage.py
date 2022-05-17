from point_metrics import PointMetrics


class EvaluationStage:
    def __init__(self, points_count):
        self.points_metrics_list = []
        for i in range(0, points_count):
            self.points_metrics_list.append(PointMetrics())
        self.points_count = points_count

    def is_stage_complete(self):
        if self.points_metrics_list:
            for point_metrics in self.points_metrics_list:
                if not point_metrics.are_point_metrics_filled():
                    return False
            return True
        else:
            return False
