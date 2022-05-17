class PointMetrics:
    def __init__(self):
        self.point = (-1, -1)
        self.accuracy = -1
        self.precision = -1

    def are_point_metrics_filled(self):
        return self.accuracy != -1 and self.precision != -1

