from evaluation_stage import EvaluationStage


class EvaluationData:
    def __init__(self):
        self.idealMetrics = EvaluationStage(9)
        self.edgeMetrics = EvaluationStage(8)
        self.darkMetrics = EvaluationStage(9)
        self.turnMetrics = EvaluationStage(9)

    def are_evaluation_data_filled(self):
        return self.idealMetrics.is_stage_complete() and self.edgeMetrics.is_stage_complete() and \
               self.darkMetrics.is_stage_complete() and self.turnMetrics.is_stage_complete()

    def get_completed_stages_count(self):
        if self.turnMetrics.is_stage_complete():
            return 4
        elif self.darkMetrics.is_stage_complete():
            return 3
        elif self.edgeMetrics.is_stage_complete():
            return 2
        elif self.idealMetrics.is_stage_complete():
            return 1
        else:
            return 0
