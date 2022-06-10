from evaluation_stage import EvaluationStage


class EvaluationData:
    def __init__(self):
        self.ideal_stage = EvaluationStage(9)
        self.edge_stage = EvaluationStage(8)
        self.dark_stage = EvaluationStage(9)
        self.turn_stage = EvaluationStage(9)

    def get_active_stage(self):
        if self.turn_stage.is_stage_complete():
            return None
        elif self.dark_stage.is_stage_complete():
            return self.turn_stage
        elif self.edge_stage.is_stage_complete():
            return self.dark_stage
        elif self.ideal_stage.is_stage_complete():
            return self.edge_stage
        else:
            return self.ideal_stage

    def add_points(self, both_point, left_point, right_point):
        if self.turn_stage.is_stage_complete():
            pass
        elif self.dark_stage.is_stage_complete():
            self.turn_stage.add_points(both_point, left_point, right_point)
        elif self.edge_stage.is_stage_complete():
            self.dark_stage.add_points(both_point, left_point, right_point)
        elif self.ideal_stage.is_stage_complete():
            self.edge_stage.add_points(both_point, left_point, right_point)
        else:
            self.ideal_stage.add_points(both_point, left_point, right_point)

    def are_evaluation_data_filled(self):
        return self.ideal_stage.is_stage_complete() and self.edge_stage.is_stage_complete() and \
               self.dark_stage.is_stage_complete() and self.turn_stage.is_stage_complete()

    def get_completed_stages_count(self):
        if self.turn_stage.is_stage_complete():
            return 4
        elif self.dark_stage.is_stage_complete():
            return 3
        elif self.edge_stage.is_stage_complete():
            return 2
        elif self.ideal_stage.is_stage_complete():
            return 1
        else:
            return 0
