from evaluation_stage import EvaluationStage


class EvaluationData:
    def __init__(self):
        self.ideal_stage = EvaluationStage(9)
        self.edge_stage = EvaluationStage(8)
        self.dark_stage = EvaluationStage(9)
        self.bright_stage = EvaluationStage(9)
        self.head_left_stage = EvaluationStage(9)
        self.head_right_stage = EvaluationStage(9)
        self.head_up_stage = EvaluationStage(9)
        self.head_down_stage = EvaluationStage(9)
        self.head_close_stage = EvaluationStage(9)
        self.head_far_stage = EvaluationStage(9)

    def get_active_stage(self):
        if self.head_far_stage.is_stage_complete():
            return None
        elif self.head_close_stage.is_stage_complete():
            return self.head_far_stage
        elif self.head_down_stage.is_stage_complete():
            return self.head_close_stage
        elif self.head_up_stage.is_stage_complete():
            return self.head_down_stage
        elif self.head_right_stage.is_stage_complete():
            return self.head_up_stage
        elif self.head_left_stage.is_stage_complete():
            return self.head_right_stage
        elif self.bright_stage.is_stage_complete():
            return self.head_left_stage
        elif self.dark_stage.is_stage_complete():
            return self.bright_stage
        elif self.edge_stage.is_stage_complete():
            return self.dark_stage
        elif self.ideal_stage.is_stage_complete():
            return self.edge_stage
        else:
            return self.ideal_stage

    def add_points(self, both_point, left_point, right_point):
        if self.head_far_stage.is_stage_complete():
            pass
        elif self.head_close_stage.is_stage_complete():
            self.head_far_stage.add_points(both_point, left_point, right_point)
        elif self.head_down_stage.is_stage_complete():
            self.head_close_stage.add_points(both_point, left_point, right_point)
        elif self.head_up_stage.is_stage_complete():
            self.head_down_stage.add_points(both_point, left_point, right_point)
        elif self.head_right_stage.is_stage_complete():
            self.head_up_stage.add_points(both_point, left_point, right_point)
        elif self.head_left_stage.is_stage_complete():
            self.head_right_stage.add_points(both_point, left_point, right_point)
        elif self.bright_stage.is_stage_complete():
            self.head_left_stage.add_points(both_point, left_point, right_point)
        elif self.dark_stage.is_stage_complete():
            self.bright_stage.add_points(both_point, left_point, right_point)
        elif self.edge_stage.is_stage_complete():
            self.dark_stage.add_points(both_point, left_point, right_point)
        elif self.ideal_stage.is_stage_complete():
            self.edge_stage.add_points(both_point, left_point, right_point)
        else:
            self.ideal_stage.add_points(both_point, left_point, right_point)

    def are_evaluation_data_filled(self):
        return self.ideal_stage.is_stage_complete() and self.edge_stage.is_stage_complete() and \
               self.dark_stage.is_stage_complete() and self.bright_stage.is_stage_complete() and \
               self.head_left_stage.is_stage_complete() and self.head_right_stage.is_stage_complete() and \
               self.head_up_stage.is_stage_complete() and self.head_down_stage.is_stage_complete() and \
               self.head_close_stage.is_stage_complete() and self.head_far_stage.is_stage_complete()

    def get_completed_stages_count(self):
        if self.head_far_stage.is_stage_complete():
            return 10
        elif self.head_close_stage.is_stage_complete():
            return 9
        elif self.head_down_stage.is_stage_complete():
            return 8
        elif self.head_up_stage.is_stage_complete():
            return 7
        elif self.head_right_stage.is_stage_complete():
            return 6
        elif self.head_left_stage.is_stage_complete():
            return 5
        elif self.bright_stage.is_stage_complete():
            return 4
        elif self.dark_stage.is_stage_complete():
            return 3
        elif self.edge_stage.is_stage_complete():
            return 2
        elif self.ideal_stage.is_stage_complete():
            return 1
        else:
            return 0
