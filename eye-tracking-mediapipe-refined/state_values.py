class StateValues:
    def __init__(self):
        self.evaluation_measuring_points = False
        self.calibration_completed = False
        self.evaluation_happening = False
        self.recording_happening = False
        self.show_diagnostics = False

    def reset(self):
        self.evaluation_measuring_points = False
        self.calibration_completed = False
        self.evaluation_happening = False
        self.recording_happening = False
