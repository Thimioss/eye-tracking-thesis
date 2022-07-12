class CalculatedValues:
    def __init__(self):
        self.last_file_name = ''
        self.screen_diagonal_in_cm = 24*2.54
        self.window = [0, 0, 1, 1]
        self.central_evaluation_points_offsets = [(0, 0), (400, 0), (0, 400), (400, 400), (-400, 0), (0, -400),
                                                  (-400, -400), (-400, +400), (+400, -400)]
        self.edge_evaluation_points_offsets = [(100, 100), (-100, -100), (100, -100), (-100, 100), (300, 300),
                                               (-300, -300), (300, -300), (-300, 300)]
        self.face_anchor_initial_points_2d = ((0, 0), (0, 0), (0, 0), (0, 0))
        self.face_anchor_initial_points_3d = []
        self.eyes_anchor_initial_points = ((0, 0), (0, 0))
        self.forehead_chin_landmark_distance = 0
        self.face_center_screen = (0, 0)
        self.face_distance = 60
        self.x_angle, self.y_angle, self.z_angle = 0, 0, 0
        self.x_cal, self.y_cal, self.z_cal = 0, 0, 0
        self.scaled_face_vector = [1, 1]
        self.camera_height_offset = 20

    def set_values_from_dictionary(self, dict):
        self.last_file_name = dict['last_file_name']
        self.screen_diagonal_in_cm = dict['screen_diagonal_in_cm']
        self.window = dict['window']
        self.central_evaluation_points_offsets = dict['central_evaluation_points_offsets']
        self.edge_evaluation_points_offsets = dict['edge_evaluation_points_offsets']
        self.face_anchor_initial_points_2d = dict['face_anchor_initial_points_2d']
        self.face_anchor_initial_points_3d = dict['face_anchor_initial_points_3d']
        self.eyes_anchor_initial_points = dict['eyes_anchor_initial_points']
        self.forehead_chin_landmark_distance = dict['forehead_chin_landmark_distance']
        self.face_center_screen = dict['face_center_screen']
        self.face_distance = dict['face_distance']
        self.x_angle, self.y_angle, self.z_angle = dict['x_angle'], dict['y_angle'], dict['z_angle']
        self.x_cal, self.y_cal, self.z_cal = dict['x_cal'], dict['y_cal'], dict['z_cal']
        self.scaled_face_vector = dict['scaled_face_vector']
        self.camera_height_offset = dict['camera_height_offset']

    def set_evaluation_points(self):
        self.central_evaluation_points_offsets = [(int(self.window[2] / 2), int(self.window[3] / 2)),
                                                  (int(self.window[2] / 2) + 400, int(self.window[3] / 2)),
                                                  (int(self.window[2] / 2), int(self.window[3] / 2) + 400),
                                                  (int(self.window[2] / 2) + 400, int(self.window[3] / 2) + 400),
                                                  (int(self.window[2] / 2) - 400, int(self.window[3] / 2)),
                                                  (int(self.window[2] / 2), int(self.window[3] / 2) - 400),
                                                  (int(self.window[2] / 2) - 400, int(self.window[3] / 2) - 400),
                                                  (int(self.window[2] / 2) - 400, int(self.window[3] / 2) + 400),
                                                  (int(self.window[2] / 2) + 400, int(self.window[3] / 2) - 400)]
        self.edge_evaluation_points_offsets = [(100, 100),
                                               (int(self.window[2]) - 100, int(self.window[3]) - 100),
                                               (100, int(self.window[3]) - 100),
                                               (int(self.window[2]) - 100, 100),
                                               (300, 300),
                                               (int(self.window[2]) - 300, int(self.window[3]) - 300),
                                               (300, int(self.window[3]) - 300),
                                               (int(self.window[2]) - 300, 300)]

