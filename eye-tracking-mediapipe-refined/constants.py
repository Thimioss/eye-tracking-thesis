class Constants:
    right_iris_index = 473
    left_iris_index = 468
    around_left_eye_cross_indexes = [143, 223, 189, 230]
    around_right_eye_cross_indexes = [372, 443, 413, 450]
    around_left_eye_landmarks_indexes = [143, 124, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]
    around_right_eye_landmarks_indexes = [372, 353, 445, 444, 443, 442, 441, 413, 464, 453, 452, 451, 450, 449, 448, 261]
    left_eye_landmarks_indexes = [33, 246, 161, 160, 159, 158, 157, 173, 155, 112, 26, 22, 23, 24, 110, 25]
    right_eye_landmarks_indexes = [263, 388, 387, 386, 385, 384, 398, 362, 341, 256, 252, 253, 254, 339, 255]
    left_eye_close_indexes = [33, 246, 161, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
    right_eye_close_indexes = [263, 388, 387, 386, 385, 384, 398, 362, 381, 380, 374, 373, 390, 249]
    left_eye_cross_landmarks_indexes = [33, 159, 155, 145]
    right_eye_cross_landmarks_indexes = [263, 386, 362, 374]
    right_face_edge_landmark_index = 300
    left_face_edge_landmark_index = 70
    between_eyebrows_landmark_index = 9
    between_eyes_landmark_index = 168
    nose_landmark_index = 4
    chin_landmark_index = 200
    forehead_landmark_index = 151
    left_face_edge_landmark_index_new = 143
    right_face_edge_landmark_index_new = 372
    face_edge_landmarks_indexes = [300, 151, 70, 200]
    # top left, top right, bottom left, bottom right, top, bottom, center, far left, far right, left nose, right nose, close left eye, close right eye
    # [53, 283, 150, 379, 151, 200, 4, 234, 454, 98, 327, 189, 413]
    # for 2d homography 53, 283, 150, 379
    face_anchors_landmarks_indexes = [53, 283, 150, 379, 151, 200, 4]