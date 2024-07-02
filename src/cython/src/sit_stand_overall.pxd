from libc.math cimport atan2, fabs, pi
cpdef calculate_angle(double[:] a, double[:] b, double[:] c)

from libc.math cimport sqrt
from cpython.datetime cimport datetime


cpdef to_timestamp(double time_float)

cpdef to_timestamp(double time_float)

cpdef calculate_distance(double[:] p1, double[:] p2)
    
cpdef bint should_start_timer(
    double hip_angle,
    double hip_displacement, 
    double knee_displacement, 
    double ankle_displacement, 
    double threshold_angle, 
    double hip_thresh,
    double knee_thresh,
    double ankle_thresh)


import numpy as np
cimport numpy as cnp
from enum import Enum


# def enum PoseLandmark:
#     LEFT_HIP = 23
#     RIGHT_HIP = 24
#     LEFT_KNEE = 25
#     RIGHT_KNEE = 26
#     LEFT_ANKLE = 27
#     RIGHT_ANKLE = 28
#     LEFT_SHOULDER = 11
#     RIGHT_SHOULDER = 12
#     LEFT_ELBOW = 13
#     RIGHT_ELBOW = 14
#     LEFT_WRIST = 15
#     RIGHT_WRIST = 16

cpdef tuple get_landmark_coordinates(object landmarks, int frame_width, int frame_height, str side)

cpdef str determine_failure(double elapsed_time, int counter, double threshold_time, double tolerance, bint)

cpdef void summarise_results(int counter, double elapsed_time, list rep_durations,list violations, list max_angles,
                             dict joint_displacement_map, dict joint_velocity_map)

cpdef void display_knee_and_hip_angle(
    cnp.ndarray image,
    double knee_angle,
    double[:] knee,
    double hip_angle,
    double[:] hip,
    double x_displacement
)

cpdef void draw_joint_displacement(double[:] prev_point, double[:] curr_point, cnp.ndarray[cnp.uint8_t, ndim=3] image)

cpdef void draw_joint_displacement(double[:] prev_point, double[:] curr_point, cnp.ndarray[cnp.uint8_t, ndim=3] image)

cpdef unicode display_x_and_y_from_point(double[:] point)

from libc.math cimport sqrt
cimport numpy as cnp

cpdef dict calculate_and_draw_joint_displacement(
    cnp.ndarray prev_frame, 
    list prev_points, 
    list curr_points, 
    cnp.ndarray[cnp.uint8_t, ndim=3] image, 
    dict joint_displacement_history, 
    double real_time
)

cpdef void display_information(cnp.ndarray[cnp.uint8_t, ndim=3] image, int counter, str stage, double max_angle)

cpdef void display_timer(cnp.ndarray[cnp.uint8_t, ndim=3] image, double elapsed_time, int x, int y)

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

# Possible problematic
cpdef void draw_landmarks_and_connections(cnp.ndarray[cnp.uint8_t, ndim=3] image, results)
cpdef double get_real_time_from_frames(int frame_counter, double fps)

cpdef bint check_completion(int counter, str stage)


cpdef sit_stand_overall(str video_path)









