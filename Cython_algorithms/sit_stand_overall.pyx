

# angle_calculation.pyx
from libc.math cimport atan2, fabs, pi

cpdef calculate_angle(double[:] a, double[:] b, double[:] c):
    """
    Calculate the angle formed at point 'b' by the line segments a-b and b-c.
    `b` is the midpoint of `a` and `c` (e.g., left hip, left knee, and left ankle).

    Parameters:
    a, b, c (double[:]): Coordinates of the points.

    Returns:
    double: The angle in degrees.
    """

    cdef:
        double radian_a, radian_b, angle

    # Using libc's atan2 for computation
    radian_a = atan2(c[1] - b[1], c[0] - b[0])
    radian_b = atan2(a[1] - b[1], a[0] - b[0])
    angle = fabs((radian_a - radian_b) * 180.0 / pi)
    
    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# calculation_module.pyx
from libc.math cimport sqrt
from cpython.datetime cimport datetime


cpdef to_timestamp(double time_float):
    """
    Converts a float timestamp to a formatted datetime string.
    """
    cdef datetime dt = datetime.fromtimestamp(time_float)
    return dt.strftime("%Y-%m-%d %H:%M:%S,%f")

cpdef calculate_distance(double[:] p1, double[:] p2):
    """
    Calculates the Euclidean distance between two points in 2D space.
    """
    cdef double dx = p2[0] - p1[0]
    cdef double dy = p2[1] - p1[1]
    return sqrt(dx * dx + dy * dy)


# import printf
from libc.stdio cimport printf
# Define the should_start_timer function
cpdef bint should_start_timer(
    double hip_angle,
    double hip_displacement, 
    double knee_displacement, 
    double ankle_displacement,
):
    """
    Determine if the timer should start based on displacement conditions.
    keypoints should be a 2D array where rows are points [knee, hip, shoulder],
    and each point is an array of two doubles (x, y coordinates).
    """
    cdef:
        double threshold_angle=100.0
        double hip_thresh=3.0
        double knee_thresh=3.0 
        double ankle_thresh=1.5
    logging.info(f"Timer not started - Hip angle: {hip_angle:.2f} degrees")
    return (hip_angle < threshold_angle and 
            (hip_displacement > hip_thresh or
             knee_displacement > knee_thresh or
             ankle_displacement > ankle_thresh))


# landmarks.pyx
import numpy as np
cimport numpy as cnp
from enum import Enum

# Define the PoseLandmark enum to mirror the Python MediaPipe PoseLandmark
class PoseLandmark(Enum):
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

cpdef tuple get_landmark_coordinates(landmarks, int frame_width, int frame_height, str side):
    """
    Fetches the coordinates of hip, knee, ankle, shoulder, wrist, and elbow landmarks,
    scaled by the frame dimensions.
    """
    # Initialize arrays for storing the coordinates
    cdef:
        cnp.ndarray[cnp.float64_t, ndim=1] hip = np.zeros(2, dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=1] knee = np.zeros(2, dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=1] ankle = np.zeros(2, dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=1] shoulder = np.zeros(2, dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=1] wrist = np.zeros(2, dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=1] elbow = np.zeros(2, dtype=np.float64)
    if side == "RIGHT":
        hip_idx = PoseLandmark.RIGHT_HIP.value
        knee_idx = PoseLandmark.RIGHT_KNEE.value
        ankle_idx = PoseLandmark.RIGHT_ANKLE.value
        shoulder_idx = PoseLandmark.RIGHT_SHOULDER.value
        wrist_idx = PoseLandmark.RIGHT_WRIST.value
        elbow_idx = PoseLandmark.RIGHT_ELBOW.value
    else:  # Default to LEFT
        hip_idx = PoseLandmark.LEFT_HIP.value
        knee_idx = PoseLandmark.LEFT_KNEE.value
        ankle_idx = PoseLandmark.LEFT_ANKLE.value
        shoulder_idx = PoseLandmark.LEFT_SHOULDER.value
        wrist_idx = PoseLandmark.LEFT_WRIST.value
        elbow_idx = PoseLandmark.LEFT_ELBOW.value

    # Populate the coordinates
    hip[0] = landmarks[hip_idx].x * frame_width
    hip[1] = landmarks[hip_idx].y * frame_height
    knee[0] = landmarks[knee_idx].x * frame_width
    knee[1] = landmarks[knee_idx].y * frame_height
    ankle[0] = landmarks[ankle_idx].x * frame_width
    ankle[1] = landmarks[ankle_idx].y * frame_height
    shoulder[0] = landmarks[shoulder_idx].x * frame_width
    shoulder[1] = landmarks[shoulder_idx].y * frame_height
    wrist[0] = landmarks[wrist_idx].x * frame_width
    wrist[1] = landmarks[wrist_idx].y * frame_height
    elbow[0] = landmarks[elbow_idx].x * frame_width
    elbow[1] = landmarks[elbow_idx].y * frame_height

    return (np.array(hip), np.array(knee), np.array(ankle), np.array(shoulder), np.array(wrist), np.array(elbow))

# failure.pyx
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cpdef str determine_failure(double elapsed_time, int counter, bint debug):
    """
    Determine if the user has failed based on the elapsed time with precision up to 5 decimal places.
    Includes a tolerance for comparisons to handle edge cases near the threshold.
    """
    # Log the elapsed time with high precision
    cdef:
        double tolerance=0.001
        double threshold_time=12.0
    if debug:
        logging.info(f"Elapsed time: {elapsed_time:.5f}")

    # Adjust the threshold with a tolerance for the comparison
    if counter == 5 and elapsed_time < threshold_time - tolerance:
        return "PASSED"
    else:
        return "FAILED"


# summarise_results.pyx
cpdef void summarise_results(int counter, double elapsed_time, list rep_durations,list violations, list max_angles):
    """
    Logs a summary of video processing results including repetition counts and times,
    and uses determine_failure to log the pass/fail status based on elapsed time.
    """
    # Setup logging - ensure logging is configured elsewhere in your Python code
    logging.info("==========================================================")
    logging.info("                     Video processing summary                     ")
    logging.info("==========================================================")
    
    logging.info(f"Pass status: {determine_failure(elapsed_time, counter, True)}")
    logging.info(f"Total repetitions: {counter} completed in {elapsed_time:.2f} seconds")

    if violations:
        logging.info(f"Hand Violation count: {len(violations)}")
    
    # cdef double reps
    # if rep_durations:
    #     for reps in rep_durations:
    #         logging.info(f"Duration: {reps:.2f} seconds")

    #     logging.info(f"Maximum duration per repetition: {max(rep_durations):.2f} seconds")
    #     average_duration = sum(rep_durations) / len(rep_durations)
    #     logging.info(f"Average duration per repetition: {average_duration:.2f} seconds")
    if max_angles:
        logging.info(f"Maximum angle per repetition: {max(max_angles):.2f} degrees")
        average_angle = sum(max_angles) / len(max_angles)
        logging.info(f"Average angle per repetition: {average_angle:.2f} degrees")    
# display_angles.pyx
import cv2


cpdef void display_knee_and_hip_angle(
    cnp.ndarray image,
    double knee_angle,
    double[:] knee,
    double hip_angle,
    double[:] hip):
    """
    Display the knee and hip angles on the screen, 60px to the right of the respective joints.
    """
    cdef:
        double x_displacement=60.0
        int knee_x = int(knee[0] + x_displacement)
        int knee_y = int(knee[1])
        int hip_x = int(hip[0] + x_displacement)
        int hip_y = int(hip[1])

    cv2.putText(
        image,
        f"{knee_angle:.2f} deg",
        (knee_x, knee_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{hip_angle:.2f} deg",
        (hip_x, hip_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# draw_displacement.pyx

cpdef void draw_joint_displacement(double[:] prev_point, double[:] curr_point, cnp.ndarray[cnp.uint8_t, ndim=3] image):
    """
    Draws a circle at the current point and a line connecting the previous point to the current point on the image.
    """
    cdef int prev_x = int(prev_point[0])
    cdef int prev_y = int(prev_point[1])
    cdef int curr_x = int(curr_point[0])
    cdef int curr_y = int(curr_point[1])

    cv2.circle(image, (curr_x, curr_y), 10, (0, 0, 255), -1)
    cv2.line(image, (prev_x, prev_y), (curr_x, curr_y), (0, 255, 0), 2)


# display_point.pyx
# display_point.pyx
cpdef unicode display_x_and_y_from_point(double[:] point):
    """
    Returns a string representation of the x and y coordinates of the point.
    """
    return u"(x:{:.2f}, y:{:.2f})".format(point[0], point[1])

from libc.math cimport sqrt
cimport numpy as cnp

cpdef dict calculate_and_draw_joint_displacement(
    cnp.ndarray prev_frame, 
    list prev_points, 
    list curr_points, 
    cnp.ndarray[cnp.uint8_t, ndim=3] image, 
    dict joint_displacement_history, 
    double real_time
):
    logging.info(
        "========================= Optical flow results ========================="
    )
    cdef tuple joint_names = ("HIP", "KNEE", "ANKLE")  # Names of the joints
    cdef double[:] prev_hip = prev_points[0]
    cdef double[:] prev_knee = prev_points[1]
    cdef double[:] prev_ankle = prev_points[2]
    cdef double[:] hip = curr_points[0]
    cdef double[:] knee = curr_points[1]
    cdef double[:] ankle = curr_points[2]
    
    cdef double displacement
    if prev_frame is not None:
        for prev_point, curr_point, joint_name in zip(
            [prev_hip, prev_knee, prev_ankle],
            [hip, knee, ankle],
            joint_names,
        ):
            draw_joint_displacement(prev_point, curr_point, image)
            displacement = calculate_distance(prev_point, curr_point)
            logging.info(
                f"{joint_name} Displacement: {displacement:.2f} px (from {display_x_and_y_from_point(prev_point)} to {display_x_and_y_from_point(curr_point)})"
            )
            joint_displacement_history[joint_name].append((real_time, displacement))
    return joint_displacement_history


cpdef void display_information(cnp.ndarray[cnp.uint8_t, ndim=3] image, int counter, str stage, double max_angle):
    """
    Display the number of repetitions on the screen.
    """
    # Setup status box
    cv2.rectangle(image, (0, 0), (550, 100), (245, 117, 16), -1)

    # Rep data
    cv2.putText(
        image,
        "REPS",
        (15, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(counter),
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Stage data
    cv2.putText(
        image,
        "STAGE",
        (65, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        stage,
        (60, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    
    # Max angle data
    cv2.putText(
        image,
        "MAX ANGLE",
        (300, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{max_angle:.2f}",
        (250, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

cpdef void display_timer(cnp.ndarray[cnp.uint8_t, ndim=3] image, double elapsed_time):
    """
    Display the elapsed time on the screen.
    """
    cdef:
        int x = 800
        int y = 60
    cv2.putText(
        image,
        f"Time: {elapsed_time:.2f} s",
        (x, y),
        # (10, 60), # for testing only
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

cpdef void draw_landmarks_and_connections(cnp.ndarray[cnp.uint8_t, ndim=3] image, results):
    """
    Draw the landmarks and connections on the image.
    """
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

cpdef double get_real_time_from_frames(int frame_counter, double fps):
    """
    Calculate the real-time from the frame counter and frames per second (fps).
    """
    return frame_counter / fps

cpdef bint check_completion(int counter, str stage):
    # Check if 5 reps are completed
    return counter >= 5 and stage == "down"
        

import mediapipe as mp
mp_pose = mp.solutions.pose

# Make a OOP class for Hand Violation
# cdef class Violation:
#     cdef str category
#     cdef str timestamp
#     cdef str reason

#     def __init__(self, category, timestamp, reason):
#         self.category = category
#         self.timestamp = timestamp
#         self.reason = reason
#     def __str__(self):
#         return f"Violation(category={self.category}, timestamp={self.timestamp}, reason={self.reason})"
#     def display(self):
#         print(f"Category: {self.category}")
#         print(f"Timestamp: {self.timestamp}")
#         print(f"Reason: {self.reason}")

cpdef sit_stand_overall(str video_path, bint display):
    '''
        display flag should only be set for debugging purposes:
        Draws on the image for visualization purposes
    '''
    # Initialize variables for counter logic

    cdef:
        int counter = 0
        object stage = None
        int confirm_frames = 5
        int stage_counter = 0
        double max_angle_per_rep = 0
        double last_angle = 0
        double up_stage_threshold_angle = 135
        double down_stage_threshold_angle = 105
        double elbow_angle_threshold = 50
    # Initialize variables for optical flow
        cnp.ndarray prev_frame = None
        cnp.ndarray prev_hip = None
        cnp.ndarray prev_knee = None
        cnp.ndarray prev_ankle = None
    # for optical flow postprocessing
        dict joint_history = {
        "HIP": [],
        "KNEE": [],
        "ANKLE": [],
        }

        dict joint_displacement_history = {
        "HIP": [],
        "KNEE": [],
        "ANKLE": [],
        }
        dict joint_velocity_history = {
        "HIP": [],
        "KNEE": [],
        "ANKLE": [],
        }
        list max_angles = []
        int frames_after_start = 0
        double frame_rate = -1 # Uninitialised state
        double frame_time = -1 # Uninitialised state
        bint timer_started = False
        double start_time = -1 # Uninitialised state
        double elapsed_time = 0
        bint finished = False
        list rep_durations = []  # List to store the duration of each repetition
        double rep_start_time = -1  # Uninitialised state
        double rep_duration = -1
        double last_rep_end_time = 0  # To store the end time of the last repetition
        list violations = []
    # Create a 2D Cython array for the keypoints
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Cannot open video file.")
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / frame_rate
    except ValueError as e:
        logging.error(e)
        return
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and not finished:
            ret, frame = cap.read()
            if not ret:
                logging.warning("No frame captured from the video source.")
                break
            frame_height, frame_width, _ = frame.shape
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if timer_started:
                elapsed_time = get_real_time_from_frames(frames_after_start, frame_rate)
                frames_after_start += 1
            
            # Process the frame
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates multiplied by frame dimensions for optical flow and angle calculation
                hip, knee, ankle, shoulder, wrist, elbow = get_landmark_coordinates(
                    landmarks, frame_width, frame_height, "LEFT"
                )
                time_after_start = get_real_time_from_frames(frames_after_start, frame_rate)
                joint_history["HIP"].append((time_after_start, hip))
                joint_history["KNEE"].append((time_after_start, knee))
                joint_history["ANKLE"].append((time_after_start, ankle))
                # Calculate angle for sit-stand logic
                if stage_counter >= confirm_frames:
                    hand_angle = calculate_angle(shoulder, elbow, wrist)
                    if hand_angle > elbow_angle_threshold:
                        violations.append(("HAND", elapsed_time, f"Hand angle violation: {hand_angle:.2f} degrees"))
                        logging.info(f"Hand angle violation detected: {hand_angle:.2f} degrees.")
                angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)
                display_knee_and_hip_angle(image, angle, knee, hip_angle, hip)
                if prev_hip is not None and prev_knee is not None and prev_ankle is not None:
                    # Calculate displacements
                    # Optical flow visualization
                    joint_displacement_history = calculate_and_draw_joint_displacement(
                        prev_frame,
                        [prev_hip, prev_knee, prev_ankle],
                        [hip, knee, ankle],
                        image,
                        joint_displacement_history,
                        elapsed_time,
                    )
                    # =============== Runs only once ========================
                    # Get most recent displacement from history
                    hip_displacement = joint_displacement_history["HIP"][-1][1]
                    knee_displacement = joint_displacement_history["KNEE"][-1][1]
                    ankle_displacement = joint_displacement_history["ANKLE"][-1][1]

                    # Determine if the action timer should start
                    # Only ran once to get the start time
                    if not timer_started and should_start_timer(
                        hip_angle,
                        hip_displacement,
                        knee_displacement,
                        ankle_displacement,
                    ):
                        timer_started = True
                        logging.info(f"Timer started at {elapsed_time:.2f} seconds.")
                prev_frame = frame.copy()
                prev_hip, prev_knee, prev_ankle = hip, knee, ankle

                # Counting logic
                if stage is None:
                    stage = "up" if angle > up_stage_threshold_angle else "down"

                if timer_started and stage == "down" and angle < up_stage_threshold_angle:
                    if rep_start_time == -1:
                        rep_start_time = elapsed_time  # Only set once per rep to avoid resetting
                # Transition from DOWN to UP
                if stage == "down" and angle > up_stage_threshold_angle:
                    stage_counter += 1
                    if stage_counter >= confirm_frames:
                        stage = "up"
                        stage_counter = 0
                        counter += 1
                        logging.info(f"Transitioned to up. Total reps: {counter}")
                        max_angles.append(max_angle_per_rep)
                        max_angle_per_rep = 0  # Reset max angle for the new repetition
                        rep_duration = elapsed_time - last_rep_end_time
                        rep_durations.append(rep_duration)  # Store the duration of the rep
                        last_rep_end_time = elapsed_time  # Set end time for the next rep
                        logging.info(
                            f"Repetition {counter} completed in {rep_duration:.2f} seconds."
                        )
                # Transition from UP to DOWN
                elif stage == "up" and angle < down_stage_threshold_angle:
                    stage_counter += 1
                    if stage_counter >= confirm_frames:
                        stage = "down"
                        stage_counter = 0
                        logging.info("Transitioned to down.")
                # Check if 5 reps are completed
                if check_completion(counter, stage):
                    logging.info(f"5 repetitions completed in {elapsed_time:.2f} seconds.")
                    finished = True
                    # break
                # Update max angle
                if angle > max_angle_per_rep:
                    max_angle_per_rep = angle
                last_angle = angle
                
                if display:
                    display_information(image, counter, stage, max_angle_per_rep)
                    draw_landmarks_and_connections(image, results)
            if display:
                display_timer(image, elapsed_time)
                cv2.imshow("5 Rep Sit Stand Test", image)
                            
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        summarise_results(counter, elapsed_time, rep_durations, violations, max_angles)

        return "5 Sit Stand", determine_failure(
            elapsed_time, counter, False
        ),counter, elapsed_time, rep_durations, violations, max_angles