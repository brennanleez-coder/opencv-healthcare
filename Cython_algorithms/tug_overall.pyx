import cv2
import mediapipe as mp
cimport numpy as cnp
import numpy as np
import logging
from libc.math cimport sqrt, atan2, fabs, pi
from scipy.stats import circmean, circstd
cv2.setNumThreads(1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


cdef class Point:
    cdef double _x, _y

    def __init__(self, double x, double y):
        self._x = x
        self._y = y
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    def __str__(self):
        return f'({self._x}, {self._y})'

cpdef calculate_angle(Point a, Point b, Point c):
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
    radian_a = atan2(c.y - b.y, c.x - b.x)
    radian_b = atan2(a.y - b.y, a.x - b.x)
    angle = fabs((radian_a - radian_b) * 180.0 / pi)
    
    if angle > 180.0:
        angle = 360.0 - angle

    return angle

cpdef calculate_distance(double[:] p1, double[:] p2):
    """
    Calculates the Euclidean distance between two points in 2D space.
    """
    cdef double dx = p2[0] - p1[0]
    cdef double dy = p2[1] - p1[1]
    return sqrt(dx * dx + dy * dy)

cpdef double calculate_horizontal_distance(Point point1, Point point2, double frame_width):
    return abs(point1._x - point2._x) * frame_width

cpdef double calculate_vertical_distance(Point point1, Point point2, double frame_height):
    return abs(point1._y - point2._y) * frame_height

cpdef Point calculate_central_point(landmark1, landmark2):
    return Point((landmark1.x + landmark2.x) / 2, (landmark1.y + landmark2.y) / 2)

cpdef double calculate_foot_slope(Point ankle, Point toe):
    cdef double slope
    slope = (toe.y - ankle.y) / (toe.x - ankle.x)
    logging.info(f"Foot slope: {slope}")    
    return slope

cpdef double calculate_magnitude(double x1, double y1, double x2, double y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

cpdef dict get_keypoints(landmarks):
    cdef Point left_hip, right_hip, nose, left_ankle, right_ankle, left_toe, right_toe
    cdef Point left_shoulder, right_shoulder, left_knee, right_knee
    cdef:
        int NOSE = 0
        int LEFT_SHOULDER = 11
        int RIGHT_SHOULDER = 12
        int LEFT_HIP = 23
        int RIGHT_HIP = 24
        int LEFT_KNEE = 25
        int RIGHT_KNEE = 26
        int LEFT_ANKLE = 27
        int RIGHT_ANKLE = 28
        int LEFT_FOOT_INDEX = 31
        int RIGHT_FOOT_INDEX = 32

    # Extract key points and initialize Point objects
    left_hip = Point(landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y)
    right_hip = Point(landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y)
    nose = Point(landmarks[NOSE].x, landmarks[NOSE].y)
    left_ankle = Point(landmarks[LEFT_ANKLE].x, landmarks[LEFT_ANKLE].y)
    right_ankle = Point(landmarks[RIGHT_ANKLE].x, landmarks[RIGHT_ANKLE].y)
    left_toe = Point(landmarks[LEFT_FOOT_INDEX].x, landmarks[LEFT_FOOT_INDEX].y)
    right_toe = Point(landmarks[RIGHT_FOOT_INDEX].x, landmarks[RIGHT_FOOT_INDEX].y)
    left_shoulder = Point(landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y)
    right_shoulder = Point(landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y)
    left_knee = Point(landmarks[LEFT_KNEE].x, landmarks[LEFT_KNEE].y)
    right_knee = Point(landmarks[RIGHT_KNEE].x, landmarks[RIGHT_KNEE].y)

    # Create a dictionary to store the key points
    cdef dict keypoints = {
        "NOSE": nose,
        "LEFT_SHOULDER": left_shoulder,
        "RIGHT_SHOULDER": right_shoulder,
        "LEFT_HIP": left_hip,
        "RIGHT_HIP": right_hip,
        "LEFT_KNEE": left_knee,
        "RIGHT_KNEE": right_knee,
        "LEFT_ANKLE": left_ankle,
        "RIGHT_ANKLE": right_ankle,
        "LEFT_TOE": left_toe,
        "RIGHT_TOE": right_toe
    }
    return keypoints

# Declare types for input arguments and constants for better performance
cpdef bint should_start_timer(double hip_angle, knee_angle):
    cdef double threshold_angle = 100.0
    # cdef double hip_thresh = 3.0
    # cdef double knee_thresh = 3.0
    # cdef double ankle_thresh = 1.5
    cdef bint hip_trigger

    # Determine whether the hip angle is below the threshold
    hip_trigger = hip_angle < threshold_angle
    knee_trigger = knee_angle > threshold_angle
    logging.info(f"{'Timer started' if hip_trigger else 'Timer not started'} - hip_angle: {hip_angle}")

    # Return the hip_trigger (currently ignoring displacements based on your commented code)
    return hip_trigger or knee_trigger


cdef enum Phase:
    STAND = 1
    WALK_TO = 2
    TURN = 3
    WALK_BACK = 4
    SIT = 5
    NO_STAGE = 6


import logging


# Function to handle the transition between phases
cpdef Phase begin_phase(Phase current_phase, Phase next_phase):
    # Check transitions between phases
    if current_phase == Phase.STAND and next_phase == Phase.WALK_TO:
        logging.info("STAND_UP -> WALK_TO")
    elif current_phase == Phase.WALK_TO and next_phase == Phase.TURN:
        logging.info("WALK_TO -> TURN")
    elif current_phase == Phase.TURN and next_phase == Phase.WALK_BACK:
        logging.info("TURN -> WALK_BACK")
    elif current_phase == Phase.WALK_BACK and next_phase == Phase.SIT:
        logging.info("WALK_BACK -> SIT")
    
    # Update the current phase
    new_current_phase = next_phase
    logging.info(f"Begin Phase Method - Current phase: {new_current_phase}")
    
    # Return the new current phase
    return new_current_phase

cpdef process_test(
    str video_path,
    double person_height_in_cm,
    double distance_required_in_cm,
    bint debug,
):
    logging.info("Starting video processing")
    cdef:
        int start_frame_id = -1
        int end_frame_id = -1
        object start_position = None
        double distance_walked = 0
        bint timer_started = False
        list strides = []
        object central_point = None
        double pixels_to_cm_ratio = 0
        double start_line = -1
        bint is_start_line_set = False
        double start_line_offset = 0.01
        cnp.ndarray previous_gray = None
        cnp.ndarray previous_keypoints = None
        bint foot_contact_start = False
        bint foot_off_ground = False
        bint reset_needed = False
        double initial_foot_slope = 0
        list keypoints_over_time = []
        double fps = -1
        int frame_counter = 0

        # Sit Stand Variables
        int sit_stand_stage = -1
        object stage = None
        int confirm_frames = 5
        int stage_counter = 0
        double up_stage_threshold_angle = 135
        double down_stage_threshold_angle = 105

        # Walk To Variables
        double walk_to_previous_distance = -1
        object walk_back_start_position = None

        # Phase flags
        Phase current_phase = Phase.STAND
        int stand_start_frame = -1
        int stand_end_frame = -1
        int walk_to_start_frame = -1
        int walk_to_end_frame = -1
        int turn_start_frame = -1
        int turn_end_frame = -1
        int walk_back_start_frame = -1
        int walk_back_end_frame = -1
        int sit_start_frame = -1
        int sit_end_frame = -1

        # Misc
        # Set the starting position for the text and the vertical line height
        int starting_x = 10
        int starting_y = 30  # Start near the top-left corner of the frame
        int line_height = 40  # Space between lines to avoid overlap
        # Optical Flow variables
        cnp.ndarray p1, st, err
        double new_x, new_y, old_x, old_y
        double motion_magnitude, motion_angle
        dict keypoint_motion = {}
        int i
                    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Cannot open video file.")
        if fps == -1:
            fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            logging.error(f"Failed to retrieve FPS")
            
    except ValueError as e:
        logging.error(e)
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("End of video or cannot read the video file")
            break

        current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logging.info(
            f"===================== Processing frame {current_frame_id} =================================="
        )
        if results.pose_landmarks:
            if debug:
                mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                    )
            landmarks = results.pose_landmarks.landmark
            keypoints = get_keypoints(landmarks)

            logging.info(f"--- Misc --- ")
            logging.info(f"--- Angles ---")
            logging.info(f"--- Hip and Knee Angles ---")
            hip_angle = calculate_angle(
                    keypoints["RIGHT_SHOULDER"],
                    keypoints["RIGHT_HIP"],
                    keypoints["RIGHT_KNEE"],
                )
            logging.info(f"Right hip angle: {hip_angle}")
            knee_angle = calculate_angle(
                keypoints["RIGHT_HIP"],
                keypoints["RIGHT_KNEE"],
                keypoints["RIGHT_ANKLE"],
            )
            logging.info(f"Right knee angle: {knee_angle}")
            logging.info(f"--- End of Angles ---")

            logging.info(f"--- Central Points ---")
            central_ankle = calculate_central_point(keypoints["LEFT_ANKLE"], keypoints["RIGHT_ANKLE"])
            logging.info(f"Central ankle point: {central_ankle}")
            central_point = calculate_central_point(
                keypoints["LEFT_HIP"], keypoints["RIGHT_HIP"]
            )
            logging.info(f"Central point: {central_point}")
            if not timer_started:
                start_position = central_point
                logging.info(f"Timer not started - Start position: {start_position}")
            logging.info(f"Body Central point: {central_point}")
            logging.info(f"--- End of Central Points ---")

            estimated_height_in_px = calculate_vertical_distance(keypoints["NOSE"], central_ankle, frame.shape[0])
            if pixels_to_cm_ratio == 0:
                pixels_to_cm_ratio = estimated_height_in_px / person_height_in_cm
                logging.info(f"Pixels to cm ratio: {pixels_to_cm_ratio} is set")
            if sit_stand_stage == -1:
                sit_stand_stage = 1 if knee_angle > up_stage_threshold_angle else 0
                logging.info(f"Initial sit_stand stage: {sit_stand_stage}")

            if debug:
                cv2.putText(
                            frame,
                            f"Phase: {current_phase}",
                            (starting_x, starting_y),  # Starting position
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),  # Green
                            2,
                            cv2.LINE_AA,
                        )

            
            # Main Processing when timer has started
            if timer_started and start_frame_id is not None:
                logging.info(f"Timer started at frame: {start_frame_id}")
                elapsed_time = (current_frame_id - start_frame_id) / fps
                logging.info(f"Distance walked: {distance_walked} cm")
                if end_frame_id is None:
                    if walk_back_start_position is not None and (current_phase == Phase.WALK_BACK or current_phase == Phase.SIT):
                        logging.info(f"Walk back start position: {walk_back_start_position} - Central point: {central_point}")
                        distance_walked += calculate_horizontal_distance(
                            walk_back_start_position, central_point, frame.shape[1]
                        ) / pixels_to_cm_ratio
                        walk_back_start_position = central_point
                    else:
                        distance_walked = calculate_horizontal_distance(start_position, central_point, frame.shape[1])/ pixels_to_cm_ratio
                    
                    if initial_foot_slope is None:
                        # log right ankle and right toe
                        initial_foot_slope = calculate_foot_slope(keypoints["RIGHT_ANKLE"], keypoints["RIGHT_TOE"]
                        )
                        right_slope = initial_foot_slope
                        logging.info(f"Initial foot slope: {initial_foot_slope}")
                    else:
                        right_slope = calculate_foot_slope(keypoints["RIGHT_ANKLE"], keypoints["RIGHT_TOE"])
                        logging.info(f"Right foot slope: {right_slope}")
                    # Detect foot contact with the ground
                    if abs(right_slope) < initial_foot_slope:
                        logging.info(f"Right foot is parallel to the ground")
                        if not foot_contact_start and not reset_needed:
                            # First ground contact
                            foot_contact_start = True
                            foot_off_ground = False
                            initial_contact_position = keypoints["RIGHT_TOE"]
                            logging.info("First ground contact detected")

                        elif foot_off_ground and reset_needed:
                            # Second ground contact, calculate stride
                            final_contact_position = keypoints["RIGHT_TOE"]
                            stride_length = (
                                calculate_horizontal_distance(
                                    final_contact_position,
                                    initial_contact_position,
                                    frame.shape[1],
                                )
                                / pixels_to_cm_ratio
                            )
                            strides.append(stride_length)
                            logging.info(f"Stride length recorded: {stride_length} cm")

                            # Reset flags for the next stride detection
                            reset_needed = False
                            foot_contact_start = False
                            foot_off_ground = False
                            initial_contact_position = keypoints["RIGHT_TOE"]
                    else:
                        # Foot off ground
                        if foot_contact_start and not reset_needed:
                            foot_off_ground = True
                            logging.info("Foot is off the ground")

                            # Allow reset to detect the next stride
                            reset_needed = True

                    if debug:
                        cv2.putText(
                            frame,
                            f"Right foot slope: {right_slope}",
                            (starting_x, starting_y + 3 * line_height),  # Offset by 3 * line_height
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 0),  # Yellow
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            f"Foot touching ground: {'YES' if foot_contact_start and abs(right_slope) < 0.9 else 'NO'}",
                            (starting_x, starting_y + 4 * line_height),  # Offset by 4 * line_height
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 0),  # Yellow
                            2,
                            cv2.LINE_AA,
                        )
                    # Calculate motion vectors using optical flow
                    logging.info(f"--- Optical Flow ---")
                    if timer_started and previous_gray is not None and previous_keypoints is not None:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_keypoints, None)

                        for i in range(p1.shape[0]):
                            new_x, new_y = p1[i,0,0], p1[i,0,1]
                            old_x, old_y = previous_keypoints[i,0,0], previous_keypoints[i,0,1]
                            motion_magnitude = calculate_magnitude(new_x, new_y, old_x, old_y)
                            motion_angle = atan2(new_y - old_y, new_x - old_x)
                            logging.info(f"{list(keypoints.keys())[i]}: {motion_magnitude}, {motion_angle}")
                            keypoint_motion[list(keypoints.keys())[i]] = (motion_magnitude, motion_angle)

                            if debug:
                                cv2.arrowedLine(
                                    frame,
                                    (int(old_x), int(old_y)),
                                    (int(new_x), int(new_y)),
                                    (0, 255, 0),
                                    2,
                                    tipLength=0.5
                                )
                        keypoints_over_time.append(keypoint_motion)
                        keypoint_motion = {}
                        previous_keypoints = p1
                    else:
                        previous_keypoints = np.array([[kp.x * frame.shape[1], kp.y * frame.shape[0]] for kp in keypoints.values()], dtype=np.float32).reshape(-1, 1, 2)
                    previous_gray = current_gray
                    logging.info(f"--- End of Optical Flow ---")
                if debug:
                    cv2.putText(
                        frame,
                        f"Time: {elapsed_time:.2f} sec",
                        (starting_x, starting_y + line_height),  # Offset by line_height
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # Green
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Distance: {distance_walked:.2f} cm",
                        (starting_x, starting_y + 2 * line_height),  # Offset by 2 * line_height
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # Green
                        2,
                        cv2.LINE_AA,
                    )
            else:
                if debug:
                    cv2.putText(frame,f"Timer not started",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2,cv2.LINE_AA,)

            
            if current_phase == Phase.STAND and not timer_started:
                logging.info(f"---Timer not started - Expected Phase: Stand - Actual: {current_phase}---")
                if not timer_started and should_start_timer(hip_angle, knee_angle):
                    timer_started = True
                    start_frame_id = current_frame_id
                    logging.info(f"Timer started - Start frame set - frame {current_frame_id}")

            if (timer_started and sit_stand_stage == 0):
                if stand_start_frame == -1:
                    stand_start_frame = current_frame_id  # Only set once per rep to avoid resetting
                    logging.info(f"Stand up phase started at frame: {stand_start_frame}")
                    
            # Stand Phase processing
            # Transition from DOWN to UP
            if (current_phase == Phase.STAND and
                timer_started and
                sit_stand_stage == 0 and
                knee_angle > up_stage_threshold_angle
                ):
                logging.info(f"---Expected Phase: Stand - Actual: {current_phase}---")
                stage_counter += 1
                logging.info(f"Stage counter: {stage_counter}")
                if stage_counter >= confirm_frames:
                    logging.info(f"Stage is confirmed. Transitioning to up stage")
                    sit_stand_stage = 1
                    stage_counter = 0
                    stand_end_frame = current_frame_id
                    logging.info(f"Stand up phase completed at frame: {stand_end_frame}")
                    walk_to_start_frame = current_frame_id
                    logging.info(f"Walk to phase started at frame: {walk_to_start_frame}")
                    returned_current_phase = begin_phase(Phase.STAND,Phase.WALK_TO)
                    current_phase = returned_current_phase
                logging.info(f"------")
                
            # Walk To Phase processing
            if current_phase == Phase.WALK_TO and timer_started:
                logging.info(f"---Expected Phase: Walk To - Actual: {current_phase}---")
                # Theres nothing much to do, just track the end of the walk_to_phase
                if walk_to_start_frame == -1:
                    walk_to_start_frame = current_frame_id
                current_distance = distance_walked
                logging.info(f"Total Distance in walk_to_phase: {current_distance}")
                
                # Transition from WALK_TO to TURN
                if walk_to_previous_distance > current_distance:
                    stage_counter += 1
                else:
                    stage_counter = 0  # Reset if the condition isn't met

                logging.info(f"Walk to previous distance: {walk_to_previous_distance}, current distance: {current_distance}")
                logging.info(f"Stage counter: {stage_counter}")
                
                if stage_counter >= 5:  # Only allow transition if the condition has been true for 5 consecutive frames
                    distance_condition_for_transition = True
                    stage_counter = 0
                else:
                    distance_condition_for_transition = False

                logging.info(f"Condition for transition (Walk to -> Turn): {distance_condition_for_transition}")

                # stage_counter +=1
                # if stage_counter >= confirm_frames:
                #     distance_condition_for_transition = walk_to_previous_distance > current_distance
                #     stage_counter = 0
                
                if (walk_to_start_frame != -1 and walk_to_end_frame == -1 and distance_condition_for_transition):
                    if walk_back_start_position is None:
                        walk_back_start_position = central_point
                    walk_to_end_frame = current_frame_id # End of walk_to_phase
                    turn_start_frame = current_frame_id # Start of turn_phase
                    returned_current_phase = begin_phase(Phase.WALK_TO,Phase.TURN)
                    current_phase = returned_current_phase
                    logging.info(f"Walk to phase completed at frame: {walk_to_end_frame}, Turn phase started at frame: {turn_start_frame}")
                walk_to_previous_distance = current_distance
                logging.info(f"------")

            
            # Turn Phase processing
            if current_phase == Phase.TURN and timer_started:
                logging.info(f"---Expected Phase: Turn - Actual: {current_phase}---")
                # Transition from TURN to WALK_BACK
                if turn_start_frame != -1 and turn_end_frame == -1:
                    stage_counter += 1
                    if stage_counter >= confirm_frames:
                        logging.info(f"Stage is confirmed. Transitioning to walk back stage")
                        stage_counter = 0
                        turn_end_frame = current_frame_id
                        returned_current_phase= begin_phase(Phase.TURN, Phase.WALK_BACK)
                        current_phase = returned_current_phase
                        logging.info(f"Turn phase completed at frame: {turn_end_frame}")
                logging.info(f"------")
                
            # Walk Back Phase processing
            if current_phase == Phase.WALK_BACK and timer_started:
                logging.info(f"---Expected Phase: Walk Back - Actual: {current_phase}---")
                if walk_back_start_frame == -1:
                    walk_back_start_frame = current_frame_id
                    logging.info(f"Walk back phase started at frame: {walk_back_start_frame}")
                logging.info(f"WALK_BACK distance: {distance_walked}")
                logging.info(f"------")
                if sit_stand_stage == 1 and knee_angle < down_stage_threshold_angle:
                    walk_back_end_frame = current_frame_id
                    logging.info(f"Walk back phase completed at frame: {walk_back_end_frame}")
                    sit_start_frame = current_frame_id
                    logging.info(f"Sit phase started at frame: {sit_start_frame}")
                    stage_counter += 1
                    if stage_counter >= confirm_frames:
                        logging.info(f"Stage is confirmed. Transitioning to down stage")
                        sit_stand_stage = 0
                        stage_counter = 0
                        returned_current_phase = begin_phase(Phase.WALK_BACK,Phase.SIT)
                        current_phase = returned_current_phase
                logging.info("------")
            
            # Sit Phase processing
            if current_phase == Phase.SIT and timer_started:
                logging.info(f"---Expected Phase: Sit - Actual: {current_phase}---")
                logging.info(f"Timer stopped at frame - Begin Post Processing : {sit_start_frame}, End frame: {end_frame_id}")
                logging.info(f"---END OF TEST---")
                sit_end_frame = current_frame_id
                end_frame_id = sit_end_frame
                logging.info(f"Processing Completed at frame: {end_frame_id}")
                timer_started = False
                break
        if debug:
            # Display the frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                logging.info("Process interrupted by user")
                break
        # Trigger to start the timer
    cap.release()
    cv2.destroyAllWindows()

    logging.info("================================= Post Processing =================================")

    cdef:
        double average_speed
        double average_stride_length
        dict keypoint_mean_magnitudes = {}
        dict keypoint_std_devs = {}
        dict keypoint_circular_mean = {}
        dict keypoint_circular_std = {}
        double strides_per_second
        int j
        double total_sum
        int n_frames = len(keypoints_over_time)
        double[:] std_devs_values = np.zeros(n_frames, dtype=np.float64)
        double[:] circ_mean_values = np.zeros(n_frames, dtype=np.float64)
        double[:] circ_std_values = np.zeros(n_frames, dtype=np.float64)
        list keypoints_names= list(keypoints_over_time[0].keys())
    # Handle strides
    logging.info(f"--- Sanity Checks ---")
    logging.info(f"Strides: {strides}")
    logging.info(f"Elapsed time: {elapsed_time}")
    logging.info(f"Start frame ID: {start_frame_id}")
    logging.info(f"End frame ID: {end_frame_id}")
    logging.info(f"Current frame ID: {current_frame_id}")
    logging.info(f"FPS: {fps}")
    logging.info(f"--- End of Sanity Checks ---")
    logging.info(f"--- Results ---")

    logging.info(f"--- Stride Information ---")
    if strides not in [None, []]:
        average_stride_length = np.mean(strides)
        logging.info(f"Average stride length: {average_stride_length} cm")
        # strides_per_second = len(strides) / elapsed_time if elapsed_time > 0 else 0
        # logging.info(f"Strides per second: {strides_per_second}")
    else:
        average_stride_length = 0
        logging.info("No strides detected")
    logging.info(f"--- End of Stride Information ---")

    logging.info(f"--- Key Points Information ---")
    # Handle keypoints over time
    if keypoints_over_time not in [None, []]:
        # Mean magnitudes
        for keypoint in keypoints_names:
            total_sum = 0
            for j in range(n_frames):
                total_sum += keypoints_over_time[j][keypoint][0]  # Sum the first element for each frame
                std_devs_values[j] = keypoints_over_time[j][keypoint][0]
                circ_mean_values[j] = keypoints_over_time[j][keypoint][1]
                circ_std_values[j] = keypoints_over_time[j][keypoint][1]
            logging.info(f"---- {keypoint} ----")
            keypoint_mean_magnitudes[keypoint] = total_sum / n_frames
            logging.info(f"Mean magnitude for {keypoint}: {keypoint_mean_magnitudes[keypoint]}")
            keypoint_std_devs[keypoint] = np.std(std_devs_values)
            logging.info(f"Standard deviation for {keypoint}: {keypoint_std_devs[keypoint]}")
            keypoint_circular_mean[keypoint] = circmean(circ_mean_values, high=np.pi, low=-np.pi)
            logging.info(f"Circular mean for {keypoint}: {keypoint_circular_mean[keypoint]}")
            keypoint_circular_std[keypoint] = circstd(circ_std_values, high=np.pi, low=-np.pi)
            logging.info(f"Circular standard deviation for {keypoint}: {keypoint_circular_std[keypoint]}")
    else:
        keypoint_mean_magnitudes = {}
        keypoint_std_devs = {}
        keypoint_circular_mean = {}
        keypoint_circular_std = {}
        logging.info("No keypoints detected")
    logging.info(f"--- End of Key Points Information ---")

    logging.info(f"--- Time and Speed Information ---")
    # Handle time and speed calculation
    if timer_started and start_frame_id is not None:
        elapsed_time = (current_frame_id - start_frame_id) / fps
        logging.info(f"Elapsed time: {elapsed_time} seconds")
        average_speed = distance_walked / elapsed_time if elapsed_time > 0 else 0
        logging.info(f"Average speed: {average_speed} cm/second")
    else:
        elapsed_time = 0
        average_speed = 0
        logging.info("Problems detected, unable to calculate")
    logging.info(f"--- End of Time and Speed Information ---")
    logging.info(f"--- End of Results ---")

    return "Gait Speed Walk Test", distance_walked,elapsed_time, average_speed, average_stride_length, keypoint_mean_magnitudes, keypoint_std_devs, keypoint_circular_mean, keypoint_circular_std