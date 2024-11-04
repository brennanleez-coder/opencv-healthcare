import cv2
import mediapipe as mp
import numpy as np
import logging
from enum import Enum

import cProfile
from scipy.stats import circmean, circstd


cv2.setNumThreads(1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    # define print method
    
    def __str__(self):
        return f'({self.x}, {self.y})'

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
                # uncomment if 3d Point
                #    + (point1['z'] - point2['z'])**2)
                
def calculate_horizontal_distance(point1, point2, frame_width):
    return abs(point1.x - point2.x) * frame_width

def calculate_vertical_distance(point1, point2, frame_height):
    return abs(point1.y - point2.y) * frame_height

def calculate_central_point(landmark1, landmark2):
    return Point((landmark1.x + landmark2.x) / 2, (landmark1.y + landmark2.y) / 2)

def low_pass_filter(new_value, prev_value, alpha=0.8):
    return alpha * new_value + (1 - alpha) * prev_value

def calculate_foot_slope(ankle, toe):
    slope = (toe.y - ankle.y) / (toe.x - ankle.x)
    logging.info(f"Right foot slope: {slope}")

    return slope
def calculate_angle(a,b,c):
    """
    Calculate the angle formed at point 'b' by the line segments a-b and b-c.
    `b` is the midpoint of `a` and `c` (e.g., left hip, left knee, and left ankle).

    Parameters:
    a, b, c (double[:]): Coordinates of the points.

    Returns:
    double: The angle in degrees.
    """
    radian_a = np.arctan2(c.y - b.y, c.x - b.x)
    radian_b = np.arctan2(a.y - b.y, a.x - c.x)
    angle = abs((radian_a - radian_b) * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle    
def calculate_joint_displacement(prev_frame, prev_points, curr_points, image, joint_displacement_history):
    joint_names = ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")  # Names of the joints
    prev_hip, prev_knee, prev_ankle = prev_points
    curr_hip, curr_knee, curr_ankle = curr_points
    if prev_frame is not None:
        for prev_point, curr_point, _, joint_name in zip(
            [prev_hip, prev_knee, prev_ankle],
            [curr_hip, curr_knee, curr_ankle],
            [(0,0,255)]*3,
            joint_names,
        ):
            displacement = calculate_distance(prev_point, curr_point)
            logging.info(f"{joint_name} displacement: {displacement}")
            joint_displacement_history[joint_name].append(displacement)
    return joint_displacement_history

def should_start_timer(hip_angle, knee_angle, hip_displacement = 0, knee_displacement = 0, ankle_displacement = 0):
    threshold_angle = 100.0
    hip_thresh = 3.0
    knee_thresh = 3.0
    ankle_thresh = 1.5
    # logging.info(f"Timer not started - hip_angle: {hip_angle} - hip_displacement: {hip_displacement} - knee_displacement: {knee_displacement} - ankle_displacement: {ankle_displacement}")
    hip_trigger = hip_angle < threshold_angle
    knee_trigger = knee_angle > threshold_angle
    logging.info(f"{'Timer started' if hip_trigger else 'Timer not started'} - hip_angle: {hip_angle} - hip_displacement: {hip_displacement} - knee_displacement: {knee_displacement} - ankle_displacement: {ankle_displacement}")
    return (
        hip_trigger or knee_trigger
        # and 
        # (
        #     hip_displacement > hip_thresh or
        #     knee_displacement > knee_thresh or
        #     ankle_displacement > ankle_thresh
        # )
    )
def get_keypoints(landmarks):
        # Convert NormalizedLandmark to Point
        left_hip = Point(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
        )
        right_hip = Point(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
        )
        nose = Point(
            landmarks[mp_pose.PoseLandmark.NOSE].x,
            landmarks[mp_pose.PoseLandmark.NOSE].y,
        )
        left_ankle = Point(
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
        )
        right_ankle = Point(
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
        )
        left_toe = Point(
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
        )
        right_toe = Point(
            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,
        )
        left_shoulder = Point(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        )
        right_shoulder = Point(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
        )
        left_knee = Point(
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
        )
        right_knee = Point(
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
        )
        keypoints = {
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
            "RIGHT_TOE": right_toe,
            
        }
        return keypoints
    
class Phase(Enum):
    STAND = 1
    WALK_TO = 2
    TURN = 3
    WALK_BACK = 4
    SIT = 5
    NO_STAGE = 6
def begin_phase(current_phase, next_phase):
    if current_phase == Phase.STAND and next_phase == Phase.WALK_TO:
        logging.info("STAND_UP -> WALK_TO")
    elif current_phase == Phase.WALK_TO and next_phase == Phase.TURN:
        logging.info("WALK_TO -> TURN")
    elif current_phase == Phase.TURN and next_phase == Phase.WALK_BACK:
        logging.info("TURN -> WALK_BACK")
    elif current_phase == Phase.WALK_BACK and next_phase == Phase.SIT:
        logging.info("WALK_BACK -> SIT")
    new_current_phase = next_phase
    logging.info(f"Begin Phase Method - Current phase: {current_phase}")
    return new_current_phase

def process_tug(
    video_path,
    sit_down_height_in_cm,
    distance_required_in_cm,
    debug=True,
):
    logging.info(f"{sit_down_height_in_cm}, {distance_required_in_cm}")
    logging.info("Starting video processing")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.error("Cannot determine the frame rate (FPS) of the video.")
        return -1, -1, -1  # Default values for stride length, elapsed time, and speed

    # Variables for TUG
    start_frame_id = None
    end_frame_id = None
    start_position = None
    distance_walked = 0
    timer_started = False
    strides = []
    central_point = None
    pixels_to_cm_ratio = None

    previous_gray = None
    previous_keypoints = None
    foot_contact_start = False  # Flag to track first contact
    foot_off_ground = False  # Flag to track if foot is lifted off ground
    reset_needed = False  # Flag to reset after stride is recorded
    initial_foot_slope = None  # Initial slope of the foot
    keypoints_over_time = []


    # Sit Stand Variables
    sit_stand_stage = None
    confirm_frames = 5
    stage_counter = 0
    up_stage_threshold_angle = 135
    down_stage_threshold_angle = 105
    # Walk to variables
    walk_to_previous_distance = -1
    walk_back_start_position = None

    # Phase flags
    current_phase = Phase.STAND  # Initial phase
    stand_start_frame = -1
    stand_end_frame = -1
    walk_to_start_frame = -1
    walk_to_end_frame = -1
    turn_start_frame = -1
    turn_end_frame = -1
    walk_back_start_frame = -1
    walk_back_end_frame = -1
    sit_start_frame = -1
    sit_end_frame = -1
    
    # Misc
    # Set the starting position for the text and the vertical line height
    starting_x = 10
    starting_y = 30  # Start near the top-left corner of the frame
    line_height = 40  # Space between lines to avoid overlap

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
            # Draw pose landmarks
            if debug:
                mp_drawing.draw_landmarks(
                    frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )

            landmarks = results.pose_landmarks.landmark
            keypoints = get_keypoints(landmarks)
            logging.info(f"--- Keypoints ---")
            for key, value in keypoints.items():
                logging.info(f"{key}: {value}")

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
            central_ankle = calculate_central_point(
                keypoints["LEFT_ANKLE"], keypoints["RIGHT_ANKLE"]
            )
            logging.info(f"Central ankle point: {central_ankle}")
            # Fix central point of the person using hip points
            central_point = calculate_central_point(
                keypoints["LEFT_HIP"], keypoints["RIGHT_HIP"]
            )
            logging.info(f"Central point: {central_point}")
            if not timer_started:
                start_position = central_point
                logging.info(f"Timer not started - Start position: {start_position}")
            logging.info(f"Body Central point: {central_point}")
            logging.info(f"--- End of Central Points ---")
            

            # Calculate the height of the person in px
            estimated_height_in_px = calculate_vertical_distance(
                keypoints["NOSE"], central_ankle, frame.shape[0]
            )  # Possibly add a offset to the height
            logging.info(f"Estimated height: {estimated_height_in_px} px")
            if pixels_to_cm_ratio is None:
                pixels_to_cm_ratio = estimated_height_in_px / sit_down_height_in_cm
                logging.info(f"Pixels to cm ratio: {pixels_to_cm_ratio} is set")
            if sit_stand_stage is None:
                sit_stand_stage = "up" if knee_angle > up_stage_threshold_angle else "down"
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
                    if previous_gray is not None and previous_keypoints is not None:
                        logging.info(f"--- Optical Flow ---")
                        p1, st, err = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_keypoints, None)
                        current_frame_keypoints = {}
                        for i, (new, old) in enumerate(zip(p1, previous_keypoints)):
                            new_x, new_y = new.ravel()
                            old_x, old_y = old.ravel()
                            motion_magnitude = np.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)
                            motion_angle = np.arctan2(new_y - old_y, new_x - old_x)
                            key = list(keypoints.keys())[i]
                            current_frame_keypoints[key] = (motion_magnitude, motion_angle)
                            logging.info(f"{key} - Magnitude: {motion_magnitude}, Angle: {motion_angle} radians")
                            # Optionally, visualize the motion (debugging)
                            if debug:
                                cv2.arrowedLine(frame,(int(old_x), int(old_y)),(int(new_x), int(new_y)),(0, 255, 0),2,tipLength=0.5,)
                        logging.info(f"--- End Optical Flow ---")
                        keypoints_over_time.append(current_frame_keypoints)
                        previous_keypoints = p1
                    else:
                        previous_keypoints = np.array([[kp.x * frame.shape[1], kp.y * frame.shape[0]] for kp in keypoints.values()],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
                    previous_gray = current_gray
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

            # Trigger to start the timer
            if current_phase == Phase.STAND and not timer_started:
                logging.info(f"---Timer not started - Expected Phase: Stand - Actual: {current_phase}---")
                if not timer_started and should_start_timer(hip_angle, knee_angle):
                    timer_started = True
                    start_frame_id = current_frame_id
                    logging.info(f"Timer started - Start frame set - frame {current_frame_id}")

            if (timer_started and sit_stand_stage == "down"):
                if stand_start_frame == -1:
                    stand_start_frame = current_frame_id  # Only set once per rep to avoid resetting
                    logging.info(f"Stand up phase started at frame: {stand_start_frame}")
                    
            # Stand Phase processing
            # Transition from DOWN to UP
            if (current_phase == Phase.STAND and
                timer_started and
                sit_stand_stage == "down" and
                knee_angle > up_stage_threshold_angle
                ):
                logging.info(f"---Expected Phase: Stand - Actual: {current_phase}---")
                stage_counter += 1
                logging.info(f"Stage counter: {stage_counter}")
                if stage_counter >= confirm_frames:
                    logging.info(f"Stage is confirmed. Transitioning to up stage")
                    sit_stand_stage = "up"
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
                if sit_stand_stage == "up" and knee_angle < down_stage_threshold_angle:
                    walk_back_end_frame = current_frame_id
                    logging.info(f"Walk back phase completed at frame: {walk_back_end_frame}")
                    sit_start_frame = current_frame_id
                    logging.info(f"Sit phase started at frame: {sit_start_frame}")
                    stage_counter += 1
                    if stage_counter >= confirm_frames:
                        logging.info(f"Stage is confirmed. Transitioning to down stage")
                        sit_stand_stage = "down"
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

    cap.release()
    cv2.destroyAllWindows()
    logging.info(
        "================================= Post Processing ================================="
    )
    elapsed_time = (current_frame_id - start_frame_id) / fps
    logging.info(f"Elapsed time: {elapsed_time} seconds")
    if strides:
        average_stride_length = np.mean(strides)
        logging.info(f"strides: {strides}")
        logging.info(f"Average stride length: {average_stride_length} cm")
        strides_per_second = len(strides) / elapsed_time if elapsed_time > 0 else 0
        logging.info(f"Strides per second: {strides_per_second}")
    else:
        average_stride_length = 0
        logging.info("No strides detected")

    if keypoints_over_time:
        keypoint_mean_magnitudes = {
            keypoint: sum(frame[keypoint][0] for frame in keypoints_over_time)
            / len(keypoints_over_time)
            for keypoint in keypoints_over_time[0]
        }
        keypoint_std_devs = {
            keypoint: np.std([frame[keypoint][0] for frame in keypoints_over_time])
            for keypoint in keypoints_over_time[0]
        }
        keypoint_circular_mean = {
            keypoint: circmean(
                [frame[keypoint][1] for frame in keypoints_over_time],
                high=np.pi,
                low=-np.pi,
            )
            for keypoint in keypoints_over_time[0]
        }
        keypoint_circular_std = {
            keypoint: circstd(
                [frame[keypoint][1] for frame in keypoints_over_time],
                high=np.pi,
                low=-np.pi,
            )
            for keypoint in keypoints_over_time[0]
        }
    else:
        keypoint_std_devs = -1
        keypoint_mean_magnitudes = -1
        logging.info("No keypoints detected")

    if ((stand_start_frame != -1 and stand_end_frame != -1) and
        (walk_to_start_frame != -1 and walk_to_end_frame != -1) and
        (turn_start_frame != -1 and turn_end_frame != -1) and
        (walk_back_start_frame != -1 and walk_back_end_frame != -1) and
        (sit_start_frame != -1 and sit_end_frame != -1)
        ):
        stand_time = (stand_end_frame - stand_start_frame) / fps
        walk_to_time = (walk_to_end_frame - walk_to_start_frame) / fps
        turn_time = (turn_end_frame - turn_start_frame) / fps
        walk_back_time = (walk_back_end_frame - walk_back_start_frame) / fps
        sit_time = (sit_end_frame - sit_start_frame) / fps
        logging.info(f"Stand time: {stand_time} seconds - started at frame: {stand_start_frame}, ended at frame: {stand_end_frame}")
        logging.info(f"Walk to time: {walk_to_time} seconds - started at frame: {walk_to_start_frame}, ended at frame: {walk_to_end_frame}")
        logging.info(f"Turn time: {turn_time} seconds - started at frame: {turn_start_frame}, ended at frame: {turn_end_frame}")
        logging.info(f"Walk back time: {walk_back_time} seconds - started at frame: {walk_back_start_frame}, ended at frame: {walk_back_end_frame}")
        logging.info(f"Sit time: {sit_time} seconds - started at frame: {sit_start_frame}, ended at frame: {sit_end_frame}")
        segment_times = {
            "Stand": stand_time,
            "Walk To": walk_to_time,
            "Turn": turn_time,
            "Walk Back": walk_back_time,
            "Sit": sit_time,
        }
    else:
        segment_times = {
            "Stand": -1,
            "Walk To": -1,
            "Turn": -1,
            "Walk Back": -1,
            "Sit": -1,
        }
        logging.info("Problems detected, unable to calculate segment times")
    if start_frame_id is not None and end_frame_id is not None:
        elapsed_time = (end_frame_id - start_frame_id) / fps
        average_speed = distance_walked / elapsed_time if elapsed_time > 0 else 0
        logging.info(f"Average speed: {average_speed} cm/second")
    else:
        elapsed_time = 0
        average_speed = 0
        logging.info("Problems detected, unable to calculate")

    return (
        "Timed Up And Go",
        distance_walked,
        elapsed_time,
        segment_times,
        average_speed,
        average_stride_length,
        keypoint_mean_magnitudes,
        keypoint_std_devs,
        keypoint_circular_mean,
        keypoint_circular_std
    )

