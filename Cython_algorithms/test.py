import sit_stand_overall  # This is your compiled Cython module

# Example test data, adjust according to the function's expected input
point_a = [1.0, 2.0]
point_b = [2.0, 3.0]
point_c = [3.0, 4.0]
import numpy as np

point_a = np.array(point_a)
point_b = np.array(point_b)
point_c = np.array(point_c)

angle = sit_stand_overall.calculate_angle(point_a, point_b, point_c)


# Call the function from your Cython module
angle = sit_stand_overall.calculate_angle(point_a, point_b, point_c)

print("Calculated angle:", angle)


# Test to_timestamp function
timestamp = 1609459200.0  # Example timestamp for January 1, 2021
formatted_time = sit_stand_overall.to_timestamp(timestamp)
print("Formatted time:", formatted_time)

# Test calculate_distance function
p1 = [1.0, 2.0]
p2 = [4.0, 6.0]

p1 = np.array(p1)
p2 = np.array(p2)
distance = sit_stand_overall.calculate_distance(p1, p2)
print("Distance:", distance)


# Test should_start_timer function
keypoints = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
hip_displacement = 4.0
knee_displacement = 2.5
ankle_displacement = 1.7

# Call the function
timer_should_start = sit_stand_overall.should_start_timer(
    keypoints, hip_displacement, knee_displacement, ankle_displacement
)
print("Timer should start:", timer_should_start)



# Mockup for the landmarks object based on what you would expect from MediaPipe
class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

landmarks = {
    23: MockLandmark(0.5, 0.5),
    25: MockLandmark(0.6, 0.6),
    27: MockLandmark(0.7, 0.7),
    11: MockLandmark(0.4, 0.4)
}


hip, knee, ankle, shoulder = sit_stand_overall.get_landmark_coordinates(landmarks, 640, 480)
print("Hip:", hip)
print("Knee:", knee)
print("Ankle:", ankle)
print("Shoulder:", shoulder)


# Test failure
print(f"Test failure: {sit_stand_overall.determine_failure(11.990)}")

# Test summarise_results function
counter = 5
elapsed_time = 11.0
rep_durations = [10.2, 12.5, 11.4, 13.0, 12.8]
joint_displacement_map = None  # Add actual data if needed
joint_velocity_map = None  # Add actual data if needed

# Call the function
sit_stand_overall.summarise_results(counter, elapsed_time, rep_durations, joint_displacement_map, joint_velocity_map)


# Test display_knee_and_hip_angle
import cv2
import numpy as np

# Mock data for testing
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
knee_angle = 45.0
knee = np.array([100.0, 200.0], dtype=np.float64)
hip_angle = 30.0
hip = np.array([150.0, 250.0], dtype=np.float64)

# Call the function
sit_stand_overall.display_knee_and_hip_angle(image, knee_angle, knee, hip_angle, hip)

# Display the image
cv2.imshow("Image with Angles", image)
cv2.waitKey(0) # Wait for any key press
cv2.destroyAllWindows()
print("Displaying image with angles...")

# test draw_joint_displacement
# Mock data for testing
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
prev_point = np.array([100.0, 200.0], dtype=np.float64)
curr_point = np.array([150.0, 250.0], dtype=np.float64)

# Call the function
sit_stand_overall.draw_joint_displacement(prev_point, curr_point, image)

# Display the image
cv2.imshow("Image with Joint Displacement", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# test display_point

# Mock data for testing
point = np.array([100.0, 200.0], dtype=np.float64)

# Call the function and print the result
result = sit_stand_overall.display_x_and_y_from_point(point)
print(result)  # Expected output: "[x: 100.00, y: 200.00]"



# Test calculate and draw joint displacement
# Mock data for testing
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
prev_frame = np.zeros((480, 640, 3), dtype=np.uint8)
prev_points = [np.array([100.0, 200.0], dtype=np.float64), np.array([150.0, 250.0], dtype=np.float64), np.array([200.0, 300.0], dtype=np.float64)]
curr_points = [np.array([110.0, 210.0], dtype=np.float64), np.array([160.0, 260.0], dtype=np.float64), np.array([210.0, 310.0], dtype=np.float64)]
joint_displacement_history = {"HIP": [], "KNEE": [], "ANKLE": []}
real_time = 1.0

# Call the function
joint_displacement_history = sit_stand_overall.calculate_and_draw_joint_displacement(prev_frame, prev_points, curr_points, image, joint_displacement_history, real_time)
print(joint_displacement_history)
# Display the image
cv2.imshow("Image with Joint Displacement", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# test display_information
# Mock data for testing
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
counter = 5
stage = "DOWN"
max_angle = 45.5

# Call the function
sit_stand_overall.display_information(image, counter, stage, max_angle)

# Display the image
cv2.imshow("Image with Information", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# test display timer
# Mock data for testing
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
elapsed_time = 123.45

# Call the function
sit_stand_overall.display_timer(image, elapsed_time, 10, 60)

# Display the image
cv2.imshow("Image with Timer", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # test draw landmarks and connections
# import mediapipe as mp

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# # Load a sample image, use absolute path
# test_image_path = '/Users/brennanlee/Desktop/opencv-healthcare/test/test-image.jpeg'
# image = cv2.imread(test_image_path)
# if image is None:
#     print(f"Error: Could not load image from path: {test_image_path}")
# else:
#     # Convert the image to RGB
#     # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # seems like dont need to convert the image to RGB
#     results = pose.process(image)

#     # Draw landmarks and connections on the image
#     sit_stand_overall.draw_landmarks_and_connections(image, results)

#     # Display the image
#     cv2.imshow("Image with Landmarks", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    
# # Test data for testing
# frame_counter = 150
# fps = 30.0

# # Call the function
# real_time = sit_stand_overall.get_real_time_from_frames(frame_counter, fps)
# print(f"Real time: {real_time} seconds")  # Expected output: 5.0 seconds



# test sit_stand_overall

# counter, elapsed_time, rep_durations, violations, max_angles = sit_stand_overall.sit_stand_overall('/Users/brennanlee/Desktop/opencv-healthcare/test/CST_self2.mp4', True)
# print("Counter:", counter)
# print("Elapsed time:", elapsed_time)
# print("Rep durations:", rep_durations)
# print("Violations:", violations)
# print("Max angles:", max_angles)

import numpy
print(numpy.get_include())