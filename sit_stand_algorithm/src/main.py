import cv2
import numpy as np
from pose_estimation import PoseEstimation
from utils import calculate_angle


def main():
    # Curl counter variables
    counter = 0
    stage = None
    frame_count = 0
    confirm_frames = 5
    stage_counter = 0

    pe = PoseEstimation("../test/CST_self1.mp4")
    
    
    while True:
        ret, frame = pe.get_frame()
        if not ret:
            break
        
        image, results = pe.process_frame(frame)
        
        try:
            landmarks = results.pose_landmarks.landmark
            hip = [
                landmarks[pe.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[pe.mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            ankle = [
                landmarks[pe.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[pe.mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            ]
            knee = [
                landmarks[pe.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[pe.mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ]
            
            angle = calculate_angle(hip, knee, ankle)
            # Visualize angle
            cv2.putText(
                image,
                str(f"knee angle: {angle}"),
                tuple(np.multiply(knee, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            # CST counter logic
            if stage is None:
                # Determine the initial stage based on the first frame's angle
                if angle > 135:
                    stage = "up"
                else:
                    stage = "down"

            if stage == "down" and angle > 135:
                stage_counter += 1
                if stage_counter >= confirm_frames:
                    stage = "up"
                    stage_counter = 0
                    counter += 1  # Increment counter on transitioning to "up"
            elif stage == "up" and angle < 90:
                stage_counter += 1
                if stage_counter >= confirm_frames:
                    stage = "down"
                    stage_counter = 0  # Reset the stage counter after confirming the stage
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

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
            pe.draw_landmarks(image, results)
        except AttributeError:
            pass
        
        cv2.imshow("Sit To Stand Counter", image)
        
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        
    pe.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()