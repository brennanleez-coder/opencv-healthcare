import cv2
import numpy as np
from pose_estimation import PoseEstimation
from utils import calculate_angle


def main():
    pe = PoseEstimation()
    counter = 0
    stage = None

    while True:
        ret, frame = pe.get_frame()
        if not ret:
            break

        image, results = pe.process_frame(frame)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [
                landmarks[pe.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[pe.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            elbow = [
                landmarks[pe.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[pe.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            wrist = [
                landmarks[pe.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[pe.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Display angle
            cv2.putText(
                image,
                str(angle),
                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Exercise logic
            if angle > 160:
                stage = "down"
            elif angle < 30 and stage == "down":
                stage = "up"
                counter += 1
                print(counter)

            # Render curl counter and stage information
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
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
                stage if stage else "",
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

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    pe.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
