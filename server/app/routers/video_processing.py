from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from fastapi import Form
from fastapi.responses import JSONResponse
from queue import Queue, Empty
import threading
import uuid
import os
from app.utils.logger import Logger
from . import sit_stand_overall as sso
from . import gait_speed_walk_overall as gswo
import app.routers.tug_overall as tugo
import app.helpers.data_engineering as de
from typing import Optional


router = APIRouter()
logger_instance = Logger()
logger = logger_instance.get_logger()


result_queues = {}  # {request_id: {"queue": Queue, "algo": "Gait Speed Walk"}}


def thread_function(video_path, algo, queue, logger, **kwargs):
    try:
        if algo == "Gait Speed Walk Test":
            result = gswo.process_test(video_path, **kwargs)
        elif algo == "5 Sit Stand":
            result = sso.process_sit_stand(video_path, **kwargs)
        elif algo == "Timed Up and Go":
            result = tugo.process_tug(video_path, **kwargs)
        queue.put(result)
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        queue.put(None)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)  # Ensure the temporary file is deleted


def process_video(
    file_content: bytes,
    filename: str,
    request_id: str,
    queue: Queue,
    algo: str,
    logger,
    **kwargs,
):
    video_path = f"/tmp/{request_id}_{filename}"
    with open(video_path, "wb") as f:
        f.write(file_content)
    thread = threading.Thread(
        target=thread_function, args=(video_path, algo, queue, logger), kwargs=kwargs
    )
    thread.start()


@router.post("/video_processing")
async def video_processing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    algo: str = Form(...),
    standingHeight: Optional[float] = Form(None),
    sittingHeight: Optional[float] = Form(None),
    distance: Optional[float] = Form(None),
):
    request_id = str(uuid.uuid4())
    file_content = await file.read()

    result_queue = Queue()
    result_queues[request_id] = {"queue": result_queue, "algo": algo}

    print(f"Input Sanity Check: {algo}, {standingHeight}, {sittingHeight}, {distance}")
    kwargs = {}
    if algo == "Timed Up and Go":
        kwargs = {
            "sit_down_height_in_cm": sittingHeight,
            "distance_required_in_cm": distance,
            "debug": False,
        }  # STUB
    elif algo == "5 Sit Stand":
        kwargs = {"display": False}
    elif algo == "Gait Speed Walk Test":
        kwargs = {
            "person_height_in_cm": standingHeight,
            "distance_required_in_cm": distance,
            "debug": False,
        }  # STUB
    else:
        return JSONResponse(
            status_code=400, content={"message": "Invalid algorithm specified"}
        )

    background_tasks.add_task(
        process_video,
        file_content,
        file.filename,
        request_id,
        result_queue,
        algo,
        logger,
        **kwargs,
    )
    logger.info(
        f"Video processing started in background for request ID {request_id} using algorithm {algo}"
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "Video processing started in background",
            "request_id": request_id,
        },
    )


@router.get("/video_result/{request_id}")
async def video_result(request_id: str):
    if request_id not in result_queues:
        return JSONResponse(status_code=404, content={"message": "Invalid request ID"})

    result_queue_with_algo = result_queues[request_id]
    result_queue = result_queue_with_algo["queue"]
    algo = result_queue_with_algo["algo"]
    try:
        result = (
            result_queue.get_nowait()
        )  # Retrieve the result from the queue without blocking
    except Empty:
        return JSONResponse(
            status_code=202, content={"message": "Video processing is still ongoing"}
        )

    if result is None:
        return JSONResponse(
            status_code=500,
            content={"message": "An error occurred during video processing"},
        )

    # Clean up the result queue
    del result_queues[request_id]

    if algo == "5 Sit Stand":
        response = process_sit_stand_response(result)
    elif algo == "Gait Speed Walk Test":
        response = process_gswt_response(result)

    elif algo == "Timed Up and Go":
        
        response = process_tug_response(result)
    else:
        return JSONResponse(
            status_code=400, content={"message": "Invalid algorithm specified"}
        )

    return JSONResponse(status_code=200, content=response)


def process_sit_stand_response(result):
    (
        type,
        pass_fail,
        counter,
        elapsed_time,
        rep_durations,
        violations,
        max_angles,
        keypoint_mean_magnitudes,
        keypoint_std_devs,
        keypoint_circular_mean,
        keypoint_circular_std,
    ) = result
    logger.info(f"Type: {type}")
    logger.info(f"Counter: {counter}")
    logger.info(f"Rep durations: {rep_durations}")
    logger.info(f"Violations: {violations}")
    logger.info(f"Max angles: {max_angles}")
    logger.info(f"Elapsed time: {elapsed_time}")
    logger.info(f"Pass/Fail: {pass_fail}")
    logger.info(f"Keypoint mean magnitudes: {keypoint_mean_magnitudes}")
    logger.info(f"Keypoint std devs: {keypoint_std_devs}")
    logger.info(f"Keypoint circular mean: {keypoint_circular_mean}")
    logger.info(f"Keypoint circular std: {keypoint_circular_std}")

    frailty_score = de.sit_stand_helper(
        type,
        pass_fail,
        counter,
        elapsed_time,
        rep_durations,
        violations,
        max_angles,
        keypoint_mean_magnitudes,
        keypoint_std_devs,
        keypoint_circular_mean,
        keypoint_circular_std,
    )
    logger.info(f"Frailty Score: {frailty_score[0]}")

    response = {
        "type": type,
        "pass_fail": pass_fail,  # "pass" or "fail
        "counter": counter,
        "elapsed_time": elapsed_time,
        "rep_durations": rep_durations,
        "violations": violations,
        "max_angles": max_angles,
        "keypoint_mean_magnitudes": keypoint_mean_magnitudes,
        "keypoint_std_devs": keypoint_std_devs,
        "keypoint_circular_mean": keypoint_circular_mean,
        "keypoint_circular_std": keypoint_circular_std,
        "frailty_score": frailty_score[0],
    }
    return response


def process_gswt_response(result):
    (
        type,
        distance_walked,
        elapsed_time,
        average_speed,
        average_stride_length,
        keypoint_mean_magnitudes,
        keypoint_std_devs,
        keypoint_circular_mean,
        keypoint_circular_std,
    ) = result
    logger.info(f"Type: {type}")
    logger.info(f"Distance Walked: {distance_walked}")
    logger.info(f"Elapsed time: {elapsed_time}")
    logger.info(f"Average Speed: {average_speed}")
    logger.info(f"Average Stride Length: {average_stride_length}")
    logger.info(f"Keypoint mean magnitudes: {keypoint_mean_magnitudes}")
    logger.info(f"Keypoint std devs: {keypoint_std_devs}")
    logger.info(f"Keypoint circular mean: {keypoint_circular_mean}")
    logger.info(f"Keypoint circular std: {keypoint_circular_std}")

    frailty_score = de.gait_speed_walk_test_helper(
        type,
        distance_walked,
        elapsed_time,
        average_speed,
        average_stride_length,
        keypoint_mean_magnitudes,
        keypoint_std_devs,
        keypoint_circular_mean,
        keypoint_circular_std,
    )
    logger.info(f"Frailty Score: {frailty_score[0]}")

    response = {
        "type": type,
        "distance_walked": distance_walked,
        "elapsed_time": elapsed_time,
        "average_speed": average_speed,
        "average_stride_length": average_stride_length,
        "keypoint_mean_magnitudes": keypoint_mean_magnitudes,
        "keypoint_std_devs": keypoint_std_devs,
        "keypoint_circular_mean": keypoint_circular_mean,
        "keypoint_circular_std": keypoint_circular_std,
        "frailty_score": frailty_score[0],
    }
    return response


def process_tug_response(result):
    (
        type,
        distance_walked,
        elapsed_time,
        segment_times,
        average_speed,
        average_stride_length,
        keypoint_mean_magnitudes,
        keypoint_std_devs,
        keypoint_circular_mean,
        keypoint_circular_std,
    ) = result
    logger.info(f"Type: {type}")
    logger.info(f"Distance Walked: {distance_walked}")
    logger.info(f"Elapsed time: {elapsed_time}")
    logger.info(f"Segment times: {segment_times}")
    logger.info(f"Average Speed: {average_speed}")
    logger.info(f"Average Stride Length: {average_stride_length}")
    logger.info(f"Keypoint mean magnitudes: {keypoint_mean_magnitudes}")
    logger.info(f"Keypoint std devs: {keypoint_std_devs}")
    logger.info(f"Keypoint circular mean: {keypoint_circular_mean}")
    logger.info(f"Keypoint circular std: {keypoint_circular_std}")
    
    frailty_score = de.tug_helper(
        type,
        distance_walked,
        elapsed_time,
        segment_times,
        average_speed,
        average_stride_length,
        keypoint_mean_magnitudes,
        keypoint_std_devs,
        keypoint_circular_mean,
        keypoint_circular_std,
    )
    

    response = {
        "type": type,
        "distance_walked": distance_walked,
        "elapsed_time": elapsed_time,
        "segment_times": segment_times,
        "average_speed": average_speed,
        "average_stride_length": average_stride_length,
        "keypoint_mean_magnitudes": keypoint_mean_magnitudes,
        "keypoint_std_devs": keypoint_std_devs,
        "keypoint_circular_mean": keypoint_circular_mean,
        "keypoint_circular_std": keypoint_circular_std,
        "frailty_score": frailty_score[0],
    }
    return response
