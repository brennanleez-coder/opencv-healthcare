from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from queue import Queue, Empty
import threading
from app.utils.logger import Logger
import app.algorithms.sit_stand_overall as sso

router = APIRouter()
logger_instance = Logger()
logger = logger_instance.get_logger()

# Global queue to hold results
result_queue = Queue()

def thread_function(queue, logger):
    result = sso.sit_stand_overall('/Users/brennanlee/Desktop/opencv-healthcare/test/CST_self2.mp4', False)
    queue.put(result)

def process_video(queue: Queue, logger):
    thread = threading.Thread(target=thread_function, args=(queue, logger))
    thread.start()
    thread.join()  # Wait for the thread to finish
    logger.info("Video processing completed")

@router.get("/video_processing")
async def video_processing(background_tasks: BackgroundTasks):
    # Start video processing in background
    background_tasks.add_task(process_video, result_queue, logger)
    logger.info("Video processing started in background")
    
    # Respond immediately while processing continues in the background
    return JSONResponse(
        status_code=200,
        content={"message": "Video processing started in background"}
    )

@router.get("/video_result")
async def video_result():
    try:
        result = result_queue.get_nowait()  # Retrieve the result from the queue without blocking
    except Empty:
        return JSONResponse(
            status_code=202,
            content={"message": "Video processing is still ongoing"}
        )
    
    counter, elapsed_time, rep_durations, violations, max_angles = result
    logger.info(f"Elapsed time: {elapsed_time} seconds")
    logger.info(f"Counter: {counter}")
    logger.info(f"Rep durations: {rep_durations}")
    logger.info(f"Violations: {violations}")
    logger.info(f"Max angles: {max_angles}")
    
    return JSONResponse(
        status_code=200,
        content={
            "counter": counter,
            "elapsed_time": elapsed_time,
            "rep_durations": rep_durations,
            "violations": violations,
            "max_angles": max_angles
        }
    )
