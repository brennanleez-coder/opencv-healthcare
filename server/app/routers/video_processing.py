from fastapi import APIRouter
from fastapi.responses import JSONResponse
from queue import Queue
import threading
from app.utils.logger import Logger
from app.algorithms.sit_stand_overall import process_video

router = APIRouter()
logger_instance = Logger()
logger = logger_instance.get_logger()

def thread_function(queue, logger):
    result = process_video(logger)
    queue.put(result)

@router.get("/video_processing")
async def video_processing():
    queue = Queue()
    thread = threading.Thread(target=thread_function, args=(queue, logger))
    thread.start()
    logger.info("Video processing started")
    thread.join()  # Optionally wait for the thread to finish if needed
    logger.info("Video processing completed")
    result = queue.get()  # Retrieve the result from the queue
    elapsed_time, counter, rep_durations, joint_displacement_history = result
    logger.info(f"Elapsed time: {elapsed_time} seconds")
    logger.info(f"Counter: {counter}")
    logger.info(f"Rep durations: {rep_durations}")
    logger.info(f"Joint displacement history: {joint_displacement_history}")
    
    return JSONResponse(
        status_code=200,
        content={"elapsed_time": elapsed_time, "counter": counter, "rep_durations": rep_durations, "joint_displacement_history": joint_displacement_history}
    )
