import pandas as pd
import numpy as np
import joblib
from app.utils.logger import Logger
logger_instance = Logger()
logger = logger_instance.get_logger()
ST_MODEL_PATH = "app/models/sit_stand_rf_model.pkl"
GS_MODEL_PATH = "app/models/gait_speed_rf_model.pkl"
TUG_MODEL_PATH = "app/models/tug_rf_model.pkl"
def load_model(type):
    if type == "5 Sit Stand":
        model_path = ST_MODEL_PATH
    elif type == "Gait Speed Walk Test":
        model_path = GS_MODEL_PATH
    elif type == "Timed Up and Go":
        model_path = TUG_MODEL_PATH
    else:
        logger.error(f"Model not found for type: {type}")
        return None
        
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

def predict_with_model(type, features):
    model = load_model(type)
    logger.info(f"model: {model}")
    predictions = model.predict(features)
    return predictions

def sit_stand_helper(type, pass_fail, counter, elapsed_time, rep_durations, violations, max_angles, keypoint_mean_magnitudes, keypoint_std_devs, keypoint_circular_mean, keypoint_circular_std):
    df = pd.DataFrame(
    columns=[
        "test",
        "status",
        "reps",
        "time",
        "rep_durations",
        "violations",
        "max_angles",
        "keypoint_mean_magnitudes",
        "keypoint_std_devs",
        "keypoint_circular_mean",
        "keypoint_circular_std",
    ]
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [[
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
                ]],
                columns=[
                    "test",
                    "status",
                    "reps",
                    "time",
                    "rep_durations",
                    "violations",
                    "max_angles",
                    "keypoint_mean_magnitudes",
                    "keypoint_std_devs",
                    "keypoint_circular_mean",
                    "keypoint_circular_std",
                    
                ],
            ),
        ]
    )

    # keypoint_mean_magnitudes, keypoint_std_devs, keypoint_circular_mean, keypoint_circular_std are dictionaries of keypoints as key and std as values, flatten them into a single column for each keypoint
    for keypoint in keypoint_mean_magnitudes:
        df[keypoint + "_mean_magnitude"] = df["keypoint_mean_magnitudes"].apply(
            lambda x: x[keypoint]
        )
        df[keypoint + "_std_dev"] = df["keypoint_std_devs"].apply(lambda x: x[keypoint])
        df[keypoint + "_circular_mean"] = df["keypoint_circular_mean"].apply(
            lambda x: x[keypoint]
        )
        df[keypoint + "_circular_std"] = df["keypoint_circular_std"].apply(
            lambda x: x[keypoint]
        )
        
    df.drop(
        columns=[
            "keypoint_mean_magnitudes",
            "keypoint_std_devs",
            "keypoint_circular_mean",
            "keypoint_circular_std",
        ],
        inplace=True,
    )
    # df['rep_durations'] = df['rep_durations'].apply(lambda x: eval(x))
    # df['max_angles'] = df['max_angles'].apply(lambda x: eval(x))

    df = df[df.columns.drop(list(df.filter(regex='circular_mean')))]
    df = df[df.columns.drop(list(df.filter(regex='circular_std')))]
    df = df[df.columns.drop(list(df.filter(regex='mean_magnitude')))]
    df = df[df.columns.drop(list(df.filter(regex='RIGHT')))]
    df = df[df.columns.drop(list(df.filter(regex='TOE')))]
    df=df.drop(columns=['test'])
    df=df.drop(columns=['reps'])
    
    # Feature Engineering
    # highest max angle in the rep
    df['max_max_angle'] = df['max_angles'].apply(lambda x: max(x))
    # lowest max angle in the rep
    df['min_max_angle'] = df['max_angles'].apply(lambda x: min(x))
    # average duration
    df['avg_duration'] = df['rep_durations'].apply(lambda x: np.mean(x))
    # highest duration in the rep
    df['max_duration'] = df['rep_durations'].apply(lambda x: max(x))
    df.drop(columns=['max_angles'], inplace=True)
    df.drop(columns=['rep_durations'], inplace=True)
    df.drop(columns=['violations'], inplace=True)
    df.drop(columns=['status'], inplace=True)
    print(df)
    
    df['shoulder_hip_ratio'] = df['LEFT_SHOULDER_std_dev'] / df['LEFT_HIP_std_dev']
    df['knee_ankle_ratio'] = df['LEFT_KNEE_std_dev'] / df['LEFT_ANKLE_std_dev']
    df['hip_knee_ratio'] = df['LEFT_HIP_std_dev'] / df['LEFT_KNEE_std_dev']
    df['angle_range'] = df['max_max_angle'] - df['min_max_angle']
    df['movement_stability'] = (df['LEFT_SHOULDER_std_dev'] + 
                                      df['LEFT_HIP_std_dev'] + 
                                      df['LEFT_KNEE_std_dev'] + 
                                      df['LEFT_ANKLE_std_dev']) / 4
    # rename columns to match the model
    df.rename(columns={'time': 'time'}, inplace=True)
    df.rename(columns={'max_max_angle': 'max_max_angle'}, inplace=True)
    df.rename(columns={'min_max_angle': 'min_max_angle'}, inplace=True)
    df.rename(columns={'LEFT_SHOULDER_std_dev': 'RIGHT_SHOULDER_std_dev'}, inplace=True)
    df.rename(columns={'LEFT_HIP_std_dev': 'RIGHT_HIP_std_dev'}, inplace=True)
    df.rename(columns={'LEFT_KNEE_std_dev': 'RIGHT_KNEE_std_dev'}, inplace=True)
    df.rename(columns={'LEFT_ANKLE_std_dev': 'RIGHT_ANKLE_std_dev'}, inplace=True)
    df.rename(columns={'avg_duration': 'avg_duration'}, inplace=True)
    df.rename(columns={'shoulder_hip_ratio': 'shoulder_hip_ratio'}, inplace=True)
    df.rename(columns={'knee_ankle_ratio': 'knee_ankle_ratio'}, inplace=True)
    df.rename(columns={'hip_knee_ratio': 'hip_knee_ratio'}, inplace=True)
    df.rename(columns={'angle_range': 'angle_range'}, inplace=True)
    df.rename(columns={'movement_stability': 'movement_stability'}, inplace=True)
    
    # rearrange columns to match the model
    df = df[['time', 'max_max_angle', 'min_max_angle', 'RIGHT_SHOULDER_std_dev',
        'RIGHT_HIP_std_dev', 'RIGHT_KNEE_std_dev', 'RIGHT_ANKLE_std_dev',
        'avg_duration', 'shoulder_hip_ratio', 'knee_ankle_ratio',
        'hip_knee_ratio', 'angle_range', 'movement_stability']]
    # 5ST: ['time', 'max_max_angle', 'min_max_angle', 'RIGHT_SHOULDER_std_dev',
    #    'RIGHT_HIP_std_dev', 'RIGHT_KNEE_std_dev', 'RIGHT_ANKLE_std_dev',
    #    'avg_duration', 'shoulder_hip_ratio', 'knee_ankle_ratio',
    #    'hip_knee_ratio', 'angle_range', 'movement_stability']
    logger.info(f"Dataframe: {df}")
    
    logger.info(f"ml input: {df.values.reshape(1, -1)}")
    frailty_score = predict_with_model(type, df.values.reshape(1, -1))
    logger.info(f"frailty_score: {frailty_score[0]}")
    return frailty_score
    