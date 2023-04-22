import os
from scripts import compare_video
from training import start_training
from inference import inference_video
from config import DataParameters as dp


if __name__ == "__main__":
    if dp.working_status == 'inference':
        inference_video(os.path.join(dp.input_video, os.listdir(dp.input_video)[0]))
        compare_video(dp.vid_1, dp.vid_2, dp.vid_3, dp.vid_out)
    elif dp.working_status == 'train':
        start_training(model_type='emotions')
        start_training(model_type='arousal')
        start_training(model_type='valence')
    else:
        print('Choose working status as inference/train')



