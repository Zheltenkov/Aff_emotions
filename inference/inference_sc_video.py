import sys
import cv2
from scripts import moving_window
from config import DataParameters as dp
from .video_detector import VideoDetector
from .graph_constructor import write_polar_animation, write_line_animation


def inference_video(video: str) -> None:
    """ """
    vd = VideoDetector(input_video_file=video)

    cap = vd.get_video_stream()
    valence_model = vd.load_valence_model(dp.val_model).eval()
    arousal_model = vd.load_arousal_model(dp.arr_model).eval()
    emotion_model = vd.load_emotional_model(dp.emo_model).eval()
    writer, fps, image_width, image_height, frame_count = vd.get_writer()

    valence_values, arousal_values = [], []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame.flags.writeable = False

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, y, w, h = vd.get_detection(img, image_width, image_height)

        face = frame[int(y): int(h), int(x): int(w)]

        emotion = vd.get_emotion_prediction(vd.transform(face, img_size=224), emotion_model)
        valence = vd.get_valence_prediction(vd.transform(face, img_size=64), valence_model)
        arousal = vd.get_arousal_prediction(vd.transform(face, img_size=64), arousal_model)

        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 1)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f'Arousal:{arousal}', (x, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f'Valence:{valence}', (x, h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        writer.write(frame)

        valence_values.append(valence)
        arousal_values.append(arousal)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            sys.exit()

    cap.release()

    valence_val = moving_window(valence_values, dp.window)
    arousal_val = moving_window(arousal_values, dp.window)

    write_polar_animation(valence_val, arousal_val, fps, frame_count)
    write_line_animation(valence_val, arousal_val, fps, frame_count)


