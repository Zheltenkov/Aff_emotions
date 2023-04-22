from dataclasses import dataclass, field


@dataclass(kw_only=True)
class MainConfig:
    gpu: int = field(default=0, metadata={'doc': 'Gpu id'})
    working_status: str = field(default='inference', metadata={'doc': 'Application working status'})
    emo_model: str = field(default='./debug/weights/emotions.pth', metadata={'doc': 'Emotional pretrained model'})
    arr_model: str = field(default='./debug/weights/arousal.pth', metadata={'doc': 'Arousal pretrained model'})
    val_model: str = field(default='./debug/weights/valence.pth', metadata={'doc': 'Valence pretrained model'})
    weights: str = field(default='./debug/weights', metadata={'doc': 'Path to model weights'})
    window: int = field(default=3, metadata={'doc': 'Size of window'})
    emotion_dict: dict[int, str] = field(default_factory=lambda: {0: "Neutral",
                                                                  1: "Happiness",
                                                                  2: "Sad",
                                                                  3: "Surprise",
                                                                  4: "Fear",
                                                                  5: "Disgust",
                                                                  6: "Anger",
                                                                  7: "Contempt"
                                                                  }, metadata={'doc': 'Emotional classes'})

    def __post_init__(self):
        if not self.emotion_dict:
            self.emotion_dict = {0: "Neutral",
                                 1: "Happiness",
                                 2: "Sad",
                                 3: "Surprise",
                                 4: "Fear",
                                 5: "Disgust",
                                 6: "Anger",
                                 7: "Contempt"}


@dataclass(kw_only=True)
class PathsConfig:
    train_path: str = field(default='./data/train/train.csv', metadata={'doc': 'Path to train data'})
    valid_path: str = field(default='./data/validation/valid.csv', metadata={'doc': 'Path to valid data'})
    logging_path: str = field(default='./debug/logs', metadata={'doc': 'Path to logging train'})
    summary_train_path: str = field(default='./debug/summary/train', metadata={'doc': 'Path to train summary writer'})
    summary_valid_path: str = field(default='./debug/summary/valid', metadata={'doc': 'Path to valid summary writer'})
    input_video: str = field(default='./data/input_video', metadata={'doc': 'Path to input video'})
    output_video: str = field(default='./data/output_video', metadata={'doc': 'Path to result video'})
    vid_1: str = field(default='./data/output_video/emotions.mp4', metadata={'doc': 'Path to result emotions video'})
    vid_2: str = field(default='./data/output_video/line.mp4', metadata={'doc': 'Path to line graph video'})
    vid_3: str = field(default='./data/output_video/polar.mp4', metadata={'doc': 'Path to polar graph video'})
    vid_out: str = field(default='./data/output_video/result.mp4', metadata={'doc': 'Path to result video'})


@dataclass(kw_only=True)
class TrainConfig:
    label_type: str = field(default='emotions', metadata={'doc': 'Type of train model: emotions, arousal, valence'})
    epochs: int = field(default=101, metadata={'doc': 'Number of training epochs'})
    batch_size: int = field(default=32, metadata={'doc': 'Batch size for training'})
    num_emotions: int = field(default=8, metadata={'doc': 'Number of emotion classes'})
    img_size: int = field(default=224, metadata={'doc': 'Size of input image for architecture model'})
    use_class_weight: bool = field(default=False, metadata={'doc': 'Flag for using class_weight for unbalanced class'})


@dataclass(kw_only=True)
class OptimizerConfig:
    momentum: float = field(default=0.9, metadata={'doc': 'Momentum training parameter'})
    weight_decay: float = field(default=5e-4, metadata={'doc': 'Weight decay training parameter'})
    learning_rate: float = field(default=0.001, metadata={'doc': 'Learning rate training parameter'})
    step_size: int = field(default=10, metadata={'doc': 'Step size for scheduler'})
    gamma: float = field(default=0.05, metadata={'doc': 'Percentage of decrease speed of learning'})


@dataclass
class DataParameters(MainConfig, PathsConfig, TrainConfig, OptimizerConfig):
    pass
