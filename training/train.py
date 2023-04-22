import os
import time
import torch
from data_loader import LoadData
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from .train_one_epoch import DrillInstructor
from config.parameters import DataParameters as pr
from architecture import AffModelRegressor, ResNet50
from scripts import get_dataset, count_class_weights, create_logger
from scripts import get_optimizer, get_loss_func, create_debug_directories


def start_training(model_type: str = None) -> None:
    # CHECK MODEL TYPE
    if model_type is None:
        raise ValueError(f"INVALID MODEL_TYPE '{model_type}', MUST BE 'emotions', 'valence' or 'arousal'.")

    # CREATE DIRECTORIES FOR DEBUGGING
    create_debug_directories()
    print(f'---CREATE WORKING DIRECTORIES---')

    # CREATE LOGGER
    log_train = create_logger(pr.logging_path, 'train')
    log_valid = create_logger(pr.logging_path, 'valid')
    print(f'--------CREATE LOGGERS--------')

    # CUDA LAUNCH
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'------WORKING DEVICE IS {str(device).upper()}------')

    # CREATE DATA DICTIONARIES
    train_data = get_dataset(pr.train_path, model_type)
    valid_data = get_dataset(pr.valid_path, model_type)

    # CALCULATE SUM OF CLASSES
    if pr.use_class_weight:
        class_weights = count_class_weights(train_data['labels'])
    else:
        class_weights = None

    # INITIALIZED DATA LOADERS AND CLASS WEIGHTS
    load_data = LoadData(img_size=pr.img_size, batch_size=pr.batch_size, class_weights=class_weights,
                         train_data=train_data, valid_data=valid_data)

    dataloaders = load_data.get_dataloader()
    train_loader, valid_loader = dataloaders['train'], dataloaders['valid']

    # LOAD MODEL
    model = None
    if model_type == 'emotions':
        if pr.emo_model is None:
            model = ResNet50(num_classes=pr.num_emotions)
            print(f'--PREPARED NEW EMOTION MODEL--')
        else:
            model = ResNet50(num_classes=pr.num_emotions)
            model.load_state_dict(torch.load(pr.emo_model))
            print(f'--PREPARED PRETRAINED EMOTION MODEL--')
    elif model_type == 'arousal':
        if pr.emo_model is None:
            model = AffModelRegressor()
            print(f'--PREPARED NEW AROUSAL MODEL--')
        else:
            model = AffModelRegressor()
            model.load_state_dict(torch.load(pr.arr_model))
            print(f'--PREPARED PRETRAINED AROUSAL MODEL--')
    elif model_type == 'valence':
        if pr.emo_model is None:
            model = AffModelRegressor()
            print(f'--PREPARED NEW VALENCE MODEL--')
        else:
            model = AffModelRegressor()
            model.load_state_dict(torch.load(pr.val_model))
            print(f'--PREPARED PRETRAINED VALENCE MODEL--')
    else:
        print(f'----------INVALID TYPE OF LABEL----------')

    if model is not None:
        for param in model.parameters():
            param.require_grad = True
    print(f'------PREPARED MODEL FOR TRAINING------')

    # SEND MODEL ON GPU
    model.to(device)
    print(f'--------SEND MODELS ON {str(device).upper()}--------')

    # PREPARE LOSS FUNCTION
    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    loss_criterion_train = get_loss_func(model_type, class_weights)
    loss_criterion_valid = get_loss_func(model_type)
    print('------PREPARED LOSS FUNCTION------')

    # PREPARE OPTIMIZER
    optimizer = get_optimizer(model=model, label=model_type, learning_rate=pr.learning_rate,
                              weight_decay=pr.weight_decay)
    print('--------PREPARED OPTIMIZER--------')

    # PREPARE SCHEDULER
    scheduler = StepLR(optimizer=optimizer, step_size=pr.step_size, gamma=pr.gamma)
    print('--------PREPARED SCHEDULER--------')

    # SET SUMMARY WRITE
    train_writer = SummaryWriter(os.path.join(pr.summary_train_path, model_type))
    valid_writer = SummaryWriter(os.path.join(pr.summary_valid_path, model_type))
    print('--------SET SUMMARY WRITER--------')

    # SET TRAIN AND VALID LOOP
    train_loop = DrillInstructor(device=device, model=model, train_writer=train_writer, train_loader=train_loader,
                                 loss_criterion=loss_criterion_train, optimizer=optimizer)

    valid_loop = DrillInstructor(device=device, model=model, valid_writer=valid_writer, valid_loader=valid_loader,
                                 loss_criterion=loss_criterion_valid, optimizer=optimizer)
    print('-----SET TRAIN AND VALID LOOP-----')

    # SET START TIME AND WRITERS
    since = time.time()
    print(f'----START TRAINING FOR {str(model_type).upper()}----')

    # LAUNCH TRAINING LOOP
    for epoch_n in range(pr.epochs):
        if model_type == 'emotions':
            loss_ls_tr, acc_ls_tr, f1_score_tr = train_loop.train_classifier(epoch_n)
            log_train.info(f'Train Loss: {loss_ls_tr} | Train Acc: {acc_ls_tr} | Train F1-Score: {f1_score_tr}')
            loss_ls_val, acc_ls_val, f1_score_val = valid_loop.valid_classifier(epoch_n)
            log_valid.info(f'Train Loss: {loss_ls_val} | Train Acc: {acc_ls_val} | Train F1-Score: {f1_score_val}')
        elif model_type in ['arousal', 'valence']:
            train_loss, train_r_squared = train_loop.train_regressor(epoch_n)
            log_train.info(f'Train Loss: {train_loss} | Train R-square: {train_r_squared}')
            valid_loss, valid_r_squared = valid_loop.valid_regressor(epoch_n)
            log_valid.info(f'Train Loss: {valid_loss} | Train R-square: {valid_r_squared}')
        scheduler.step()

        if epoch_n % 1 == 0:
            torch.save(model.state_dict(), os.path.join(pr.weights, '{0}_{1}.pth'.format(model_type, epoch_n + 300)))
            print('-----------CHECK POINT------------')

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    time_elapsed = time.time() - since
    print('Training time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    del train_loader, valid_loader, dataloaders, model, train_loop, valid_loop
