import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


@dataclass(kw_only=True)
class DrillInstructor:

    model: nn.Module = None
    device: torch.device = None
    train_loader: DataLoader = None
    valid_loader: DataLoader = None
    loss_criterion: nn.Module = None
    train_writer: SummaryWriter = None
    valid_writer: SummaryWriter = None
    optimizer: torch.optim.Optimizer = None

    def train_classifier(self, epoch_n: int) -> Tuple[float, float, float]:
        """
        :param epoch_n:
        :return:
        """
        self.model.train()

        train_loss, train_accuracy, train_f1_score = 0.0, 0.0, 0.0
        num_batches = len(self.train_loader)

        for i, data in enumerate(self.train_loader):
            images = data['image'].to(self.device)
            labels = data['label'].to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(images)

            loss = self.loss_criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                prediction_class = torch.argmax(predictions, dim=-1)
                f1_scr = f1_score(prediction_class.cpu(), labels.cpu(), average='weighted')
                correct = prediction_class.eq(labels).sum().item()
                accuracy = correct / len(labels)

            train_loss += loss.item() * images.size(0)
            train_accuracy += accuracy
            train_f1_score += f1_scr

            print('-----LOGGING TRAIN BATCH------')
            print(f'Epoch - {epoch_n}, Batch - {(i+1)}, Total batch - {num_batches}')
            print('Losses/total_loss', train_loss / (i+1))
            print('Accuracy/total_accuracy', train_accuracy / (i+1))
            print('Train_f1_score/total_train_f1_score', train_f1_score / (i+1))
            print(40 * '--')

        # GET AVERAGE
        train_loss /= num_batches
        train_accuracy /= num_batches
        train_f1_score /= num_batches

        self.train_writer.add_scalar('Losses/total_loss', train_loss, epoch_n)
        self.train_writer.add_scalar('Accuracy/total_accuracy', train_accuracy, epoch_n)
        self.train_writer.add_scalar('Train_f1_score/total_train_f1_score', train_f1_score, epoch_n)

        print(40 * '++')
        print(f'Train Metrics on epoch {epoch_n}:')
        print(f'Accuracy: {train_accuracy:.3f}, Avg loss: {train_loss:.8f}')
        print(f'F1-score: {train_f1_score:.3f}')
        print(40 * '++')

        return train_loss, train_accuracy, train_f1_score

    def train_regressor(self, epoch_n: int) -> Tuple[float, float]:
        """
        :param epoch_n:
        :return:
        """
        self.model.train()

        train_loss, train_r_squared = 0.0, 0.0
        num_batches = len(self.train_loader)

        for i, data in enumerate(self.train_loader):
            images = data['image'].to(self.device)
            labels = data['label'].to(self.device)
            self.optimizer.zero_grad()

            predictions = self.model(images)

            loss = self.loss_criterion(predictions, labels.unsqueeze(1).float())
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                r_squared = (1 - loss / torch.var(labels)).item()

            train_loss += loss.item()
            train_r_squared += r_squared

            print('-----LOGGING TRAIN BATCH------')
            print(f'Epoch - {epoch_n}, Batch - {(i+1)}, Total batch - {num_batches}')
            print('MSE/total_loss', train_loss / (i+1))
            print('Train_R-squared/total_train_R-squared', train_r_squared / (i+1))
            print(40 * '--')

        # GET AVERAGE
        train_loss /= num_batches
        train_r_squared /= num_batches

        self.train_writer.add_scalar('MSE/total_loss', train_loss, epoch_n)
        self.train_writer.add_scalar('Train_R-squared/total_train_R-squared', train_r_squared, epoch_n)

        print(40 * '++')
        print(f'Train Metrics on epoch {epoch_n}:')
        print(f'R-squared: {train_r_squared:.3f}')
        print(40 * '++')

        return train_loss, train_r_squared

    def valid_classifier(self, epoch_n: int) -> Tuple[float, float, float]:
        """
        :param epoch_n:
        :return:
        """
        self.model.eval()

        valid_loss, valid_accuracy, valid_f1_score = 0.0, 0.0, 0.0
        num_batches = len(self.valid_loader)

        with torch.no_grad():
            for i, data in enumerate(self.valid_loader):
                images = data['image'].to(self.device)
                labels = data['label'].to(self.device)

                predictions = self.model(images)
                loss = self.loss_criterion(predictions, labels)

                prediction_class = torch.argmax(predictions, dim=-1)
                f1_scr = f1_score(prediction_class.cpu(), labels.cpu(), average='weighted')
                correct = prediction_class.eq(labels).sum().item()
                accuracy = correct / len(labels)

                valid_loss += loss.item()
                valid_accuracy += accuracy
                valid_f1_score += f1_scr

                print('-----LOGGING VALIDATION BATCH------')
                print(f'Epoch - {epoch_n}, Batch - {(i+1)}, Total batch - {num_batches}')
                print('Losses/total_loss', valid_loss / (i+1))
                print('Accuracy/total_accuracy', valid_accuracy / (i+1))
                print('Valid_f1_score/total_valid_f1_score', valid_f1_score / (i+1))
                print(40 * '--')

        # GET AVERAGE
        valid_loss /= num_batches
        valid_accuracy /= num_batches
        valid_f1_score /= num_batches

        self.valid_writer.add_scalar('Losses/total_loss', valid_loss, epoch_n)
        self.valid_writer.add_scalar('Accuracy/total_accuracy', valid_accuracy, epoch_n)
        self.valid_writer.add_scalar('Valid_f1_score/total_valid_f1_score', valid_f1_score, epoch_n)

        print(40 * '++')
        print(f'Valid Metrics on epoch {epoch_n}:')
        print(f'Accuracy: {valid_accuracy:.3f}, Avg loss: {valid_loss:.8f}')
        print(f'F1-score: {valid_f1_score:.3f}')
        print(40 * '++')

        return valid_loss, valid_accuracy, valid_f1_score

    def valid_regressor(self, epoch_n: int) -> Tuple[float, float]:
        """
        :param epoch_n:
        :return:
        """
        self.model.train()

        valid_loss, valid_r_squared = 0.0, 0.0
        num_batches = len(self.valid_loader)

        with torch.no_grad():
            for i, data in enumerate(self.valid_loader):
                images = data['image'].to(self.device)
                labels = data['label'].to(self.device)
                print(f'Label {labels[0]}')

                predictions = self.model(images)
                print(f'Prediction {predictions[0]}')

                loss = self.loss_criterion(predictions, labels.unsqueeze(1).float())
                print(f'Loss {loss}')

                r_squared = (1 - loss / torch.var(labels)).item()

                valid_loss += loss.item()
                valid_r_squared += r_squared

                print('-----LOGGING VALIDATION BATCH------')
                print(f'Epoch - {epoch_n}, Batch - {(i+1)}, Total batch - {num_batches}')
                print('MSE/total_loss', valid_loss / (i+1))
                print('Valid_R-squared/total_valid_R-squared', valid_r_squared / (i+1))
                print(40 * '--')

        # GET AVERAGE
        valid_loss /= num_batches
        valid_r_squared /= num_batches

        self.valid_writer.add_scalar('MSE/total_loss', valid_loss, epoch_n)
        self.valid_writer.add_scalar('Valid_R-squared/total_valid_R-squared', valid_r_squared, epoch_n)

        print(40 * '++')
        print(f'Valid Metrics on epoch {epoch_n}:')
        print(f'R-squared: {valid_r_squared:.3f}')
        print(40 * '++')

        return valid_loss, valid_r_squared
