import os
import shutil
import yaml
import torch
import pandas as pd
import numpy as np
from datetime import datetime

from torch import nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score

from data_aug.dataset_test import MolTestDatasetWrapper
from models.ginet_finetune import GINet


def _save_config_file(log_dir, config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'config_finetune.yaml'), 'w') as config_file:
            yaml.dump(config, config_file)


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.dataset = dataset

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = config['fine_tune_from'].split('/')[0] + '-' + \
            config['fine_tune_from'].split('/')[-1] + '-' + config['task_name']
        subdir_name = current_time + '-' + config['dataset']['target']
        self.log_dir = os.path.join('experiments', dir_name, subdir_name)

        model_yaml_dir = os.path.join(config['fine_tune_from'], 'checkpoints')
        for fn in os.listdir(model_yaml_dir):
            if fn.endswith(".yaml"):
                model_yaml_fn = fn
                break
        model_yaml = os.path.join(model_yaml_dir, model_yaml_fn)
        model_config = yaml.load(open(model_yaml, "r"), Loader=yaml.FullLoader)
        self.model_config = model_config['model']
        self.model_config['dropout'] = self.config['model']['dropout']
        self.model_config['pool'] = self.config['model']['pool']

        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8']:
                # self.criterion = nn.L1Loss()
                self.criterion = nn.SmoothL1Loss()
            else:
                self.criterion = nn.MSELoss()

        # save config file
        _save_config_file(self.log_dir, self.config)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data):
        pred = model(data)

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.view(-1))
        elif self.config['dataset']['task'] == 'regression':
            # loss = self.criterion(pred, data.y)
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7']:
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        n_batches = len(train_loader)
        if n_batches < self.config['log_every_n_steps']:
            self.config['log_every_n_steps'] = n_batches
        
        model = GINet(self.config['dataset']['task'], **self.model_config).to(self.device)
        model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'output_layers' in name:
                print(name)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        if self.config['optim']['type'] == 'SGD':
            init_lr = self.config['optim']['base_lr'] * self.config['batch_size'] / 256
            optimizer = torch.optim.SGD(
                [   {'params': params, 'lr': init_lr}, 
                    {'params': base_params, 'lr': init_lr * self.config['optim']['base_ratio']}
                ],
                momentum=self.config['optim']['momentum'],
                weight_decay=self.config['optim']['weight_decay']
            )
        elif self.config['optim']['type'] == 'Adam':
            optimizer = torch.optim.Adam(
                [   {'params': params, 'lr': self.config['optim']['lr']}, 
                    {'params': base_params, 'lr': self.config['optim']['lr'] * self.config['optim']['base_ratio']}
                ],
                weight_decay=self.config['optim']['weight_decay']
            )
        else:
            raise ValueError('Not defined optimizer type!')

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rmse = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                data = data.to(self.device)
                loss = self._step(model, data)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    print(epoch_counter, bn, loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_roc_auc = self._validate(model, valid_loader)
                    if valid_roc_auc > best_valid_roc_auc:
                        best_valid_roc_auc = valid_roc_auc
                        # save the model weights
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rmse, valid_mae = self._validate(model, valid_loader)
                    if self.config["task_name"] in ['qm7', 'qm8'] and valid_mae < best_valid_mae:
                        best_valid_mae = valid_mae
                        # save the model weights
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model.pth'))
                    elif valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        # save the model weights
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model.pth'))
                
                valid_n_iter += 1

        return self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'checkpoints')
            ckp_path = os.path.join(checkpoints_folder, 'model.pth')
            state_dict = torch.load(ckp_path, map_location=self.device)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model {} with success.".format(ckp_path))
        
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                pred = model(data)
                loss = self._step(model, data)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions, squared=False)
            mae = mean_absolute_error(labels, predictions)
            print('Validation loss:', valid_loss, 'RMSE:', rmse, 'MAE:', mae)
            return valid_loss, rmse, mae

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.log_dir, 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded {} with success.".format(model_path))

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                pred = model(data)
                loss = self._step(model, data)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

        test_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions, squared=False)
            mae = mean_absolute_error(labels, predictions)
            print('Test loss:', test_loss, 'RMSE:', rmse, 'MAE:', mae)
            return test_loss, rmse, mae

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Test loss:', test_loss, 'ROC AUC:', roc_auc)
            return test_loss, roc_auc


def run(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config)
    return fine_tune.train()


def get_config():
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/bbbp/raw/BBBP.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox21/raw/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/clintox/raw/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/hiv/raw/HIV.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/bace/raw/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/sider/raw/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders", "Investigations", 
            "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances", 
            "Immune system disorders", "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", 
            "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders", "Blood and lymphatic system disorders", 
            "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders", 
            "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/muv/raw/muv.csv'
        target_list = [
            "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-692", "MUV-712", "MUV-713", 
            "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/freesolv/raw/SAMPL.csv'
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/esol/raw/delaney-processed.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/lipophilicity/raw/Lipophilicity.csv'
        target_list = ["exp"]

    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", 
            "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]

    else:
        raise ValueError('Unspecified dataset!')

    print(config)
    return config, target_list


if __name__ == '__main__':
    config, target_list = get_config()

    os.makedirs('experiments', exist_ok=True)
    dir_name = config['fine_tune_from'].split('/')[0] + '-' + \
        config['fine_tune_from'].split('/')[-1] + '-' + config['task_name']
    save_dir = os.path.join('experiments', dir_name)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    if config['dataset']['task'] == 'classification':
        save_list = []
        for target in target_list:
            config['dataset']['target'] = target
            roc_list = [target]
            test_loss, roc_auc = run(config)
            roc_list.append(roc_auc)
            save_list.append(roc_list)
        
        df = pd.DataFrame(save_list)
        fn = '{}_{}_ROC.csv'.format(config["task_name"], current_time)
        df.to_csv(os.path.join(save_dir, fn), index=False, header=['label', 'ROC-AUC'])
    
    elif config['dataset']['task'] == 'regression':
        save_rmse_list, save_mae_list = [], []
        for target in target_list:
            config['dataset']['target'] = target
            rmse_list, mae_list = [target], [target]
            test_loss, rmse, mae = run(config)
            rmse_list.append(rmse)
            mae_list.append(mae)
            
            save_rmse_list.append(rmse_list)
            save_mae_list.append(mae_list)
        
        df = pd.DataFrame(save_rmse_list)
        fn = '{}_{}_RMSE.csv'.format(config["task_name"], current_time)
        df.to_csv(os.path.join(save_dir, fn), index=False, header=['label', 'RMSE'])

        df = pd.DataFrame(save_mae_list)
        fn = '{}_{}_MAE.csv'.format(config["task_name"], current_time)
        df.to_csv(os.path.join(save_dir, fn), index=False, header=['label', 'MAE'])