import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib
matplotlib.use('Agg')
import numpy as np
from data_loader.video_dataloader import test_data_loader
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from models.Text import *
from models.Exp_CLIP import ExpCLIP_Test
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--load-model', type=str)
parser.add_argument('--job-id', type=str)
args = parser.parse_args()
pretrain_model_path = './checkpoint/' + args.job_id + "-model.pth"

print('************************')
for k, v in vars(args).items():
    print(k,'=',v)
print('************************')


# create model and load pre_trained parameters
model = ExpCLIP_Test(args)
model = torch.nn.DataParallel(model).cuda() 
state_dict = model.state_dict()
pre_train_model = torch.load(pretrain_model_path)
for name, param in pre_train_model.items():
    if "mlp.weight" in name:
        state_dict["module.projection_head.mlp.weight"].copy_(param)
model.eval()


def zero_shot_test(set=0, dataset_=None, mode_task=None, FER_prompt_=None, prompt_type=None):

    DATASET_PATH_MAPPING = {
        "RAFDB": "/data/EECS-IoannisLab/datasets/Static_FER_Datasets/RAFDB_Face/test/",
        "AffectNet7": "/data/EECS-IoannisLab/datasets/Static_FER_Datasets/AffectNet7_Face/test/",
        "AffectNet8": "/data/EECS-IoannisLab/datasets/Static_FER_Datasets/AffectNet8_Face/test/",
        "FERPlus": "/data/EECS-IoannisLab/datasets/Static_FER_Datasets/FERPlus_Face/test/",
        "DFEW": "./annotation/DFEW_set_"+str(set+1)+"_test.txt",
        "FERV39k": "./annotation/FERV39k_test.txt",
        "MAFW": "./annotation/MAFW_set_"+str(set+1)+"_test.txt",
        "AFEW": "./annotation/AFEW_validation.txt",
        }
    test_data_path = DATASET_PATH_MAPPING[dataset_]
    zero_shot_prompt = FER_prompt_[dataset_]
    if dataset_ in ["RAFDB", "AffectNet7", "DFEW", "FERV39k", "AFEW"]:
        prompt_number = int(len(zero_shot_prompt) / 7)
    elif dataset_ in ["AffectNet8", "FERPlus"]:
        prompt_number = int(len(zero_shot_prompt) / 8)
    elif dataset_ in ["MAFW"]:
        prompt_number = int(len(zero_shot_prompt) / 11)
 
    if mode_task == "Static_FER":
        batch_size_ = 512
        test_data = datasets.ImageFolder(test_data_path,
                                         transforms.Compose([transforms.Resize((224, 224)),
                                                             transforms.ToTensor()]))
        confusion_matrix_path = "./confusion_matrix/"+args.load_model+"-"+dataset_+'-'+prompt_type+'.pdf'
    elif mode_task == "Dynamic_FER":
        batch_size_ = 64
        test_data = test_data_loader(list_file=test_data_path,
                                     num_segments=16,
                                     duration=1,
                                     image_size=224)
        confusion_matrix_path = "./confusion_matrix/"+args.load_model+"-"+dataset_+ '-' + str(set)+'-'+prompt_type+'.pdf'
        
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size_,
                                              shuffle=False,
                                              num_workers=8,
                                              pin_memory=True)
    correct = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):

            images = images.cuda()
            target = target.cuda()
        
            if mode_task == "Static_FER":
                n,_,_,_ = images.shape
                logit_scale, image_features, text_features = model(image=images,text=zero_shot_prompt, mode_task="Static_FER") 
            elif mode_task == "Dynamic_FER":
                n,_,_,_,_ = images.shape
                logit_scale, image_features, text_features = model(image=images,text=zero_shot_prompt, mode_task="Dynamic_FER") 

            output = logit_scale * image_features @ text_features.t()
            output = output.view(n, -1, prompt_number)
            output = torch.mean(output, dim=-1)

            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(test_loader.dataset)
    
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    tick_marks = np.arange(len(Emotion_name_dic[dataset_]))
    plt.xticks(tick_marks, Emotion_name_dic[dataset_], rotation=45)
    plt.yticks(tick_marks, Emotion_name_dic[dataset_])

    fmt = '.2f'
    thresh = normalized_cm.max() / 2.
    for i, j in itertools.product(range(normalized_cm.shape[0]), range(normalized_cm.shape[1])):
        plt.text(j, i, format(normalized_cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if normalized_cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    return uar, war


def zero_shot_test_FER(FER_prompt_,type_):

    datasets_ = ["RAFDB", "AffectNet7", "AffectNet8", "FERPlus"]
    
    for dataset in datasets_:
        uar, war = zero_shot_test(dataset_=dataset, mode_task="Static_FER", FER_prompt_=FER_prompt_, prompt_type=type_)
        if dataset=="RAFDB":
            print('******************** Static FER Zero-shot Performance ********************') 
        print(f'************************* {dataset}')
        print(f"UAR/WAR: {uar:.2f}/{war:.2f}")

    datasets_ = [("DFEW", 5), ("FERV39k", 1), ("MAFW", 5), ("AFEW", 1)]
    print(f'******************** Dynamic FER Zero-shot Performance ********************')
    for dataset, all_fold in datasets_:
        UAR, WAR = 0.0, 0.0
        for set in range(all_fold):
            uar, war = zero_shot_test(set, dataset_=dataset, mode_task="Dynamic_FER", FER_prompt_=FER_prompt_, prompt_type=type_)
            UAR += float(uar)
            WAR += float(war)
        avg_uar = UAR / all_fold
        avg_war = WAR / all_fold
        print(f'************************* {dataset}')
        print(f"UAR/WAR: {avg_uar:.2f}/{avg_war:.2f}")


class RecorderMeter(object):
    pass    


if __name__ == "__main__":

    for i, FER_prompt in enumerate(FER_prompt_list):
        print(f'************************************************************************** Zero-shot Prompt Type: ', FER_prompt_type_list[i])
        type_ = "type"+str(i+1)
        zero_shot_test_FER(FER_prompt, type_)
    
