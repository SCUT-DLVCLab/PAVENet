# -*- coding:utf-8 -*-

import torch,argparse,time,pickle,os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from dataset import Numbers,TestSampler,collate_fn
from model import PAVENet
from utils import create_logger,load_ckpt
from dist import dist_skilled_forgery,dist_random_forgery
from verify import verify

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--train_user_number',type=int,default=202)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--cuda',type=bool,default=True)
parser.add_argument('--genuine_sample',type=int,default=20)
parser.add_argument('--forgery_sample',type=int,default=20)
parser.add_argument('--folder',type=str,default='data')
parser.add_argument('--weights',type=str,default='./weights')
parser.add_argument('--output',type=str,default='./output')
parser.add_argument('--log_root',type=str,default='./logs')
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--d_hidden',type=int,default=512)
parser.add_argument('--num_pattern',type=int,default=8)
parser.add_argument('--epoch',type=int,default=300)
parser.add_argument('--dropout',type=float,default=0.1)
parser.add_argument('--template_num',type=int,default=4)
parser.add_argument('--test_all_eer',action='store_true')
parser.add_argument('--name',type=str,default='ECSEVer')
parser.add_argument('--notes',type=str,default='')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

os.makedirs(opt.output,exist_ok=True)

logger = create_logger(opt.log_root,name=opt.name,test=True)
test_data_path = f'./{opt.folder}/hrds4bv-across-test.pkl'
# test_data_path = 'data/all1-random-test-200.pkl'
with open(test_data_path,'rb') as f:
    handwriting_info = pickle.load(f,encoding='iso-8859-1')
test_dataset = Numbers(handwriting_info,train=False)
users_cnt = test_dataset.config['users_cnt']
test_sampler = TestSampler(users_cnt,opt.genuine_sample,opt.forgery_sample)
test_loader = DataLoader(test_dataset,batch_sampler=test_sampler,collate_fn=collate_fn)

model = PAVENet(test_dataset.feature_dims,opt.num_pattern,opt.d_hidden,num_classes=opt.train_user_number)
if opt.cuda and torch.cuda.is_available():
    torch.cuda.set_device(f'cuda:{opt.gpu}')
    device = torch.device(f'cuda:{opt.gpu}')
else:
    device = torch.device('cpu')
model = model.to(device)

logger.info(f'data root: {test_data_path}\n'
    f'test loader length: {len(test_loader)}\ngenuine_sample: {opt.genuine_sample}\n'
    f'forgery_sample: {opt.forgery_sample}\npattern num: {opt.num_pattern}\n'
    f'seed: {opt.seed}\nmodel: {model.__class__.__name__}\nnotes: {opt.notes}')

@torch.no_grad()
def test():
    logger.info('<======test begins.======>')
    model.eval()
    output = []
    for _,(features,features_lens,_,_) in enumerate(test_loader):
        features = torch.from_numpy(features).to(device)
        features_lens = torch.tensor(features_lens).long().to(device)
        y_vector = model(features,features_lens)[0]
        # y_vector,_,_ = model(features)
        output.append(y_vector.cpu().numpy())

    dist_genuine,dist_forgery,dist_template = dist_skilled_forgery(output,opt.template_num,0,opt.genuine_sample,opt.forgery_sample)
    np.save(f'{opt.output}/dist_genuine.npy',dist_genuine)
    np.save(f'{opt.output}/dist_forgery.npy',dist_forgery)
    np.save(f'{opt.output}/dist_template.npy',dist_template)
    verify(logger,opt.template_num,opt.genuine_sample,opt.forgery_sample,rf=False)

    dist_genuine,dist_forgery,dist_template = dist_random_forgery(output,opt.template_num,0,opt.genuine_sample,opt.forgery_sample)
    np.save(f'{opt.output}/dist_genuine.npy',dist_genuine)
    np.save(f'{opt.output}/dist_forgery.npy',dist_forgery)
    np.save(f'{opt.output}/dist_template.npy',dist_template)
    verify(logger,opt.template_num,opt.genuine_sample,opt.forgery_sample,rf=True)

@torch.no_grad()
def test_all(template_num):
    logger.info('<======test all begins.======>')
    model.eval()
    output = []
    for _,(features,features_lens,_,_) in enumerate(test_loader):
        features = torch.from_numpy(features).to(device)
        features_lens = torch.tensor(features_lens).long().to(device)
        y_vector = model(features,features_lens)[0]
        output.append(y_vector.cpu().numpy())

    if template_num == 4 or template_num == 1:
        dist_genuine,dist_forgery,dist_template = dist_skilled_forgery(output,template_num,0,opt.genuine_sample,opt.forgery_sample)
        np.save(f'{opt.output}/dist_genuine.npy',dist_genuine)
        np.save(f'{opt.output}/dist_forgery.npy',dist_forgery)
        np.save(f'{opt.output}/dist_template.npy',dist_template)
        verify(logger,template_num,opt.genuine_sample,opt.forgery_sample,rf=False)

        dist_genuine,dist_forgery,dist_template = dist_random_forgery(output,template_num,0,opt.genuine_sample,opt.forgery_sample)
        np.save(f'{opt.output}/dist_genuine.npy',dist_genuine)
        np.save(f'{opt.output}/dist_forgery.npy',dist_forgery)
        np.save(f'{opt.output}/dist_template.npy',dist_template)
        verify(logger,template_num,opt.genuine_sample,opt.forgery_sample,rf=True)
    else:
        dist_genuine,dist_forgery,dist_template = dist_skilled_forgery(output,template_num,0,opt.genuine_sample,opt.forgery_sample)
        np.save(f'{opt.output}/dist_genuine.npy',dist_genuine)
        np.save(f'{opt.output}/dist_forgery.npy',dist_forgery)
        np.save(f'{opt.output}/dist_template.npy',dist_template)
        verify(logger,template_num,opt.genuine_sample,opt.forgery_sample,rf=False)

def main():
    if not opt.test_all_eer:
        load_ckpt(model,opt.weights,device,logger)
        time_elapsed_start = time.time()
        test()
        logger.info(f'time elapsed: {time.time() - time_elapsed_start:.5f}s\n')
    else:
        load_ckpt(model,opt.weights,device,logger)
        for i in range(4,0,-1):
            time_elapsed_start = time.time()
            test_all(i)
            logger.info(f'time elapsed: {time.time() - time_elapsed_start:.5f}s\n')

if __name__ == '__main__':
    main()