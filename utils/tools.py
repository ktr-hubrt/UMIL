import numpy
import torch.distributed as dist
import torch
import clip
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.signal as signal
from matplotlib import pyplot as plt

def match(scores1, scores2):
    #score shape: T,2
    score1 = scores1[:, 1]
    score2 = scores2[:, 1]
    iou = np.stack((score1,score2),axis=1).min(1).sum()
    iou_ = np.stack((score1,1-score2),axis=1).min(1).sum()

    if iou > iou_:
        return score2
    else:
        return 1-score2

def evaluate_result(vid2abnormality, anno_file, root=''):
    LABEL_PATH = anno_file
    gt = []
    ans = []
    GT = []
    ANS = []
    video_path_list = []
    videos = {}
    for video in open(LABEL_PATH):
        vid = video.strip().split(' ')[0].split('/')[-1]
        video_len = int(video.strip().split(' ')[1])
        sub_video_gt = np.zeros((video_len,), dtype=np.int8)
        anomaly_tuple = video.split(' ')[3:]
        for ind in range(len(anomaly_tuple) // 2):
            start = int(anomaly_tuple[2 * ind])
            end = int(anomaly_tuple[2 * ind + 1])
            if start > 0:
                sub_video_gt[start:end] = 1
        videos[vid] = sub_video_gt

    for vid in videos:
        if vid not in vid2abnormality.keys():
            print("The video %s is excluded on the result!" % vid)
            continue

        cur_ab = np.array(vid2abnormality[vid])
        if cur_ab.shape[0]==1:
            cur_ab = cur_ab[0, :,]
        else:
            cur_ab = cur_ab[:, 0,]
        cur_gt = np.array(videos[vid])
        ratio = float(len(cur_gt)) / float(len(cur_ab))
        cur_ans = np.zeros_like(cur_gt, dtype='float32')
        for i in range(len(cur_ab)):
            b = int(i * ratio + 0.5)
            e = int((i + 1) * ratio + 0.5)
            cur_ans[b: e] = cur_ab[i]

        cur_ans = postpress(cur_ans, seg_size=32)

        if cur_gt.max() >=1:
            gt.extend(cur_gt.tolist())
            ans.extend(cur_ans.tolist())

        GT.extend(cur_gt.tolist())
        ANS.extend(cur_ans.tolist())

    ret = roc_auc_score(gt, ans)
    Ret = roc_auc_score(GT, ANS)
    fpr, tpr, threshold = roc_curve(GT, ANS)

    if root != '':
        output_file = path + "AUC.npz"
        np.savez(output_file, fpr=fpr, tpr=tpr, thre=threshold)

    return Ret, ret

def postpress(curve, seg_size=32):
    leng = curve.shape[0]
    window_size = leng//seg_size
    new_curve = np.zeros_like(curve)
    for i in range(seg_size):
        new_curve[window_size*i:window_size*(i+1)] = np.mean(curve[window_size*i:window_size*(i+1)])
    if leng>window_size*seg_size:
        new_curve[seg_size*window_size:] = np.mean(curve[seg_size*window_size:])
    return new_curve

def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, optimizer_u, lr_scheduler_u, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if (epoch + 1) % 10 == 0:
        save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])

    return classes
