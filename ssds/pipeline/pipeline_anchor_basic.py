import sys
from tqdm import tqdm
import ssds.core.tools as tools

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

def train_anchor_based_epoch(model, data_loader, optimizer, criterion, priors, writer, epoch, device):
    model.train()

    title = 'Train: '
    progress = tqdm(tools.IteratorTimer(data_loader), ncols=150, total=len(data_loader), \
                    smoothing=.9, miniters=1, leave=True, desc=title)

    loss_writer = {'loc_loss':tools.AverageMeter(), 'cls_loss':tools.AverageMeter()}
    loss = {}

    for batch_idx, (images, targets) in enumerate(progress):
        # TODO: need to move the transpose at the preprocess
        images, targets = images.to(device), [anno.transpose(1,0).to(device) for anno in targets]

        out = model(images, phase='train')
        loss_l, loss_c = criterion(out, targets, priors)

        # some bugs in coco train2017. maybe the annonation bug.
        if loss_l.item() == float("Inf"):
            continue

        optimizer.zero_grad()
        total_loss = loss_l + loss_c
        total_loss.backward()
        optimizer.step()

        loss['loc_loss'] = loss_l.item()
        loss['cls_loss'] = loss_c.item()
        loss_writer['loc_loss'].update(loss['loc_loss'])
        loss_writer['cls_loss'].update(loss['cls_loss'])
        loss['total_loss'] = loss['loc_loss'] + loss['cls_loss']
        loss['lr'] = optimizer.param_groups[0]['lr']

        # log per iter
        progress.set_description(title + \
            tools.format_dict_of_loss(loss))
        progress.update(1)

    progress.close()
    loss['loc_loss'] = loss_writer['loc_loss'].avg
    loss['cls_loss'] = loss_writer['cls_loss'].avg
    loss['total_loss'] = loss['loc_loss'] + loss['cls_loss']
    loss['lr'] = optimizer.param_groups[0]['lr']

    print(CURSOR_UP_ONE + ERASE_LINE + '===>Avg Train: ' + \
                tools.format_dict_of_loss(loss))  

    # log for tensorboard
    for key, value in loss.items():
        writer.add_scalar('Train/{}'.format(key), value, epoch)

    return


# from ssds.utils.eval_utils import *


# def eval_anchor_based_epoch(model, data_loader, detector, criterion, writer, epoch, device):
#     model.eval()

#     title = 'eval: '
#     progress = tqdm(tools.IteratorTimer(data_loader), ncols=150, total=len(data_loader), \
#                     smoothing=.9, miniters=1, leave=True, desc=title)

#     loss_writer = {'loc_loss':tools.AverageMeter(), 'cls_loss':tools.AverageMeter()}
#     loss = {}

#     label = [list() for _ in range(model.num_classes)]
#     gt_label = [list() for _ in range(model.num_classes)]
#     score = [list() for _ in range(model.num_classes)]
#     size = [list() for _ in range(model.num_classes)]
#     npos = [0] * model.num_classes


#     for batch_idx, (images, targets) in enumerate(progress):
#         images, targets = images.to(device), targets.to(device)

#         # forward
#         out = model(images, phase='train')

#         # loss
#         loss_l, loss_c = criterion(out, targets)
#         out = (out[0], model.softmax(out[1].view(-1, model.num_classes)))

#         # detect
#         detections = detector.forward(out)

#         # evals
#         label, score, npos, gt_label = cal_tp_fp(detections, targets, label, score, npos, gt_label)
#         size = cal_size(detections, targets, size)
#         loc_loss += loss_l.data[0]
#         conf_loss += loss_c.data[0]

#         # log per iter
#         log = '\r==>Eval: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
#                 prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
#                 time=time, loc_loss=loss.items(), cls_loss=loss_c.items())

#         sys.stdout.write(log)
#         sys.stdout.flush()

#     # eval mAP
#     prec, rec, ap = cal_pr(label, score, npos)

#     # log per epoch
#     sys.stdout.write('\r')
#     sys.stdout.flush()
#     log = '\r==>Eval: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || mAP: {mAP:.6f}\n'.format(mAP=ap,
#             time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
#     sys.stdout.write(log)
#     sys.stdout.flush()

#     # log for tensorboard
#     writer.add_scalar('Eval/loc_loss', loc_loss/epoch_size, epoch)
#     writer.add_scalar('Eval/conf_loss', conf_loss/epoch_size, epoch)
#     writer.add_scalar('Eval/mAP', ap, epoch)
#     viz_pr_curve(writer, prec, rec, epoch)
#     viz_archor_strategy(writer, size, gt_label, epoch)