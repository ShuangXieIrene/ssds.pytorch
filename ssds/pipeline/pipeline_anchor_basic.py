import sys
from tqdm import tqdm
import ssds.core.tools as tools

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

def train_anchor_basic(model, data_loader, optimizer, criterion, priors, writer, epoch, device):
    model.train()

    title = 'Train: '
    progress = tqdm(tools.IteratorTimer(data_loader), ncols=150, total=len(data_loader), \
                    smoothing=.9, miniters=1, leave=True, desc=title)

    loss_writer = {'loc_loss':tools.AverageMeter(), 'cls_loss':tools.AverageMeter()}
    loss = {}

    for batch_idx, (images, targets) in enumerate(progress):
        images, targets = images.to(device), targets.to(device)

        out = model(images, phase='train')
        loss_l, loss_c = criterion(out, targets, priors)

        # some bugs in coco train2017. maybe the annonation bug.
        if loss_l.data[0] == float("Inf"):
            continue

        optimizer.zero_grad()
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        loss['loc_loss'] = loss_writer['loc_loss']
        loss['cls_loss'] = loss_writer['cls_loss']
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