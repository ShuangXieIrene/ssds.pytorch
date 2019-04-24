import torch

import os
def save_checkpoints(model, output_dir, checkpoint_prefix, epochs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
    filename = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), filename)
    with open(os.path.join(output_dir, 'checkpoint_list.txt'), 'a') as f:
        f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
    print('Wrote snapshot to: {:s}'.format(filename))


def find_previous_checkpoint(output_dir):
    if not os.path.exists(os.path.join(output_dir, 'checkpoint_list.txt')):
        return False
    with open(os.path.join(output_dir, 'checkpoint_list.txt'), 'r') as f:
        lineList = f.readlines()
    epoches, resume_checkpoints = [list() for _ in range(2)]
    for line in lineList:
        epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
        checkpoint = line[line.find(':') + 2:-1]
        epoches.append(epoch)
        resume_checkpoints.append(checkpoint)
    return epoches, resume_checkpoints


def resume_checkpoint(model, resume_checkpoint, resume_scope):
    if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
        print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
        return False
    print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
    checkpoint = torch.load(resume_checkpoint)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # print("=> Weigths in the checkpoints:")
    # print([k for k, v in list(checkpoint.items())])

    # remove the module in the parrallel model
    if 'module.' in list(checkpoint.items())[0][0]:
        pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        checkpoint = pretrained_dict

    # change the name of the weights which exists in other model
    # change_dict = {
            # }
    # for k, v in list(checkpoint.items()):
    #     for _k, _v in list(change_dict.items()):
    #         if _k in k:
    #             new_key = k.replace(_k, _v)
    #             checkpoint[new_key] = checkpoint.pop(k)

    # remove the output layers from the checkpoint
    # remove_list = {
    # }
    # for k in remove_list:
    #     checkpoint.pop(k+'.weight', None)
    #     checkpoint.pop(k+'.bias', None)

    # extract the weights based on the resume scope
    if resume_scope != '':
        pretrained_dict = {}
        for k, v in list(checkpoint.items()):
            for resume_key in resume_scope.split(','):
                if resume_key in k:
                    pretrained_dict[k] = v
                    break
        checkpoint = pretrained_dict

    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
    # print("=> Resume weigths:")
    # print([k for k, v in list(pretrained_dict.items())])

    checkpoint = model.state_dict()
    # print(set(pretrained_dict)-set(checkpoint))
    unresume_dict = set(checkpoint)-set(pretrained_dict)
    if len(unresume_dict) != 0:
        print("=> UNResume weigths:")
        print(unresume_dict)

    checkpoint.update(pretrained_dict)

    return model.load_state_dict(checkpoint)