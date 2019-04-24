import time

class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = next(self.iterator)
        self.last_duration = (time.time() - start)
        return n

    next = __next__


class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self):
        return self.val


def format_dict_of_loss(dict_loss):
    try:
        string = ', '.join([('{}: {:' + ('.3f' \
                            if value >= 0.001 else '.1e') +'}').format(name, value) 
                            for name, value in dict_loss.items()])
    except (TypeError, ValueError) as e:
        print(dict_loss)
        string = '[Log Error] ' + str(e)

    return string