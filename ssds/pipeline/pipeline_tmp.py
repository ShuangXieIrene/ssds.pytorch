    def train_epoch(self, model, data_loader, optimizer, criterion, writer, epoch, use_gpu):
        model.train()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        for iteration in iter(range((epoch_size))):
            images, targets = next(batch_iterator)
            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            _t.tic()
            # forward
            out = model(images, phase='train')

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)

            # some bugs in coco train2017. maybe the annonation bug.
            if loss_l.data[0] == float("Inf"):
                continue

            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            time = _t.toc()
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]

            # log per iter
            log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.data[0], cls_loss=loss_c.data[0])

            sys.stdout.write(log)
            sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        lr = optimizer.param_groups[0]['lr']
        log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Train/lr', lr, epoch)


    def eval_epoch(self, model, data_loader, detector, criterion, writer, epoch, use_gpu):
        model.eval()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        label = [list() for _ in range(model.num_classes)]
        gt_label = [list() for _ in range(model.num_classes)]
        score = [list() for _ in range(model.num_classes)]
        size = [list() for _ in range(model.num_classes)]
        npos = [0] * model.num_classes

        for iteration in iter(range((epoch_size))):
        # for iteration in iter(range((10))):
            images, targets = next(batch_iterator)
            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            _t.tic()
            # forward
            out = model(images, phase='train')

            # loss
            loss_l, loss_c = criterion(out, targets)

            out = (out[0], model.softmax(out[1].view(-1, model.num_classes)))

            # detect
            detections = detector.forward(out)

            time = _t.toc()

            # evals
            label, score, npos, gt_label = cal_tp_fp(detections, targets, label, score, npos, gt_label)
            size = cal_size(detections, targets, size)
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]

            # log per iter
            log = '\r==>Eval: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.data[0], cls_loss=loss_c.data[0])

            sys.stdout.write(log)
            sys.stdout.flush()

        # eval mAP
        prec, rec, ap = cal_pr(label, score, npos)

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        log = '\r==>Eval: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || mAP: {mAP:.6f}\n'.format(mAP=ap,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Eval/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Eval/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Eval/mAP', ap, epoch)
        viz_pr_curve(writer, prec, rec, epoch)
        viz_archor_strategy(writer, size, gt_label, epoch)

    # TODO: HOW TO MAKE THE DATALOADER WITHOUT SHUFFLE
    # def test_epoch(self, model, data_loader, detector, output_dir, use_gpu):
    #     # sys.stdout.write('\r===> Eval mode\n')

    #     model.eval()

    #     num_images = len(data_loader.dataset)
    #     num_classes = detector.num_classes
    #     batch_size = data_loader.batch_size
    #     all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    #     empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

    #     epoch_size = len(data_loader)
    #     batch_iterator = iter(data_loader)

    #     _t = Timer()

    #     for iteration in iter(range((epoch_size))):
    #         images, targets = next(batch_iterator)
    #         targets = [[anno[0][1], anno[0][0], anno[0][1], anno[0][0]] for anno in targets] # contains the image size
    #         if use_gpu:
    #             images = Variable(images.cuda())
    #         else:
    #             images = Variable(images)

    #         _t.tic()
    #         # forward
    #         out = model(images, is_train=False)

    #         # detect
    #         detections = detector.forward(out)

    #         time = _t.toc()

    #         # TODO: make it smart:
    #         for i, (dets, scale) in enumerate(zip(detections, targets)):
    #             for j in range(1, num_classes):
    #                 cls_dets = list()
    #                 for det in dets[j]:
    #                     if det[0] > 0:
    #                         d = det.cpu().numpy()
    #                         score, box = d[0], d[1:]
    #                         box *= scale
    #                         box = np.append(box, score)
    #                         cls_dets.append(box)
    #                 if len(cls_dets) == 0:
    #                     cls_dets = empty_array
    #                 all_boxes[j][iteration*batch_size+i] = np.array(cls_dets)

    #         # log per iter
    #         log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
    #                 prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
    #                 time=time)
    #         sys.stdout.write(log)
    #         sys.stdout.flush()

    #     # write result to pkl
    #     with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
    #         pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    #     print('Evaluating detections')
    #     data_loader.dataset.evaluate_detections(all_boxes, output_dir)


    def test_epoch(self, model, data_loader, detector, output_dir, use_gpu):
        model.eval()

        dataset = data_loader.dataset
        num_images = len(dataset)
        num_classes = detector.num_classes
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

        _t = Timer()

        for i in iter(range((num_images))):
            img = dataset.pull_image(i)
            scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            if use_gpu:
                images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
            else:
                images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

            _t.tic()
            # forward
            out = model(images, phase='eval')

            # detect
            detections = detector.forward(out)

            time = _t.toc()

            # TODO: make it smart:
            for j in range(1, num_classes):
                cls_dets = list()
                for det in detections[0][j]:
                    if det[0] > 0:
                        d = det.cpu().numpy()
                        score, box = d[0], d[1:]
                        box *= scale
                        box = np.append(box, score)
                        cls_dets.append(box)
                if len(cls_dets) == 0:
                    cls_dets = empty_array
                all_boxes[j][i] = np.array(cls_dets)

            # log per iter
            log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
                    prograss='#'*int(round(10*i/num_images)) + '-'*int(round(10*(1-i/num_images))), iters=i, epoch_size=num_images,
                    time=time)
            sys.stdout.write(log)
            sys.stdout.flush()

        # write result to pkl
        with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
        print('Evaluating detections')
        data_loader.dataset.evaluate_detections(all_boxes, output_dir)


    def visualize_epoch(self, model, data_loader, priorbox, writer, epoch, use_gpu):
        model.eval()

        img_index = random.randint(0, len(data_loader.dataset)-1)

        # get img
        image = data_loader.dataset.pull_image(img_index)
        anno = data_loader.dataset.pull_anno(img_index)

        # visualize archor box
        viz_prior_box(writer, priorbox, image, epoch)

        # get preproc
        preproc = data_loader.dataset.preproc
        preproc.add_writer(writer, epoch)
        # preproc.p = 0.6

        # preproc image & visualize preprocess prograss
        images = Variable(preproc(image, anno)[0].unsqueeze(0), volatile=True)
        if use_gpu:
            images = images.cuda()

        # visualize feature map in base and extras
        base_out = viz_module_feature_maps(writer, model.base, images, module_name='base', epoch=epoch)
        extras_out = viz_module_feature_maps(writer, model.extras, base_out, module_name='extras', epoch=epoch)
        # visualize feature map in feature_extractors
        viz_feature_maps(writer, model(images, 'feature'), module_name='feature_extractors', epoch=epoch)

        model.train()
        images.requires_grad = True
        images.volatile=False
        base_out = viz_module_grads(writer, model, model.base, images, images, preproc.means, module_name='base', epoch=epoch)

        # TODO: add more...