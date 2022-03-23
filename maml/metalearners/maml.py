import torch
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from maml.utils import update_parameters_multiGPU, tensors_to_device, compute_mIoU, get_IoU
import os
import copy
import shutil
from skimage.io import imsave

__all__ = ['ModelAgnosticMetaLearning', 'MAML']

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]


def inverse_image(image, mean=np.array([0.485, 0.456, 0.406]),
                  std=np.array([0.229, 0.224, 0.225])):
    # input image cxhxw
    for c in range(3):
        image[c] = image[c] * std[c] + mean[c]
    image = np.clip(image, 0, 1.0) * 255
    image = image.transpose(1, 2, 0)
    ret = image.astype(np.uint8)

    return ret


def visualize_mask(mask):
    h, w = mask.shape
    mask3 = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        mask3[y][x] = COLORS[mask[y][x] - 1]

    return mask3


class ModelAgnosticMetaLearning(object):
    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None, test_every=200,
                 loss_function=F.cross_entropy, device=None):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.test_every = test_every
        self.loss_function = loss_function
        self.device = device

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                                                             dtype=param.dtype, device=self.device,
                                                             requires_grad=learn_step_size)) for (name, param)
                                         in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                                          device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
            if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                                         for group in self.optimizer.param_groups])

    def get_outer_loss(self, meta_batch, meta_info, summary_writer, summary, meta_split, query_set_before=False):
        support_images, support_masks, query_images, query_masks = meta_batch
        batch_sz = support_images.size(0)
        results = {
            'batch_sz': batch_sz,
            'inner_losses': np.zeros((self.num_adaptation_steps,  # support set
                                      batch_sz), dtype=np.float32),
            'outer_losses': np.zeros((batch_sz,), dtype=np.float32),  # query set
            'mean_outer_loss': 0.
        }

        results.update({
            'loss_before_query': np.zeros((batch_sz,), dtype=np.float32),  # query set
            'loss_after_query': np.zeros((batch_sz,), dtype=np.float32),  # query set
            'mIoU_before_query': np.zeros((batch_sz,), dtype=np.float32),  # query set
            'mIoU_after_query': np.zeros((batch_sz,), dtype=np.float32)  # query set
        })

        mean_outer_loss = torch.tensor(0., device=self.device)
        for episode_id, (support_inputs, support_targets, query_inputs, query_targets, task_name) \
                in enumerate(zip(support_images, support_masks, query_images, query_masks, meta_info)):
            # benchmark inner loss on query set
            if query_set_before:
                params = OrderedDict(self.model.module.meta_named_parameters(task_name))
                params_query = OrderedDict()
                for name, param in params.items():
                    params_query[name] = param.clone()
                # Todo: params expand to 0 dim
                self.expand_params(query_inputs, params_query)
                params_query = tensors_to_device(params_query, device=self.device)
                before_query_logits = self.model(query_inputs, params=params_query)
                before_query_loss = self.loss_function(before_query_logits, query_targets)
                results['loss_before_query'][episode_id] = before_query_loss.item()
                results['mIoU_before_query'][episode_id] = compute_mIoU(before_query_logits, query_targets)

            params, adaptation_results = self.adapt_multiGPU(support_inputs,
                                                             support_targets,
                                                             task_name,
                                                             num_adaptation_steps=self.num_adaptation_steps,
                                                             step_size=self.step_size,
                                                             first_order=self.first_order,
                                                             support_set_after=True)

            results['inner_losses'][:, episode_id] = adaptation_results['inner_losses']

            with torch.set_grad_enabled(self.model.training):
                self.expand_params(query_inputs, params)
                query_logits = self.model(query_inputs, params=params)
                outer_loss = self.loss_function(query_logits, query_targets)
                results['loss_after_query'][episode_id] = outer_loss.item()
                results['outer_losses'][episode_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            results['mIoU_after_query'][episode_id] = compute_mIoU(query_logits, query_targets)

            print('Meta-training query loss {} before {}, after {}'.format(task_name,
                                                                           results['loss_before_query'][episode_id]
                                                                           if query_set_before else None,
                                                                           results['loss_after_query'][episode_id]))

            print('Meta-training query mIoU {} before {}, after {}'.format(task_name,
                                                                           results['mIoU_before_query'][episode_id]
                                                                           if query_set_before else None,
                                                                           results['mIoU_after_query'][episode_id]))

        # outer loop
        mean_outer_loss.div_(batch_sz)
        results['mean_outer_loss'] = mean_outer_loss.item()

        # task specific log
        support_losses_task = {}
        if query_set_before:
            query_loss_before_task = {}
            query_mIoU_before_task = {}

        query_loss_after_task = {}
        query_mIoU_after_task = {}

        for episode in range(batch_sz):
            if meta_info[episode] not in support_losses_task:
                support_losses_task[meta_info[episode]] = [results['inner_losses'][-1, episode]]
            else:
                support_losses_task[meta_info[episode]].append(results['inner_losses'][-1, episode])

            if query_set_before:
                if meta_info[episode] not in query_loss_before_task:
                    query_loss_before_task[meta_info[episode]] = [results['loss_before_query'][episode]]
                else:
                    query_loss_before_task[meta_info[episode]].append(results['loss_before_query'][episode])
                if meta_info[episode] not in query_mIoU_before_task:
                    query_mIoU_before_task[meta_info[episode]] = [results['mIoU_before_query'][episode]]
                else:
                    query_mIoU_before_task[meta_info[episode]].append(results['mIoU_before_query'][episode])
            if meta_info[episode] not in query_loss_after_task:
                query_loss_after_task[meta_info[episode]] = [results['loss_after_query'][episode]]
            else:
                query_loss_after_task[meta_info[episode]].append(results['loss_after_query'][episode])
            if meta_info[episode] not in query_mIoU_after_task:
                query_mIoU_after_task[meta_info[episode]] = [results['mIoU_after_query'][episode]]
            else:
                query_mIoU_after_task[meta_info[episode]].append(results['mIoU_after_query'][episode])

        for task in set(meta_info):
            support_losses_task[task] = np.array(support_losses_task[task]).mean()
            summary_writer.add_scalar(
                '{}/support_loss/{}'.format(meta_split, task),
                support_losses_task[task], summary['step'])

            loss_dict = {}
            mIoU_dict = {}
            if query_set_before:
                loss_dict['before'] = np.array(query_loss_before_task[task]).mean()
                mIoU_dict['before'] = np.array(query_mIoU_before_task[task]).mean()

            loss_dict['after'] = np.array(query_loss_after_task[task]).mean()
            mIoU_dict['after'] = np.array(query_mIoU_after_task[task]).mean()

            summary_writer.add_scalars(
                '{}/query_loss/{}'.format(meta_split, task),
                loss_dict, summary['step'])
            summary_writer.add_scalars(
                '{}/query_mIoU/{}'.format(meta_split, task),
                mIoU_dict, summary['step'])

        summary_writer.add_scalar('{}/mean_query_loss'.format(meta_split), mean_outer_loss.item(), summary['step'])
        summary_writer.add_scalar('{}/mean_query_mIoU'.format(meta_split), np.mean(results['mIoU_after_query']),
                                  summary['step'])

        return mean_outer_loss, results

    def expand_params(self, inputs, params):
        if not isinstance(params, OrderedDict):
            raise ValueError()

        sample_size = inputs.size(0)

        for key in params:
            temp = params[key]
            weight_l = len(temp.shape)
            if weight_l == 4:
                params[key] = temp.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
            elif weight_l == 1:
                params[key] = temp.unsqueeze(0).repeat(sample_size, 1)
            elif weight_l == 2:
                params[key] = temp.unsqueeze(0).repeat(sample_size, 1, 1)
            else:
                raise ValueError()

    def adapt_multiGPU(self, inputs, targets, datasource_name,
                       num_adaptation_steps=1, step_size=0.1,
                       first_order=False, support_set_after=False,
                       calculate_inner_iou=False):

        # theta from last outer loop (without batch expansion)
        params = OrderedDict(self.model.module.meta_named_parameters(datasource_name))

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            # Todo: params expand to 0 dim
            self.expand_params(inputs, params)
            params = tensors_to_device(params, device=self.device)
            logits = self.model(inputs, params=params)

            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if calculate_inner_iou and step == 0:
                results['mIoU_before'] = compute_mIoU(logits, targets)

            self.model.zero_grad()
            # should assume params-in has expansion, but params-out is squeezed
            params = update_parameters_multiGPU(inner_loss,
                                                step_size=step_size, params=params,
                                                first_order=(not self.model.training) or first_order)

            if support_set_after:
                self.expand_params(inputs, params)
                new_logits = self.model(inputs, params=params)
                new_inner_loss = self.loss_function(new_logits, targets)
                for name, param in params.items():
                    params[name] = param[0]

            loss_dict = {'before': inner_loss.item()}
            if support_set_after:
                loss_dict['after'] = new_inner_loss.item()
                print('Meta-training support loss {} before {}, after {}'.format(datasource_name,
                                                                                 inner_loss.item(),
                                                                                 new_inner_loss.item()))
                if calculate_inner_iou:
                    new_inner_mIoU = compute_mIoU(new_logits, targets)
                    print('Meta-training support mIoU {} before {}, after {}'.format(datasource_name,
                                                                                     results['mIoU_before'],
                                                                                     new_inner_mIoU))
        return params, results

    def train_batch(self, dataloaders, validation_dataloader_support, validation_dataloader_query,
                        meta_validation_lr, summary_writer, summary, max_iterations, weight_path, validation_epochs):
        self.model.train()
        torch.set_grad_enabled(True)
        iteration = 0
        best_validation_mIoU = 0.0
        while iteration < max_iterations:
            for batch in dataloaders:
                if summary['step'] % self.test_every == 0:
                    # context switching, backbone + meta-train-head
                    meta_train_model_checkpoint = copy.deepcopy(self.model.state_dict())

                    # meta-validation-head is always just initialized
                    meta_validation_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_validation_lr)

                    validation_mIoU = self.meta_validation(epochs=validation_epochs, unet=self.model,
                                                           loaders={'support': validation_dataloader_support,
                                                                    'query': validation_dataloader_query},
                                                           device=self.device,
                                                           optimizer=meta_validation_optimizer,
                                                           criterion=self.loss_function,
                                                           summary_writer=summary_writer,
                                                           summary=summary)

                    print("validation_mIoU {} in step {}, best previous validation mIoU {}".format(validation_mIoU,
                                                                                                   summary['step'],
                                                                                                   best_validation_mIoU))
                    if validation_mIoU > best_validation_mIoU:
                        best_validation_mIoU = validation_mIoU
                        torch.save(meta_train_model_checkpoint, weight_path)

                    self.model.train()
                    torch.set_grad_enabled(True)
                    # context switch back
                    self.model.load_state_dict(meta_train_model_checkpoint)

                meta_info = batch[2]
                batch = tensors_to_device(batch[:2], device=self.device)

                self.optimizer.zero_grad()
                loss = 0

                for task_name in set(meta_info):
                    params = OrderedDict(self.model.module.meta_named_parameters(task_name))
                    images = batch[0][[info == task_name for info in meta_info]]
                    masks = batch[1][[info == task_name for info in meta_info]]
                    self.expand_params(images, params)
                    params = tensors_to_device(params, device=self.device)
                    pred = self.model(images, params=params)
                    loss_task = self.loss_function(pred, masks)
                    loss += loss_task

                    summary_writer.add_scalar(
                        'meta-train/loss_task/{}'.format(task_name), loss_task.item() / images.shape[0],
                        summary['step'])

                loss.backward()
                self.optimizer.step()

                print("training step {} loss {}".format(summary['step'], loss.item()))
                summary_writer.add_scalar('meta-train/batch_loss', loss.item(), summary['step'])
                summary['step'] += 1
                iteration += 1
                if iteration > max_iterations:
                    break

            if self.scheduler is not None:
                self.scheduler.step(epoch=None)

    def train_episodic(self, dataloaders, validation_dataloader_train, validation_dataloader_test,
                           meta_validation_lr, summary_writer, summary, max_iterations, weight_path, validation_epochs):
        best_validation_mIoU = 0.0
        iteration = 0

        while iteration < max_iterations:
            for meta_batch in dataloaders:
                if summary['step'] % self.test_every == 0:
                    # context switching, backbone + meta-train-head
                    meta_train_model_checkpoint = copy.deepcopy(self.model.state_dict())

                    # meta-validation-head is always just initialized
                    meta_validation_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_validation_lr)
                    validation_mIoU = self.meta_validation(epochs=validation_epochs, unet=self.model,
                                                           loaders={'support': validation_dataloader_train,
                                                                    'query': validation_dataloader_test},
                                                           device=self.device,
                                                           optimizer=meta_validation_optimizer,
                                                           criterion=self.loss_function,
                                                           summary_writer=summary_writer,
                                                           summary=summary)
                    print("validation_mIoU {} in step {}, best previous validation mIoU {}".format(validation_mIoU,
                                                                                                   summary['step'],
                                                                                                   best_validation_mIoU))
                    if validation_mIoU > best_validation_mIoU:
                        best_validation_mIoU = validation_mIoU
                        torch.save(meta_train_model_checkpoint, weight_path)

                    self.model.train()
                    torch.set_grad_enabled(True)
                    # context switch back
                    self.model.load_state_dict(meta_train_model_checkpoint)

                meta_info = meta_batch[4]
                meta_batch = tensors_to_device(meta_batch[:4], device=self.device)

                self.optimizer.zero_grad()
                outer_loss, results = self.get_outer_loss(meta_batch, meta_info, summary_writer,
                                                          summary, meta_split='meta_train', query_set_before=True)
                # yield results
                outer_loss.backward()
                self.optimizer.step()
                print("meta-training step {} complete".format(summary['step']))
                summary['step'] += 1
                iteration += 1

                if iteration >= max_iterations:
                    break

            if self.scheduler is not None:
                self.scheduler.step(epoch=None)

    def meta_validation(self, epochs, unet, loaders, device, optimizer, criterion,
                        summary_writer, summary, prediction_folder=None, save_model=False):

        if save_model:
            assert prediction_folder is not None
        step = 0
        best_meta_validation_mIoU = 0.0
        for epoch in range(epochs):
            for phase in ["support", "query"]:
                if phase == "support":
                    unet.train()
                else:
                    unet.eval()

                validation_pred = []
                validation_true = []

                for i, data in enumerate(loaders[phase]):
                    if phase == "support":
                        step += 1
                    image, target = data
                    image, target = image.to(device), target.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "support"):
                        pred = unet(image)
                        loss = criterion(pred, target)

                        if phase == "support":
                            print('meta-validation support epoch {} step {} loss {}'.format(epoch, step, loss.item()))
                            loss.backward()
                            optimizer.step()

                        if phase == "query":
                            pred_np = pred.detach().cpu().numpy()
                            validation_pred.extend(
                                [pred_np[s] for s in range(pred_np.shape[0])]
                            )
                            target_np = target.detach().cpu().numpy()
                            validation_true.extend(
                                [target_np[s] for s in range(target_np.shape[0])]
                            )

                if phase == "query":
                    print('num_class {}'.format(validation_pred[0].shape[0]))
                    avg_miou = get_IoU(
                        validation_pred,
                        validation_true,
                        num_classes=validation_pred[0].shape[0]
                    )

                    print('meta-validation query epoch {} average mIoU {}'.format(epoch, avg_miou))
                    if avg_miou > best_meta_validation_mIoU:
                        best_meta_validation_mIoU = avg_miou
                        if save_model:
                            torch.save(self.model.state_dict(), os.path.join(prediction_folder, 'best_validation.pt'))

        summary_writer.add_scalar('meta-validation/query_mIoU', best_meta_validation_mIoU, summary['step'])
        print('best validation mIoU is {}'.format(best_meta_validation_mIoU))

        if save_model:
            print('saving prediction results by best validation model')
            ckpt = torch.load(os.path.join(prediction_folder, 'best_validation.pt'), map_location=device)
            self.model.load_state_dict(ckpt)

            if os.path.exists(prediction_folder):
                shutil.rmtree(prediction_folder)
            os.mkdir(prediction_folder)

            file_id = 0
            for data in loaders["query"]:
                image, target = data
                image, target = image.to(device), target.to(device)
                pred = unet(image)
                pred_np = pred.detach().cpu().numpy()
                pred_np = pred_np.argmax(axis=1)
                target_np = target.detach().cpu().numpy()
                image_np = image.detach().cpu().numpy()

                for i in range(pred_np.shape[0]):
                    x = inverse_image(image_np[i])
                    filename = "{}-image.png".format(file_id)
                    filepath = os.path.join(prediction_folder, filename)
                    imsave(filepath, x)

                    pred_mask = visualize_mask(pred_np[i])
                    filename = "{}-predict.png".format(file_id)
                    filepath = os.path.join(prediction_folder, filename)
                    imsave(filepath, pred_mask)
                    print('saving {}'.format(filename))

                    true_mask = visualize_mask(target_np[i])
                    filename = "{}-true.png".format(file_id)
                    filepath = os.path.join(prediction_folder, filename)
                    imsave(filepath, true_mask)
                    file_id += 1

        return best_meta_validation_mIoU


MAML = ModelAgnosticMetaLearning
