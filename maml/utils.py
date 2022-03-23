import torch

from collections import OrderedDict
import numpy as np


def update_parameters_multiGPU(loss, params, step_size=0.5, first_order=False):
    grads = torch.autograd.grad(loss, params.values(),
                                create_graph=not first_order)

    out = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            out[name] = param[0] - step_size[name] * grad.sum(dim=0)
    else:
        for (name, param), grad in zip(params.items(), grads):
            out[name] = param[0] - step_size * grad.sum(dim=0)

    return out


def get_IoU(pred_list, gt_list, num_classes):
    Avg_mIoU = 0.0
    for pred, gt in zip(pred_list, gt_list):
        pred = pred.argmax(axis=0)
        assert (pred.shape == gt.shape)
        gt = gt.astype(np.float32)
        pred = pred.astype(np.float32)

        count = np.zeros((num_classes,))
        for j in range(num_classes):
            x = np.where(pred == j)
            p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
            x = np.where(gt == j)
            GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
            # pdb.set_trace()
            n_jj = set.intersection(p_idx_j, GT_idx_j)
            u_jj = set.union(p_idx_j, GT_idx_j)

            if len(GT_idx_j) != 0:
                count[j] = float(len(n_jj)) / float(len(u_jj))

        result_class = count
        # Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))
        miou = np.sum(result_class[:]) / num_classes
        print('query mIoU {}'.format(miou))
        Avg_mIoU += miou

    Avg_mIoU /= len(pred_list)

    return Avg_mIoU


def compute_mIoU(logits, targets):
    """Compute the mIoU """

    logits_np = logits.detach().cpu().numpy()
    logits_list = [logits_np[s] for s in range(logits_np.shape[0])]

    targets_np = targets.detach().cpu().numpy()
    targets_list = [targets_np[s] for s in range(targets_np.shape[0])]

    return get_IoU(logits_list, targets_list, num_classes=logits.shape[1])


def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
                             for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
                              for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """

    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'
