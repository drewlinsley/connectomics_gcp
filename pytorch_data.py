import os
import numpy as np
import pandas as pd
import torch
import torch_xla.core.xla_model as xm
import tensorflow as tf  # for reading TFRecord Dataset
import tensorflow_datasets as tfds  # for making tf.data.Dataset to return numpy arrays
import rising
from rising import loading
from rising.loading import Dataset
import pytorch_lightning as pl
from typing import Sequence, Optional, Union
from rising.transforms import Compose, ResizeNative
from rising.transforms.affine import BaseAffine
import random
from rising.transforms import NormZeroMeanUnitStd
from rising.loading import DataLoader
from tqdm import tqdm


# TPU = xm.xla_device()


from rising.transforms import Compose, ResizeNative

def common_per_sample_trafos():
        return Compose(ResizeNative(size=(32, 64, 32), keys=('data',), mode='trilinear'),
                       ResizeNative(size=(32, 64, 32), keys=('label',), mode='nearest'))

class RandomAffine(BaseAffine):
    """Base Affine with random parameters for scale, rotation and translation"""
    def __init__(self, scale_range: Optional[tuple] = None,
                 rotation_range: Optional[tuple] = None,
                 translation_range: Optional[tuple] = None,
                 degree: bool = True,
                 image_transform: bool = True,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 output_size: Optional[tuple] = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'nearest',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 reverse_order: bool = False,
                 **kwargs,):

        """
        Args:
            scale_range: tuple containing minimum and maximum values for scale.
                Actual values will be sampled from uniform distribution with these
                constraints.
            rotation_range: tuple containing minimum and maximum values for rotation.
                Actual values will be sampled from uniform distribution with these
                constraints.
            translation_range: tuple containing minimum and maximum values for translation.
                Actual values will be sampled from uniform distribution with these
                constraints.
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            degree: whether the given rotation(s) are in degrees.
                Only valid for rotation parameters, which aren't passed
                as full transformation matrix.
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be
                calculated dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the
                affine transf
        """
        super().__init__(scale=None, rotation=None, translation=None,
                         degree=degree,
                         image_transform=image_transform,
                         keys=keys,
                         grad=grad,
                         output_size=output_size,
                         adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         reverse_order=reverse_order,
                         **kwargs)

        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Samples Parameters for scale, rotation and translation
        before actual matrix assembly.

        Args:
            **data: dictionary containing a batch

        Returns:
            torch.Tensor: assembled affine matrix
        """
        ndim = data[self.keys[0]].ndim - 2

        if self.scale_range is not None:
            self.scale = [random.uniform(*self.scale_range) for _ in range(ndim)]

        if self.translation_range is not None:
            self.translation = [random.uniform(*self.translation_range) for _ in range(ndim)]

        if self.rotation_range is not None:
            if ndim == 3:
                self.rotation = [random.uniform(*self.rotation_range) for _ in range(ndim)]
            elif ndim == 1:
                self.rotation = random.uniform(*self.rotation_range)

        return super().assemble_matrix(**data)


class SoftDiceLoss(torch.nn.Module):
    """Soft Dice Loss"""
    def __init__(self, square_nom: bool = False,
                 square_denom: bool = False,
                 weight: Optional[Union[Sequence, torch.Tensor]] = None,
                 smooth: float = 1.):
        """
        Args:
            square_nom: whether to square the nominator
            square_denom: whether to square the denominator
            weight: additional weighting of individual classes
            smooth: smoothing for nominator and denominator

        """
        super().__init__()
        self.square_nom = square_nom
        self.square_denom = square_denom

        self.smooth = smooth

        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight)

            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes SoftDice Loss

        Args:
            predictions: the predictions obtained by the network
            targets: the targets (ground truth) for the :attr:`predictions`

        Returns:
            torch.Tensor: the computed loss value
        """
        # number of classes for onehot
        n_classes = predictions.shape[1]
        with torch.no_grad():
            targets_onehot = rising.transforms.functional.channel.one_hot_batch(
                targets.unsqueeze(1), num_classes=n_classes)
        # sum over spatial dimensions
        dims = tuple(range(2, predictions.dim()))

        # compute nominator
        if self.square_nom:
            nom = torch.sum((predictions * targets_onehot.float()) ** 2, dim=dims)
        else:
            nom = torch.sum(predictions * targets_onehot.float(), dim=dims)
        nom = 2 * nom + self.smooth

        # compute denominator
        if self.square_denom:
            i_sum = torch.sum(predictions ** 2, dim=dims)
            t_sum = torch.sum(targets_onehot ** 2, dim=dims)
        else:
            i_sum = torch.sum(predictions, dim=dims)
            t_sum = torch.sum(targets_onehot, dim=dims)

        denom = i_sum + t_sum.float() + self.smooth

        # compute loss
        frac = nom / denom

        # apply weight for individual classesproperly
        if self.weight is not None:
            frac = self.weight * frac

        # average over classes
        frac = - torch.mean(frac, dim=1)

        return frac


def binary_dice_coefficient(pred: torch.Tensor, gt: torch.Tensor,
                            thresh: float = 0.5, smooth: float = 1e-7) -> torch.Tensor:
    """
    computes the dice coefficient for a binary segmentation task

    Args:
        pred: predicted segmentation (of shape Nx(Dx)HxW)
        gt: target segmentation (of shape NxCx(Dx)HxW)
        thresh: segmentation threshold
        smooth: smoothing value to avoid division by zero

    Returns:
        torch.Tensor: dice score
    """

    assert pred.shape == gt.shape

    pred_bool = pred > thresh

    intersec = (pred_bool * gt).float()
    return 2 * intersec.sum() / (pred_bool.float().sum()
                                 + gt.float().sum() + smooth)


class Unet(pl.LightningModule):
    """Simple U-Net without training logic"""
    def __init__(self, hparams: dict):
        """
        Args:
            hparams: the hyperparameters needed to construct the network.
                Specifically these are:
                * start_filts (int)
                * depth (int)
                * in_channels (int)
                * num_classes (int)
        """
        super().__init__()
        # 4 downsample layers
        out_filts = hparams.get('start_filts', 16)
        depth = hparams.get('depth', 3)
        in_filts = hparams.get('in_channels', 1)
        num_classes = hparams.get('num_classes', 2)

        for idx in range(depth):
            down_block = torch.nn.Sequential(
                torch.nn.Conv3d(in_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True)
            )
            in_filts = out_filts
            out_filts *= 2

            setattr(self, 'down_block_%d' % idx, down_block)

        out_filts = out_filts // 2
        in_filts = in_filts // 2
        out_filts, in_filts = in_filts, out_filts

        for idx in range(depth-1):
            up_block = torch.nn.Sequential(
                torch.nn.Conv3d(in_filts + out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True)
            )

            in_filts = out_filts
            out_filts = out_filts // 2

            setattr(self, 'up_block_%d' % idx, up_block)

        self.final_conv = torch.nn.Conv3d(in_filts, num_classes, kernel_size=1)
        self.max_pool = torch.nn.MaxPool3d(2, stride=2)
        self.up_sample = torch.nn.Upsample(scale_factor=2)
        self._hparams = hparams

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forwards the :attr`input_tensor` through the network to obtain a prediction

        Args:
            input_tensor: the network's input

        Returns:
            torch.Tensor: the networks output given the :attr`input_tensor`
        """
        depth = self._hparams.get('depth', 3)

        intermediate_outputs = []

        # Compute all the encoder blocks' outputs
        for idx in range(depth):
            intermed = getattr(self, 'down_block_%d' % idx)(input_tensor)
            if idx < depth - 1:
                # store intermediate values for usage in decoder
                intermediate_outputs.append(intermed)
                input_tensor = getattr(self, 'max_pool')(intermed)
            else:
                input_tensor = intermed

        # Compute all the decoder blocks' outputs
        for idx in range(depth-1):
            input_tensor = getattr(self, 'up_sample')(input_tensor)

            # use intermediate values from encoder
            from_down = intermediate_outputs.pop(-1)
            intermed = torch.cat([input_tensor, from_down], dim=1)
            input_tensor = getattr(self, 'up_block_%d' % idx)(intermed)

        return getattr(self, 'final_conv')(input_tensor)


# Read the tfrecords
def read_labeled_tfrecord(example):
    tfrec_format = {
        "volume": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)

    volume = tf.reshape(example["volume"], shape=[])
    label = tf.reshape(example["label"], shape=[])
    volume = tf.io.decode_raw(volume, tf.float32)
    label = tf.io.decode_raw(label, tf.float32)
    volume = tf.reshape(volume, [64, 128, 128, 2])
    label = tf.reshape(label, [64, 128, 128, 6])
    return {"volume": volume, "label": label}


def get_dataset(files, batch_size=16, repeat=False, cache=False, shuffle=False, labeled=True, return_image_ids=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE)  # , compression_type="GZIP")
    if cache:
        # You'll need around 15GB RAM if you'd like to cache val dataset, and 50~60GB RAM for train dataset.
        ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 2)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    ds = ds.map(read_labeled_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return tfds.as_numpy(ds)


class TFRecordDataLoader(Dataset):
    def __init__(self, files=None, batch_size=2, cache=True, shuffle=True, train=True):
        files = "gs://serrelab/connectomics/tfrecords/celltype/cell_type_10_64_15.tfrecords_train.tfrecords"
        self.ds = get_dataset(
            files,
            batch_size=batch_size,
            cache=cache)
        self._iterator = iter(self.ds)
        self.num_examples = 500

        self.batch_size = batch_size
        # self._iterator = None

    def __iter__(self):
        raise NotImplementedError
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __getitem__(self, item: int) -> dict:
        """
        Loads and Returns a single sample

        Args:
            item: index specifying which item to load

        Returns:
            dict: the loaded sample
        """
        volume, label = next(self._iterator)  # .__iter__()
        return {'volume': torch.from_numpy(img).float(),
                'label': torch.from_numpy(label).float()}

    def __len__(self):
        n_batches = self.num_examples // self.batch_size
        if self.num_examples % self.batch_size == 0:
            return n_batches
        else:
            return n_batches + 1


class AverageMeter(object):
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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def max_memory_allocated():
    MB = 1024.0 * 1024.0
    mem = torch.cuda.max_memory_allocated() / MB
    return f"{mem:.0f} MB"


def train_fn(files, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0

    train_loader = TFRecordDataLoader(
        files, batch_size=CFG.batch_size, shuffle=True)
    for step, d in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = torch.from_numpy(d[0]).to(device)
        labels = torch.from_numpy(d[1]).to(device)

        batch_size = labels.size(0)
        y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels.view(-1))
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}] '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  'Elapsed: {remain:s} '
                  'Max mem: {mem:s}'
                  .format(
                   epoch+1, CFG.epochs, step, len(train_loader),
                   loss=losses,
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],
                   remain=timeSince(start, float(step + 1) / len(train_loader)),
                   mem=max_memory_allocated()))
    return losses.avg



def valid_fn(files, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    filenames = []
    targets = []
    preds = []
    start = end = time.time()
    valid_loader = TFRecordDataLoader(
        files, batch_size=CFG.batch_size * 2, shuffle=False)
    for step, d in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        targets.extend(d[1].reshape(-1).tolist())
        filenames.extend([f.decode("UTF-8") for f in d[2]])
        
        images = torch.from_numpy(d[0]).to(device)
        labels = torch.from_numpy(d[1]).to(device)

        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels.view(-1))
        losses.update(loss.item(), batch_size)

        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0:
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds).reshape(-1)
    return losses.avg, predictions, np.array(targets), np.array(filenames)




# ====================================================
# Train loop
# ====================================================
def train_loop(train_tfrecords: np.ndarray, val_tfrecords: np.ndarray, fold: int):
    
    LOGGER.info(f"========== fold: {fold} training ==========")
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, pretrained=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_tfrecords, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds, targets, files = valid_fn(val_tfrecords, model, criterion, device)
        valid_result_df = pd.DataFrame({"target": targets, "preds": preds, "id": files})
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(targets, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        SAVEDIR / f'{CFG.model_name}_fold{fold}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        SAVEDIR / f'{CFG.model_name}_fold{fold}_best_loss.pth')
    
    valid_result_df["preds"] = torch.load(SAVEDIR / f"{CFG.model_name}_fold{fold}_best_loss.pth",
                                          map_location="cpu")["preds"]

    return valid_result_df


class TrainableUNet(Unet):
    """A trainable UNet (extends the base class by training logic)"""
    def __init__(self, hparams: Optional[dict] = None):
        """
        Args:
            hparams: the hyperparameters needed to construct and train the network.
                Specifically these are:
                * start_filts (int)
                * depth (int)
                * in_channels (int)
                * num_classes (int)
                * min_scale (float)
                * max_scale (float)
                * min_rotation (int, float)
                * max_rotation (int, float)
                * batch_size (int)
                * num_workers(int)
                * learning_rate (float)

                For all of them exist usable default parameters.
        """
        if hparams is None:
            hparams = {}
        super().__init__(hparams)

        # define loss functions
        self.dice_loss = SoftDiceLoss(weight=[0., 1.])
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def train_dataloader(self) -> DataLoader:
        """
        Specifies the train dataloader

        Returns:
            DataLoader: the train dataloader
        """
        # construct dataset
        dataset = TFRecordDataLoader(train=True)  #, data_dir=data_dir)

        # specify batch transforms
        batch_transforms = Compose([
            RandomAffine(scale_range=(self._hparams.get('min_scale', 0.9), self._hparams.get('max_scale', 1.1)),
                         rotation_range=(self._hparams.get('min_rotation', -10), self._hparams.get('max_rotation', 10)),
                        keys=('data', 'label')),
            NormZeroMeanUnitStd(keys=('data',))
        ])

        # construct loader
        dataloader = DataLoader(dataset,
                                batch_size=self._hparams.get('batch_size', 1),
                                batch_transforms=batch_transforms,
                                shuffle=True,
                                sample_transforms=common_per_sample_trafos(),
                                pseudo_batch_dim=True,
                                num_workers=self._hparams.get('num_workers', 4))
        return dataloader

    def val_dataloader(self) -> DataLoader:
        # construct dataset
        dataset = TFRecordDataLoader(train=False)  # , data_dir=data_dir)

        # specify batch transforms (no augmentation here)
        batch_transforms = NormZeroMeanUnitStd(keys=('data',))

        # construct loader
        dataloader = DataLoader(dataset,
                                batch_size=self._hparams.get('batch_size', 1),
                                batch_transforms=batch_transforms,
                                shuffle=False,
                                sample_transforms=common_per_sample_trafos(),
                                pseudo_batch_dim=True,
                                num_workers=self._hparams.get('num_workers', 4))

        return dataloader

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimier to use for training

        Returns:
            torch.optim.Optimier: the optimizer for updating the model's parameters
        """
        return torch.optim.Adam(self.parameters(), lr=self._hparams.get('learning_rate', 1e-3))

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Defines the training logic

        Args:
            batch: contains the data (inputs and ground truth)
            batch_idx: the number of the current batch

        Returns:
            dict: the current loss value
        """
        x, y = batch['data'], batch['label']

        # remove channel dim from gt (was necessary for augmentation)
        y = y[:, 0].long()

        # obtain predictions
        pred = self(x)
        softmaxed_pred = torch.nn.functional.softmax(pred, dim=1)

        # Calculate losses
        ce_loss = self.ce_loss(pred, y)
        dice_loss = self.dice_loss(softmaxed_pred, y)
        total_loss = (ce_loss + dice_loss) / 2

        # calculate dice coefficient
        dice_coeff = binary_dice_coefficient(torch.argmax(softmaxed_pred, dim=1), y)

        # log values
        self.logger.experiment.add_scalar('Train/DiceCoeff', dice_coeff)
        self.logger.experiment.add_scalar('Train/CE', ce_loss)
        self.logger.experiment.add_scalar('Train/SoftDiceLoss', dice_loss)
        self.logger.experiment.add_scalar('Train/TotalLoss', total_loss)

        return {'loss': total_loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Defines the validation logic

        Args:
            batch: contains the data (inputs and ground truth)
            batch_idx: the number of the current batch

        Returns:
            dict: the current loss and metric values
        """
        x, y = batch['data'], batch['label']

        # remove channel dim from gt (was necessary for augmentation)
        y = y[:, 0].long()

        # obtain predictions
        pred = self(x)
        softmaxed_pred = torch.nn.functional.softmax(pred, dim=1)

        # calculate losses
        ce_loss = self.ce_loss(pred, y)
        dice_loss = self.dice_loss(softmaxed_pred, y)
        total_loss = (ce_loss + dice_loss) / 2

        # calculate dice coefficient
        dice_coeff = binary_dice_coefficient(torch.argmax(softmaxed_pred, dim=1), y)

        # log values
        self.logger.experiment.add_scalar('Val/DiceCoeff', dice_coeff)
        self.logger.experiment.add_scalar('Val/CE', ce_loss)
        self.logger.experiment.add_scalar('Val/SoftDiceLoss', dice_loss)
        self.logger.experiment.add_scalar('Val/TotalLoss', total_loss)

        return {'val_loss': total_loss, 'dice': dice_coeff}

    def validation_epoch_end(self, outputs: list) -> dict:
        """Aggregates data from each validation step

        Args:
            outputs: the returned values from each validation step

        Returns:
            dict: the aggregated outputs
        """
        mean_outputs = {}
        for k in outputs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outputs]).mean()

        tqdm.write('Dice: \t%.3f' % mean_outputs['dice'].item())
        return mean_outputs


def main(path):
    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')
    

    train_loader = TFRecordDataLoader(
        path, batch_size=2, shuffle=True)

    output_dir = 'logs'
    os.makedirs(output_dir, exist_ok=True)


    net = Unet({'num_classes': 6, 'in_channels': 2, 'depth': 3, 'start_filts': 128})



    import pdb;pdb.set_trace()
    for step, data in enumerate(train_loader):
        volume = data["volume"]
        label = data["label"]
        images = torch.from_numpy(d[0]).to(DEVICE)
        labels = torch.from_numpy(d[1]).to(DEVICE)



if __name__ == '__main__':
    # data = TFRecordDataLoader("gs://serrelab/connectomics/tfrecords/celltype/cell_type_10_64_15.tfrecords_train.tfrecords")
    path = "gs://serrelab/connectomics/tfrecords/celltype/cell_type_10_64_15.tfrecords_train.tfrecords"
    # main(path)
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning import Trainer

    early_stop_callback = EarlyStopping(monitor='dice', min_delta=0.001, patience=10, verbose=False, mode='max')

    nb_epochs = 50
    num_start_filts = 16
    num_workers = 8

    model = TrainableUNet({'start_filts': num_start_filts, 'num_workers': num_workers})
    output_dir = 'logs'
    os.makedirs(output_dir, exist_ok=True)
    trainer = Trainer(tpu_cores=8)  # , early_stop_callback=early_stop_callback, max_nb_epochs=nb_epochs)
    trainer.fit(model)

