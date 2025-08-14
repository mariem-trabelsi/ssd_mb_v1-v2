import os
import sys
import logging
import argparse
import datetime
import itertools
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from vision.utils.misc import str2bool, Timer, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config, mobilenetv1_ssd_config, mobilenetv2_ssd_config, squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
parser.add_argument("--dataset-type", default="voc", type=str)
parser.add_argument('--datasets', '--data', nargs='+', default=["data/model_v0/"])
parser.add_argument('--balance-data', action='store_true')
parser.add_argument('--net', default="mb2-ssd-lite")
parser.add_argument('--resolution', type=int, default=300)
parser.add_argument('--freeze-base-net', action='store_true')
parser.add_argument('--freeze-net', action='store_true')
parser.add_argument('--mb2-width-mult', default=1.0, type=float)
parser.add_argument('--base-net', help='Pretrained base model')
parser.add_argument('--pretrained-ssd', default='models/mb2-ssd-lite-mp-0_686.pth', type=str)
parser.add_argument('--resume', default=None, type=str, help='Checkpoint file to resume training from')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--base-net-lr', default=0.001, type=float)
parser.add_argument('--extra-layers-lr', default=None, type=float)
parser.add_argument('--scheduler', default="cosine", type=str)
parser.add_argument('--milestones', default="80,100", type=str)
parser.add_argument('--t-max', default=100, type=float)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--num-epochs', '--epochs', default=900, type=int)
parser.add_argument('--num-workers', default=2, type=int)
parser.add_argument('--validation-epochs', default=5, type=int)
parser.add_argument('--validation-mean-ap', default=True, type=str2bool)
parser.add_argument('--debug-steps', default=10, type=int)
parser.add_argument('--use-cuda', default=False, type=str2bool)
parser.add_argument('--checkpoint-folder', '--model-dir', default='models/model_mb2/')
parser.add_argument('--log-level', default='info', type=str)

args = parser.parse_args()

logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)

tensorboard = SummaryWriter(log_dir=os.path.join(args.checkpoint_folder, "tensorboard", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
logging.info(f"Using device: {DEVICE}")

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num_batches = 0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        num_batches += 1

        if i and i % debug_steps == 0:
            logging.info(f"Epoch: {epoch}, Step: {i}/{len(loader)}, Avg Loss: {running_loss/debug_steps:.4f}")

    logging.info(f"Epoch: {epoch}, Training Loss: {running_loss/num_batches:.4f}")

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1
        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
        running_loss += loss.item()
    return running_loss / num

def extract_epoch_from_filename(filename):
    """Extract epoch number from checkpoint filename like 'mb2-ssd-lite-Epoch-225-Loss-2.6423.pth'"""
    try:
        parts = filename.split('-')
        for i, part in enumerate(parts):
            if part.lower() == 'epoch' and i + 1 < len(parts):
                return int(parts[i + 1])
    except (ValueError, IndexError):
        pass
    return None

if __name__ == '__main__':
    timer = Timer()
    logging.info(args)

    if args.checkpoint_folder:
        args.checkpoint_folder = os.path.expanduser(args.checkpoint_folder)
        os.makedirs(args.checkpoint_folder, exist_ok=True)

    # Select net and config
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
        config.set_image_size(args.resolution)
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv2_ssd_config
        config.set_image_size(args.resolution)
    else:
        logging.fatal("The net type is wrong.")
        sys.exit(1)

    # Datasets
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform, target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path, transform=train_transform, target_transform=target_transform, dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        datasets.append(dataset)
    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)

    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(dataset_path, transform=test_transform, target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path, transform=test_transform, target_transform=target_transform, dataset_type="test")
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)

    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = float('inf')
    
    # Load pretrained model if specified and no resume checkpoint
    if args.pretrained_ssd and not args.resume:
        logging.info(f"Loading pretrained SSD model from {args.pretrained_ssd}")
        if os.path.isfile(args.pretrained_ssd):
            try:
                pretrained_dict = torch.load(args.pretrained_ssd, map_location=DEVICE)
                if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
                    net.load_state_dict(pretrained_dict['model_state_dict'])
                else:
                    net.load_state_dict(pretrained_dict)
                logging.info("Pretrained model loaded successfully")
            except Exception as e:
                logging.warning(f"Could not load pretrained model: {e}")
        else:
            logging.warning(f"Pretrained model file not found: {args.pretrained_ssd}")
    
    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    params = [
        {'params': net.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters()), 'lr': extra_layers_lr},
        {'params': itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())}
    ]

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.scheduler == 'multi-step':
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args.t_max)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        sys.exit(1)

    last_epoch = -1
    if args.resume:
        if not os.path.isfile(args.resume):
            logging.fatal(f"Checkpoint file not found: {args.resume}")
            sys.exit(1)
        
        logging.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format checkpoint (dictionary with metadata)
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("Optimizer state loaded")
                except Exception as e:
                    logging.warning(f"Could not load optimizer state: {e}")
            
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logging.info("Scheduler state loaded")
                except Exception as e:
                    logging.warning(f"Could not load scheduler state: {e}")
            
            last_epoch = checkpoint.get('epoch', -1)
            min_loss = checkpoint.get('loss', float('inf'))
            logging.info(f"Resumed from epoch {last_epoch} with loss {min_loss:.4f}")
        else:
            # Old format checkpoint (just model weights) or pretrained model
            net.load_state_dict(checkpoint)
            
            # Try to extract epoch from filename for old format checkpoints
            filename = os.path.basename(args.resume)
            extracted_epoch = extract_epoch_from_filename(filename)
            if extracted_epoch is not None:
                last_epoch = extracted_epoch
                logging.info(f"Extracted epoch {last_epoch} from filename")
            
            logging.info("Loaded weights only (no optimizer/scheduler state). Training will continue from extracted/default epoch.")

    # Train for remaining epochs
    for epoch in range(last_epoch + 1, args.num_epochs):
        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step()

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}")
            tensorboard.add_scalar('Loss/val', val_loss, epoch)

            # Save checkpoint in new format
            checkpoint_path = os.path.join(args.checkpoint_folder, f"checkpoint-epoch-{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'args': vars(args)  # Save training arguments for reference
            }, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Also save in old format for compatibility (optional)
            if val_loss < min_loss:
                min_loss = val_loss
                old_format_path = os.path.join(args.checkpoint_folder, f"mb2-ssd-lite-Epoch-{epoch}-Loss-{val_loss:.4f}.pth")
                torch.save(net.state_dict(), old_format_path)
                logging.info(f"Saved best model in old format: {old_format_path}")

    tensorboard.close()
