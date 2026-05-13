"""Training script for DTNN on high-frequency stock data.

Usage::

    python train.py --data-path /path/to/data/ --k 1 --epochs 150
"""

import argparse
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
from torchinfo import summary

from dtnn.models import DTNN
from dtnn.data_utils import StockDataset, get_sampler
from dtnn.train_utils import train_model, evaluate
from sklearn.metrics import accuracy_score, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description='Train DTNN on stock data')
    parser.add_argument('--data-path', type=str, default='',
                        help='Path to directory containing train/test .txt files')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--k', type=int, default=1,
                        help='Label column index (0-based)')
    parser.add_argument('--T', type=int, default=100,
                        help='Time window length')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--model-name', type=str, default='dtnn_pytorch',
                        help='Filename prefix for saved model and loss plot')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-sampler', action='store_true',
                        help='Use WeightedRandomSampler for class imbalance')
    parser.add_argument('--train-file', type=str, default='Train_Dst_NoAuction_DecPre_CF_7.txt')
    parser.add_argument('--test-files', type=str, nargs='+',
                        default=['Test_Dst_NoAuction_DecPre_CF_7.txt',
                                 'Test_Dst_NoAuction_DecPre_CF_8.txt',
                                 'Test_Dst_NoAuction_DecPre_CF_9.txt'])
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'depth={args.depth}, LR={args.lr}, k={args.k}')

    # --- Load data ---
    path = args.data_path
    raw_data = np.loadtxt(path + args.train_file)
    train_raw = raw_data[:, :int(np.floor(raw_data.shape[1] * 0.8))]
    val_raw = raw_data[:, int(np.floor(raw_data.shape[1] * 0.8)):]

    test_parts = [np.loadtxt(path + f) for f in args.test_files]
    test_raw = np.hstack(test_parts)

    print(train_raw.shape, val_raw.shape, test_raw.shape)

    # --- Build datasets ---
    dataset_train = StockDataset(data_array=train_raw, k=args.k, num_classes=args.num_classes, T=args.T)
    dataset_val = StockDataset(data_array=val_raw, k=args.k, num_classes=args.num_classes, T=args.T)
    dataset_test = StockDataset(data_array=test_raw, k=args.k, num_classes=args.num_classes, T=args.T)

    if args.use_sampler:
        sampler = get_sampler(dataset_train.y)
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=args.batch_size, sampler=sampler
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=args.batch_size, shuffle=True
        )

    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)

    print(dataset_train.x.shape, dataset_train.y.shape)

    # --- Build model ---
    model = DTNN(
        time_slices=dataset_train.x.shape[1],
        num_classes=args.num_classes,
        dim=dataset_train.x.shape[2],
        depth=args.depth,
        heads=args.heads,
    )
    model.to(device)

    summary(model, [1, args.T, dataset_train.x.shape[2]])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Train ---
    model_path = f'./{args.model_name}.pth'
    train_losses, val_losses = train_model(
        model, criterion, optimizer,
        train_loader, val_loader,
        epochs=args.epochs, model_path=model_path, device=device,
    )

    # --- Plot loss curves ---
    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='validation loss')
    plt.savefig(f'{args.model_name}_loss.jpg')

    # --- Evaluate ---
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_acc, all_targets, all_predictions = evaluate(model, test_loader, device)

    print(f'Test acc: {test_acc:.4f}')
    print('accuracy_score:', accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))


if __name__ == '__main__':
    main()
