import os
import json
import wandb
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dfine.memory import ReplayBuffer
from dfine.train import train_backbone, train_z_decoder
from dfine.test import test_k_step_prediction
from dfine.data_loader import load_from_file
from sklearn.preprocessing import StandardScaler


def generate_id():
    """
        generates run id based on the time
    """
    return datetime.now().strftime("%Y%m%d_%H%M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFINE")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log-dir", type=str, default="log", help="logging directory")
    parser.add_argument("--run-id", type=str, default=generate_id(), help="id associated with this run")
    parser.add_argument("--data-path", type=str, required=True, help="name of the minari dataset")
    parser.add_argument("--num-updates", type=int, default=2500, help="number of gradient descent steps")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="train-test split ratio")
    parser.add_argument("--test-interval", type=int, default=10, help="number of training steps before testing")
    parser.add_argument("--x-dim", type=int, default=30, help="x(state) dimension")
    parser.add_argument("--a-dim", type=int, default=100, help="a(intermediate state) dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden layer dimension for encoder and decoder")
    parser.add_argument("--min-var", type=float, default=1e-2, help="minimum var for states")
    parser.add_argument("--chunk-length", type=int, default=50, help="length of chunks used for the update step")
    parser.add_argument("--prediction-k", type=int, default=24, help="number of future steps prediction")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay of the optimizer")
    parser.add_argument("--clip-grad-norm", type=float, default=1000.0, help="clip gradients to this value")
    parser.add_argument("--disable-gpu", action="store_true", default=False, help="disable using gpu")
    parser.add_argument("--notes", type=str, default="", help="extra notes to add to the run")
    parser.add_argument("--run-name", type=str, default="DFINE", help="name of the run")
    parser.add_argument("--filtering-weight", type=float, default=1.0, help="weight for the filtering in the loss")
    parser.add_argument("--test-k", nargs="+", type=int, help="a list of k, for k step ahead prediction test")

    args = parser.parse_args()

    # prepare logging
    save_dir = Path(args.log_dir) / args.run_id
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f)
    
    wandb.init(
        project="Controlling from high-dimensional observations",
        name=args.run_name,
        config=vars(args),
        notes=args.notes,
    )

    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # load the dataset and normalize it
    # load the dataset and normalize it
    data_path = Path(args.data_path)
    y, z = load_from_file(data_path=data_path)
    test_size = int(y.shape[0] * args.test_ratio)
    train_size = y.shape[0] - test_size
    y_train, y_test = y[:train_size], y[train_size:]
    z_train, z_test = z[:train_size], z[train_size:]
    y_scaler, z_scaler = StandardScaler(), StandardScaler()
    y_scaler.fit(y_train)
    z_scaler.fit(z_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)
    z_train = z_scaler.transform(z_train)
    z_test = z_scaler.transform(z_test)
    train_buffer = ReplayBuffer.from_numpy(y=y_train, z=z_train)
    test_buffer = ReplayBuffer.from_numpy(y=y_test, z=z_test)

    print("training backbone ...")
    encoder, decoder, dynamics_model = train_backbone(args=args, train_buffer=train_buffer, test_buffer=test_buffer)

    print("training z (behavior) decoder")
    z_decoder = train_z_decoder(
        args=args,
        encoder=encoder,
        dynamics_model=dynamics_model,
        train_buffer=train_buffer,
        test_buffer=test_buffer
    )

    print("testing")
    for k in args.test_k:
        test_k_step_prediction(
            args=args,
            encoder=encoder,
            decoder=decoder,
            z_decoder=z_decoder,
            dynamics_model=dynamics_model,
            z=torch.tensor(z_test).unsqueeze(1),
            y=torch.tensor(y_test).unsqueeze(1),
            prediction_k=k,
        )

    wandb.finish()