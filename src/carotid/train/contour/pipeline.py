import shutil

from monai.data import DataLoader
from os import path, makedirs
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

from carotid.utils import read_and_fill_default_toml, check_device, write_json
from carotid.train.utils import prediction_loop
from carotid.transform.contour.utils import ContourNet
from carotid.train.contour.utils import AnnotatedPolarDataset, compute_contour_df


def train(
    output_dir: str,
    raw_dir: str,
    contour_dir: str,
    contour_tsv: str = None,
    train_config_path: str = None,
    polar_config_dict: str = None,
    device: str = None,
    force: bool = False,
):
    device = check_device(device=device)

    if force and path.exists(output_dir):
        shutil.rmtree(output_dir)
    makedirs(output_dir)

    train_config_dict = read_and_fill_default_toml(
        train_config_path, default_filename="default_train.toml"
    )["contour_transform"]
    polar_config_dict = read_and_fill_default_toml(polar_config_dict)["polar_transform"]
    write_json(
        {
            "polar_transform": polar_config_dict,
            "train_params": train_config_dict,
            "data": {"raw_dir": raw_dir, "contour_dir": contour_dir},
        },
        path.join(output_dir, "parameters.json"),
    )

    if contour_tsv is not None:
        contour_df = pd.read_csv(contour_tsv, sep="\t")
    else:
        contour_df = compute_contour_df(raw_dir=raw_dir, contour_dir=contour_dir)

    # Generate splits
    participants = contour_df.participant_id.unique()

    splits = KFold(
        n_splits=train_config_dict["n_splits"],
        random_state=train_config_dict["random_seed"],
        shuffle=True,
    )
    for split_idx, (train_participants_idx, valid_participants_idx) in enumerate(
        splits.split(participants, participants)
    ):
        split_dir = path.join(output_dir, f"split-{split_idx}")
        makedirs(split_dir, exist_ok=True)

        # Create train and validation loaders
        train_df = contour_df[
            contour_df.participant_id.isin(participants[train_participants_idx])
        ].reset_index(drop=True)
        valid_df = contour_df[
            contour_df.participant_id.isin(participants[valid_participants_idx])
        ].reset_index(drop=True)

        train_dataset = AnnotatedPolarDataset(
            raw_dir,
            contour_dir,
            train_df,
            polar_config_dict,
            augmentation=True,
            dimension=train_config_dict["dimension"],
        )
        valid_dataset = AnnotatedPolarDataset(
            raw_dir,
            contour_dir,
            valid_df,
            polar_config_dict,
            augmentation=False,
            dimension=train_config_dict["dimension"],
        )

        train_loader = DataLoader(
            train_dataset, batch_size=train_config_dict["batch_size"], shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=train_config_dict["batch_size"], shuffle=False
        )

        # Write log
        training_tsv = path.join(split_dir, "training.tsv")
        training_df = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss"])
        training_df.to_csv(training_tsv, sep="\t", index=False)

        # Create model
        model = ContourNet(
            dropout=train_config_dict["dropout"],
            dimension=train_config_dict["dimension"],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_config_dict["learning_rate"]
        )
        loss_fn = torch.nn.SmoothL1Loss()
        best_validation_loss = torch.inf

        # Training loop
        for epoch in tqdm(range(train_config_dict["n_epochs"])):
            model.train()
            train_loss = 0
            validation_loss = 0

            # Training
            for train_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(train_batch["image"].to(device)).transpose(1, 2)
                target = train_batch["labels"].to(device)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(target)

            # Validation
            for validation_batch in valid_loader:
                model.eval()
                with torch.no_grad():
                    output = model(validation_batch["image"].to(device)).transpose(1, 2)
                target = validation_batch["labels"].to(device)
                loss = loss_fn(output, target)
                validation_loss += loss.item() * len(target)

            # Log epoch loss values
            row_df = pd.DataFrame(
                [
                    [
                        epoch,
                        train_loss / len(train_dataset),
                        validation_loss / len(valid_dataset),
                    ]
                ],
                columns=["epoch", "train_loss", "validation_loss"],
            )
            row_df.to_csv(training_tsv, sep="\t", index=False, header=False, mode="a")

            # Save model if best validation loss
            if validation_loss < best_validation_loss:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "class": model.__class__,
                        "epoch": epoch,
                        "validation_loss": validation_loss / len(valid_dataset),
                        "train_loss": train_loss / len(train_dataset),
                    },
                    path.join(split_dir, "model.pt"),
                )
                best_validation_loss = validation_loss

        # TODO deal with dimension in polar transform
        model = ContourNet(
            dropout=train_config_dict["dropout"],
            dimension=train_config_dict["dimension"],  # TODO doc
        ).to(device)
        model.load_state_dict(
            torch.load(path.join(split_dir, "model.pt"), map_location=device)["model"]
        )
        train_dataset = AnnotatedPolarDataset(
            raw_dir,
            contour_dir,
            train_df,
            polar_config_dict,
            augmentation=False,
            dimension=train_config_dict["dimension"],
        )
        train_loader = DataLoader(
            train_dataset, batch_size=train_config_dict["batch_size"], shuffle=False
        )

        prediction_loop(
            split_dir,
            model=model,
            test_loader=train_loader,
            group="train",
            device=device,
        )
        prediction_loop(
            split_dir,
            model=model,
            test_loader=valid_loader,
            group="validation",
            device=device,
        )
