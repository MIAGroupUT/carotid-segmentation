import pandas as pd
import monai
import torch
from os import path


def prediction_loop(
    output_path: str,
    test_loader: monai.data.DataLoader,
    model: torch.nn.Module,
    group: str,
    device: str = "cpu",
):
    """Loop over test_loader with model to write predictions at output_path."""

    # Prepare writers
    metrics_columns = ["MSE", "L1", "SmoothL1"]
    columns = ["participant_id", "side", "label", "z"] + metrics_columns
    loss_fn = [getattr(torch.nn, f"{metric}Loss")() for metric in metrics_columns]

    # Prepare results
    prediction_df = pd.DataFrame(columns=columns)

    for test_batch in test_loader:
        with torch.no_grad():
            outputs = model(test_batch["image"].to(device)).transpose(1, 2)
        targets = test_batch["labels"].to(device)
        for i in range(len(test_batch["participant_id"])):
            row = [
                test_batch["participant_id"][i],
                test_batch["side"][i],
                test_batch["label"][i],
                int(test_batch["z"][i]),
            ]
            row += [fn(outputs[i], targets[i]).item() for fn in loss_fn]
            row_df = pd.DataFrame([row], columns=columns)
            prediction_df = pd.concat([prediction_df, row_df])

    prediction_df.set_index(["participant_id", "side", "label", "z"], drop=True)
    prediction_df.sort_index(inplace=True)
    prediction_df.to_csv(
        path.join(output_path, f"group-{group}_prediction.tsv"),
        sep="\t",
    )
    metrics_df = prediction_df[metrics_columns].mean().to_frame().transpose()
    metrics_df.to_csv(
        path.join(output_path, f"group-{group}_metrics.tsv"),
        sep="\t",
        index=False,
    )