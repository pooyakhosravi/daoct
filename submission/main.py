import os, pdb
import sys
from argparse import Namespace

import infer_test_monai
import train_test_monai_semi


def find_model(submission_dir):
    for name in [
        "unet_maestro2_semi.pth",
        "unet_maestro2.pth",
        "model.pth",
        "model.pt",
        os.path.join("checkpoints", "unet_maestro2_semi.pth"),
        os.path.join("checkpoints", "unet_maestro2.pth"),
    ]:
        model_path = os.path.join(submission_dir, name)
        if os.path.exists(model_path):
            return model_path
    return None


def find_training_root(input_dir):
    if os.path.exists(os.path.join(input_dir, "train")):
        return os.path.join(input_dir, "train")
    return input_dir


def find_inference_dir(input_dir):
    if os.path.exists(os.path.join(input_dir, "val", "images")):
        return os.path.join(input_dir, "val", "images")
    if os.path.exists(os.path.join(input_dir, "testing_data")):
        return os.path.join(input_dir, "testing_data")
    return input_dir


def main():
    # data that cannot be seen except by participant algorithm and is input to their algorithm
    input_dir = os.path.abspath(sys.argv[1]) # /app/input_data/
    # this is the predictions folder
    output_dir = os.path.abspath(sys.argv[2]) # /app/output/
    # this is the ingested program
    submission_dir = os.path.abspath(sys.argv[3]) # /app/ingested_program

    os.chdir(submission_dir)

    model_path = find_model(submission_dir)
    # pdb.set_trace()
    if model_path is None:
        train_args = Namespace(data_root=find_training_root(input_dir))
        print("Running train_test_monai_semi\n")
        train_test_monai_semi.main(train_args)
        model_path = os.path.join(submission_dir, "checkpoints", "unet_maestro2_semi.pth")

    infer_args = Namespace(
        input_dir=find_inference_dir(input_dir),
        output_dir=output_dir,
        model_path=model_path,
    )
    print("Running infer_test_monai\n")
    infer_test_monai.main(infer_args)


if __name__ == "__main__":
    main()
