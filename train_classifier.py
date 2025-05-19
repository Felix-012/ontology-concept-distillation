import os
import argparse
import time
import numpy as np
from downstream.chexnet_trainer import ChexnetTrainer
import torch

# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChexNet Training and Testing CLI")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: 'train' or 'test'")

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", required=True, help="Path to the data directory")
    train_parser.add_argument("--data_dir_train", required=False, help="Path to the train data directory (if its different to val, test)")
    train_parser.add_argument("--train_file", required=True, help="Path to the training set file")
    train_parser.add_argument("--test_file", required=False, help="Path to the test set file")
    train_parser.add_argument("--centercrop_only", action="store_true", help="Use only centercropping instead of randomresizedcrop at training time")
    train_parser.add_argument("--outfile", default="auroc_results.txt", required=True, help="Name of the test.txt file")
    train_parser.add_argument("--arch", default="DENSE-NET-121", choices=["DENSE-NET-121", "DENSE-NET-169", "DENSE-NET-201"], help="Model architecture")
    train_parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pre-trained weights")
    train_parser.add_argument("--use_cache", action="store_true", help="Store all train images in cache")
    train_parser.add_argument("--num_classes", type=int, default=12, help="Number of output classes")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--resize", type=int, default=256, help="Resize image dimension")
    train_parser.add_argument("--crop", type=int, default=224, help="Crop image dimension")
    train_parser.add_argument("--save_path", default="models", help="Path to save the trained model")
    train_parser.add_argument("--debug", action="store_true", help="Debug mode: overfit on 4 batches of training data")

    # Test parser
    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument("--data_dir", required=True, help="Path to the data directory")
    test_parser.add_argument("--test_file", required=True, help="Path to the test set file")
    test_parser.add_argument("--model_path", required=True, help="Path to the trained model file")
    test_parser.add_argument("--outfile", default="auroc_results.txt", required=True, help="Name of the test.txt file")
    test_parser.add_argument("--arch", default="DENSE-NET-121", choices=["DENSE-NET-121", "DENSE-NET-169", "DENSE-NET-201"], help="Model architecture")
    test_parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pre-trained weights")
    test_parser.add_argument("--num_classes", type=int, default=12, help="Number of output classes")
    test_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    test_parser.add_argument("--resize", type=int, default=256, help="Resize image dimension")
    test_parser.add_argument("--crop", type=int, default=224, help="Crop image dimension")

    args = parser.parse_args()

    if args.mode == "train":
        runTrain(args)
    elif args.mode == "test":
        runTest(args)
    else:
        parser.print_help()

# --------------------------------------------------------------------------------

def runTrain(args):
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = f"{timestampDate}-{timestampTime}"

    model_save_path = os.path.join(args.save_path, f"m-{timestampLaunch}.pth.tar")
    os.makedirs(args.save_path, exist_ok=True)

    if args.data_dir_train is not None: 
        print(f"Using different training data basedir {args.data_dir_train}")
    else: 
        args.data_dir_train = args.data_dir

    print(f"Training NN architecture = {args.arch}")

    ChexnetTrainer.train(
        args.data_dir,
        args.data_dir_train,
        args.train_file,
        args.arch,
        args.pretrained,
        args.num_classes,
        args.batch_size,
        args.epochs,
        args.resize,
        args.crop,
        timestampLaunch,
        None,
        model_save_path,
        args.centercrop_only,
        args.use_cache,
        debug=args.debug
    )

    print("Training complete.")
    if args.test_file is not None: 
        print("Testing the trained model...")
        ChexnetTrainer.test(
            args.data_dir,
            args.test_file,  # Assuming test file is related
            model_save_path,
            args.arch,
            args.num_classes,
            args.pretrained,
            args.batch_size,
            args.resize,
            args.crop,
            timestampLaunch,
            args.outfile
        )

# --------------------------------------------------------------------------------

def runTest(args):
    print(f"Testing the model: {args.model_path}")
    ChexnetTrainer.test(
        args.data_dir,
        args.test_file,
        args.model_path,
        args.arch,
        args.num_classes,
        args.pretrained,
        args.batch_size,
        args.resize,
        args.crop,
        "",
        args.outfile,
    )
    print("Testing complete.")

# --------------------------------------------------------------------------------

if __name__ == "__main__":
    main()