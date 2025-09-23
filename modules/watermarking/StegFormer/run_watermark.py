# run_watermark.py
import os
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils import load_and_preprocess_image, get_watermark_from_db, load_weights, symbols_to_message_image, make_output_writable    
import numpy as np
from tqdm import tqdm

# Import project modules
# - StegFormer class and training-time utilities
from model import StegFormer
# reverse_message_image will be used for accuracy; we also add robust fallbacks here
from datasets import data_inference, inference
from utils import get_message_accuracy, compute_image_score
from torch.utils.data import DataLoader

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed); np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Models
# -------------------------
def build_models(args):
    # Mirror train.py variants
    if args.use_model == 'StegFormer-S':
        print("Using StegFormer-S model with custom depth and parameters")
        encoder = StegFormer(img_resolution=args.image_size, 
                            input_dim=(3 + args.secret_channels), cnn_emb_dim=8, output_dim=3,
                            drop_key=False, patch_size=2, window_size=8, output_act=args.output_act, depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
        decoder = StegFormer(img_resolution=args.image_size, 
                            input_dim=3, cnn_emb_dim=8, output_dim=args.secret_channels,
                            drop_key=False, patch_size=2, window_size=8, output_act=args.output_act, depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
    elif args.use_model == 'StegFormer-B':
        print("Using StegFormer-B model with default parameters")
        encoder = StegFormer(img_resolution=args.image_size, 
                            input_dim=(3 + args.secret_channels), cnn_emb_dim=16, output_dim=3)
        decoder = StegFormer(img_resolution=args.image_size, input_dim=3, cnn_emb_dim=16, output_dim=args.secret_channels)
    elif args.use_model == 'StegFormer-L':
        print("Using StegFormer-L model with larger parameters")
        encoder = StegFormer(img_resolution=args.image_size, 
                            input_dim=(3 + args.secret_channels), cnn_emb_dim=32, output_dim=3, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])
        decoder = StegFormer(img_resolution=args.image_size, input_dim=3, cnn_emb_dim=32, output_dim=args.secret_channels, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])
    else:
        raise ValueError(f"Unknown use_model: {args.use_model}")
    return encoder, decoder
# -------------------------
# Runner
# -------------------------

def main():
    parser = argparse.ArgumentParser("StegFormer watermark runner (DB-based)")
    parser.add_argument('--train_dataset', type=str, default='celeba_hq',
                        help='Name of the training dataset used for the model') 
    # Model/ckpt
    parser.add_argument('--dataset', type=str, choices=['facelab_london', 'CFD', 'ONOT', 'LFW'], default='facelab_london')
    parser.add_argument('--use_model', type=str, choices=['StegFormer-S','StegFormer-B','StegFormer-L'], default='StegFormer-B')
    parser.add_argument("--exp_name", type=str, default="1_1_clamp_StegFormer-B_baseline")
    parser.add_argument("--model_name", type=str, default="stegformer")
    parser.add_argument("--tag", type=str, default="last", help="which best_{tag}_*.pth.tar to load (e.g., psnr, ssim, acc, last)")
    # Data/shape
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--bpp", type=int, default=1, choices=[1,2,3,4,6,8])
    parser.add_argument("--norm_train", type=str, default="clamp", choices=["clamp","none"])
    parser.add_argument("--output_act", type=str, default=None)
    # Misc
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--set_name', type=str, choices=['test','templates'], default='test')
    parser.add_argument('--test', action='store_true', 
                        help='Run in test mode for single image watermark extraction verification.')
    parser.add_argument('--test_image_filename', type=str, 
                        help='Original Filename (processed not watermarked) of the image to test extraction from (e.g., "001_image.jpg"). Required with --test.')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # select DB name based on bpp
    if args.bpp == 1:
        db_name = "watermarks_BBP_1_65536_500.db"
        if args.set_name == 'templates':
            db_name = "watermarks_BBP_1_65536_500_templates.db" 
        secret_channels = 1
    elif args.bpp == 2:
        db_name = "watermarks_BBP_2_131072_13107.db"
        if args.set_name == 'templates':
            db_name = "watermarks_BBP_2_131072_13107_templates.db"
        secret_channels = 2
    elif args.bpp == 3:
        db_name = "watermarks_BBP_3_196608_39321.db"
        if args.set_name == 'templates':
            db_name = "watermarks_BBP_3_196608_39321_templates.db"
        secret_channels = 3
    elif args.bpp == 4: 
        db_name = ""
        secret_channels = 1
    elif args.bpp == 6:
        db_name = ""  
        secret_channels = 3         
    elif args.bpp == 8:
        db_name = ""
        secret_channels = 2
    else:
        raise ValueError(f"Unsupported bpp: {args.bpp}")
    
    # define paths
    input_dir = Path('/app/facial_data')
    model_path = Path(f'/app/runs/{args.exp_name}/{args.train_dataset}/model')
    base_path = Path(f'/app/output/{args.model_name}') / args.exp_name / "inference" / f"{args.dataset}"
    output_save_dir = base_path / 'watermarked_images'
    if args.set_name == 'templates':
        output_save_dir = base_path / 'watermarked_templates'
    output_save_dir.mkdir(parents=True, exist_ok=True)    
    args.secret_channels = secret_channels
    watermark_db_path = os.path.join(input_dir, args.dataset, 'processed', 'watermarks', db_name)

    # Load the model
    device = torch.device(args.device)
    encoder, decoder = build_models(args)
    encoder.to(device); decoder.to(device)

    # load the weights
    load_weights(
    encoder=encoder,
    decoder=decoder,
    save_path=model_path,
    tag=args.tag,  # decide what is better to restore: 'psnr', 'ssim' 'last' or 'acc'
    )

    encoder.eval(); decoder.eval()

    # Metrics
    cal_psnr = PeakSignalNoiseRatio().to(device)
    cal_ssim = StructuralSimilarityIndexMeasure().to(device)

    # Logic for test mode
    if args.test and args.set_name == 'test':
        inference_test_path = os.path.join(input_dir, args.dataset, 'processed', 'test')

        if args.test_image_filename:
            filenames = [args.test_image_filename]
            print(f"[TEST] for a single image: {args.test_image_filename}")
        else:
            inference_data_path = output_save_dir
            filenames = [p.name for p in Path(inference_data_path).glob('*')]
            filenames.sort()
            if not filenames:
                print(f"[TEST] Did not find any images in: {inference_data_path}")
                return
            print(f"[TEST] Batch mode: {len(filenames)} founded images in {inference_data_path}")

        acc_list = []
        psnr_list = []
        ssim_list = []
        processed = 0

        with torch.no_grad():
            for fname in tqdm(filenames, desc="Processing watermarked images..."):
                # Get the true watermark from DB
                # remember watermaked images are stored as png but the db uses jpg or original extension
                if args.dataset != 'ONOT':
                    true_message = get_watermark_from_db(watermark_db_path, fname.replace('png','jpg'))
                else:
                    # ONOT uses png images in test set
                    true_message = get_watermark_from_db(watermark_db_path, fname)
                    
                if true_message is None:
                    print(f"[WARNING] Did not find watermark for image: {fname}")
                    continue
                # convert to image watermark
                true_message_img = symbols_to_message_image(true_message, args.bpp, (args.image_size, args.image_size)) 

                # 2. Read the watermarked image
                wm_path = (output_save_dir / fname).with_suffix(".png")
                if args.dataset != 'ONOT':
                    img_path = Path(inference_test_path) / fname.replace('png','jpg') # original image path
                else:
                    # ONOT uses png images in test set
                    img_path = Path(inference_test_path) / fname

                if not img_path.exists():
                    print(f"[SKIP] Does not exist: {fname} in: {img_path}")
                    continue

                if not wm_path.exists():
                    print(f"[SKIP] Does not exist: {fname} in: {wm_path}")
                    continue

                # 3. Load and preprocess the cover image
                watermarked_image = load_and_preprocess_image(wm_path, args.image_size)
                image = load_and_preprocess_image(img_path, args.image_size)

                # 4. Decode the message
                image = image.unsqueeze(0).to(device)
                watermarked_image = watermarked_image.unsqueeze(0).to(device)
                decoded_message = decoder(watermarked_image)

                # 5) compute the metrics
                acc = get_message_accuracy(true_message_img, decoded_message, args.bpp)
                p, s =  compute_image_score(image, watermarked_image, cal_psnr, cal_ssim)

                #print(f"[TEST] {fname} - Extracted Accuracy: {accuracy:.4f}, PSNR: {p:.4f}, SSIM: {s:.4f}")  
                acc_list.append(acc)
                psnr_list.append(p)
                ssim_list.append(s)
                processed += 1

                if args.test_image_filename:
                    print(f"\n--- Single Image Watermark Extraction Test Results for {fname} ---")
                    print(f"Extracted Accuracy: {acc:.4f}")
                    print(f"PSNR: {p:.4f}")
                    print(f"SSIM: {s:.4f}")
                    
                    # optional info
                    #print(f"Decoded message shape: {decoded_message.shape}")
                    #print(f"Decoded message: {decoded_message}")
                    # The decoder output shape might be (batch_size, message_N, msg_L)
                    #print(f"True message shape: {true_message_reshaped_tensor.shape}")
                    #print(f"True message: {true_message_reshaped_tensor}")

                    return  
            
            if processed == 0:
                print("[TEST] No images were processed. Please check the filenames and paths.")
                return
            
            acc_offline_mean = np.array(acc_list, dtype=float).mean()
            acc_offline_std = np.array(acc_list, dtype=float).std() if processed>1 else 0.0
            psnr_offine_mean = np.array(psnr_list, dtype=float).mean()
            psnr_offine_std = np.array(psnr_list, dtype=float).std() if processed>1 else 0.0
            ssim_offine_mean = np.array(ssim_list, dtype=float).mean()
            ssim_offine_std = np.array(ssim_list, dtype=float).std() if processed>1 else 0.0

            print("\n--- Batch Watermark Extraction (from stored images) ---")
            print(f"Processed images : {processed}")
            print(f"Average Extracted Accuracy: {acc_offline_mean:.4f}")
            print(f"Accuracy STD: {acc_offline_std:.4f}")
            print(f"Average PSNR: {psnr_offine_mean:.4f}")
            print(f"PSNR STD: {psnr_offine_std:.4f}")
            print(f"Average SSIM: {ssim_offine_mean:.4f}")
            print(f"SSIM STD: {ssim_offine_std:.4f}")
            print("---------------------------------------------------")

            # open the results_summary.json file and strore this results there
            results_filepath = base_path / "results_summary.json"
            if results_filepath.exists():
                with open(results_filepath, "r") as f:
                    results_data = json.load(f)

                results_data["accuracy_offline"] = acc_offline_mean.item()
                results_data["accuracy_offline_std"] = acc_offline_std.item()
                results_data["psnr_offline"] = psnr_offine_mean.item()
                results_data["psnr_offline_std"] = psnr_offine_std.item()
                results_data["ssim_offline"] = ssim_offine_mean.item()
                results_data["ssim_offline_std"] = ssim_offine_std.item()
                
                with open(results_filepath, "w") as f:
                    json.dump(results_data, f, indent=2)
                print(f"Results updated in: {results_filepath}")
            else:
                print(f"Results file not found, skipping update: {results_filepath}")
            
        return # Exit main         
    
    # end logic for test mode
    else:
        # test data loader
        inference_data_path = os.path.join(input_dir, args.dataset, 'processed', 'test')
        if args.set_name == 'templates':
            inference_data_path = os.path.join(input_dir, args.dataset, 'processed', 'templates')

        # inference face dataloader
        inference_on_face_loader = DataLoader(
            data_inference(data_path=inference_data_path,
                        db_path=watermark_db_path,
                        image_size=(args.image_size, args.image_size),
                        dataset=args.dataset,
                        bpp=args.bpp,
                        ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # inference process
        test_acc, test_psnr, test_ssim, test_acc_std, test_psnr_std, test_ssim_std = inference(inference_on_face_loader, encoder, decoder, device, 
                                                   args.bpp, args.norm_train, output_save_dir, cal_psnr, cal_ssim)

        print(f"\n--- Inference results ---")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Accuracy STD: {test_acc_std:.4f}")
        print(f"PSNR: {test_psnr:.4f}")
        print(f"PSNR STD: {test_psnr_std:.4f}")
        print(f"SSIM: {test_ssim:.4f}")
        print(f"SSIM STD: {test_ssim_std:.4f}")
        print("-------------------------")

        if args.set_name == 'test':
            # Store the results in a JSON file 
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "model_name": args.model_name,  
                "training_dataset": args.train_dataset,
                "inference_dataset": args.dataset,
                "experiment_name": args.exp_name,
                "fine_tuned_icao": False,  # Assuming the model is fine-tuned
                "OFIQ_score": 0.0,  # Placeholder for OFIQ score
                "ICAO_compliance": False,  # Placeholder for ICAO compliance
                "bpp": args.bpp,
                "watermark_lenght": args.image_size * args.image_size * args.bpp, # valid until bpp=3
                "accuracy": test_acc.item(),
                "accuracy_std": test_acc_std.item(),
                "psnr": test_psnr.item(),
                "psnr_std": test_psnr_std.item(),
                "ssim": test_ssim.item(),
                "ssim_std": test_ssim_std.item(),
            } 

            # Generate a unique filename based on the current timestamp and dataset
            results_filepath = base_path / "results_summary.json"

            with open(results_filepath, "w") as f:
                json.dump(results_data, f, indent=2)
            
            print(f"Results saved to: {results_filepath}")
            make_output_writable("/app/output")


if __name__ == "__main__":
    main()
