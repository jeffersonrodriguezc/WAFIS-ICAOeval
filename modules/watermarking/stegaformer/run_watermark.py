"""
Author: Jefferson Rodríguez & Gemini - University of Cagliari - 2025-06-25
"""
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import torch
from utils import get_message_accuracy, MIMData_inference
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from stegaformer import Encoder, Decoder

def load_weights(encoder: Encoder, decoder: Decoder, save_path: Path, tag: str = 'acc') -> None:
    """
    Load the model weights from the specified path.
    
    Args:
        encoder (Encoder): The encoder model.
        decoder (Decoder): The decoder model.
        save_path (Path): Path to the directory containing the model checkpoints.
        tag (str): Tag to select the best checkpoint based on a specific metric.
    """
    resume_encoder_path = os.path.join(save_path, f'best_{tag}_encoder.pth.tar')
    resume_decoder_path = os.path.join(save_path, f'best_{tag}_decoder.pth.tar')

    if os.path.exists(resume_encoder_path) and os.path.exists(resume_decoder_path):
        print(f"🔄 Restoring checkpoint from '{tag.upper()}'...")

        encoder_ckpt = torch.load(resume_encoder_path)
        decoder_ckpt = torch.load(resume_decoder_path)

        encoder.load_state_dict(encoder_ckpt['state_dict'])
        decoder.load_state_dict(decoder_ckpt['state_dict'])    
    
    print(f"Loaded weights from {save_path} with tag '{tag}'")

def ssim(img1, img2, cal_ssim):
    s = cal_ssim(img1, img2)
    return s.cpu().detach().numpy()

def psnr(cover, generated, cal_psnr):
    psnr = cal_psnr(generated, cover)
    return psnr.cpu().detach().numpy()

def compute_image_score(cover, generated, cal_psnr, cal_ssim):
    
    p = psnr(cover, generated, cal_psnr)
    s = ssim(cover, generated, cal_ssim)
    
    return p, s

def inference(inference_loader, encoder, decoder, message_N, device_id, msg_range, output_save_dir, cal_psnr, cal_ssim):
    encoder.eval()
    decoder.eval()

    os.makedirs(output_save_dir, exist_ok=True)
    acc = []
    psnr = []
    ssim = []

    with torch.no_grad():
        #for i in range(inference_loader.__len__()):
        for i, (images, messages, filenames) in enumerate(inference_loader):
            messages = messages.cuda(device_id)
            images = images.cuda(device_id)
            
            enco_images = encoder(images, messages)
            enco_images_clamped = torch.clamp(enco_images,0,255).cpu()
            
            deco_messages = decoder(enco_images)
            #please sigmoid first for the decoded message
            if msg_range == 1:
                deco_messages = torch.sigmoid(deco_messages)

            ac = get_message_accuracy(messages, deco_messages, message_N)
            p, s = compute_image_score(images, enco_images, cal_psnr, cal_ssim)
            
            acc.append(ac)
            psnr.append(p)
            ssim.append(s)

            # store the watermarked images
            for j in range(enco_images_clamped.shape[0]):
                current_filename = filenames[j]
                save_path_full = output_save_dir / current_filename
                save_image(enco_images_clamped[j], save_path_full, normalize=True, range=(0, 1)) 
    
    encoder.train()
    decoder.train()
    
    return sum(acc)/len(acc), sum(psnr)/len(psnr), sum(ssim)/len(ssim)

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed watermark messages into images")
    parser.add_argument('--input_dir', type=Path, default=Path('/app/facial_data'))
    parser.add_argument('--dataset', type=str, choices=['facelab_london', 'CFD', 'ONOT'], default='facelab_london')
    parser.add_argument('--db_name', type=str, default='watermarks_BBP_1_65536_500_facelab_london.db')
    parser.add_argument('--output_dir', type=Path, default=Path('/app/output/stegaformer/1_1_255_w16_learn_im/'),
                        help='Directory to save the output watermarked images')
    parser.add_argument('--model', type=Path, default=Path('/app/runs/1_1_255_w16_learn_im/celeba_hq/model'),
                        help='Directory containing model checkpoints')
    parser.add_argument('--bpp', type=int, default=1)
    parser.add_argument('--query', type=str, default='im')
    parser.add_argument('--msg_scale', type=int, default=1)
    parser.add_argument('--enable_img_norm', action='store_true', help='Image Normalize to 0 ~ 1 if True')
    parser.add_argument('--msg_range', type=int, default=1)
    parser.add_argument('--msg_pose', type=str, default='learn')
    parser.add_argument('--win_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # Setting parameters and hyperparameters used in training
    message_L = 16*args.bpp
    message_N = 64*64
    im_size = args.img_size
    scale = args.msg_scale
    encoder_eb_dim = int(message_L*scale)
    decoder_eb_dim = int(message_L*scale)
    msg_range = args.msg_range
    query_type = args.query
    device_id = args.device
    output_save_dir = args.output_dir / "inference" / f"{args.dataset}"
    output_save_dir.mkdir(parents=True, exist_ok=True)

    # create the encoder and decoder models
    encoder = Encoder(msg_L=message_L, embed_dim=encoder_eb_dim, 
                      Q=query_type, win_size=args.win_size, msg_pose=args.msg_pose)
    decoder = Decoder(img_size=im_size, msg_L=message_L, embed_dim=decoder_eb_dim,
                       win_size=args.win_size, msg_pose=args.msg_pose)
    
    encoder.cuda(device_id)
    decoder.cuda(device_id)

    # metrics
    cal_psnr = PeakSignalNoiseRatio().cuda(device_id)
    cal_ssim = StructuralSimilarityIndexMeasure().cuda(device_id)

    # load the weights
    load_weights(
    encoder=encoder,
    decoder=decoder,
    save_path=Path(args.model),
    tag='acc'  # decide what is better to restore: 'psnr', 'ssim' or 'acc'
    )

    # Test dataloader
    inference_data_path = os.path.join(args.input_dir,args.dataset, 'processed', 'test')
    watermark_db_path = os.path.join(args.input_dir,args.dataset, 'processed','watermarks', args.db_name)
    test_dataset = MIMData_inference(data_path=inference_data_path, db_path=watermark_db_path, num_message=message_N, message_size=message_L, 
                                     image_size=(im_size, im_size),dataset=args.dataset, msg_r=msg_range)
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_message_acc, test_psnr, test_ssim = inference(test_dataloader, encoder, decoder, message_N, device_id, msg_range,
                                                       output_save_dir=output_save_dir, cal_psnr=cal_psnr, cal_ssim=cal_ssim)

    print(f"\n--- Inference results ---")
    print(f"Accuracy: {test_message_acc:.4f}")
    print(f"PSNR: {test_psnr:.4f}")
    print(f"SSIM: {test_ssim:.4f}")


    # Store the results in a JSON file for Quarto documentation
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": str(args.output_dir).split('/')[-2],  # Extract the model name from the output directory
        "training_dataset": str(args.model).split('/')[-2],  # Extract the training dataset name from the model path
        "inference_dataset": args.dataset,
        "fine_tuned_icao": False,  # Assuming the model is fine-tuned
        "OFIQ_score": 0.0,  # Placeholder for OFIQ score
        "ICAO_compliance": False,  # Placeholder for ICAO compliance
        "bpp": args.bpp,
        "watermark_lenght": message_L*message_N,
        "accuracy": test_message_acc,
        "psnr": test_psnr,
        "ssim": test_ssim,
    }

    # Definir la carpeta de destino para los archivos de resultados de Quarto
    quarto_results_dir = Path("/app/docs") / "_data" / "watermarking_inference_runs"
    quarto_results_dir.mkdir(parents=True, exist_ok=True) 

    # Generate a unique filename based on the current timestamp and dataset
    results_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_filepath = quarto_results_dir / results_filename

    with open(results_filepath, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to: {results_filepath}")



if __name__ == '__main__':
    main()