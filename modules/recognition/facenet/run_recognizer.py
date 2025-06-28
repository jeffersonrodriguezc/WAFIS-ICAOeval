import os
from pathlib import Path
import argparse
from recognizer import FaceNetRecognizer
from collections import defaultdict
from itertools import combinations
import torch
import pandas as pd
import json
from tqdm import tqdm  # For progress bar
from PIL import Image, ImageOps
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser(description="Face recognition using FaceNet")
    parser.add_argument('--dataset', type=str, choices=['facelab_london', 'CFD', 'ONOT'], default='facelab_london')
    parser.add_argument('--watermarking_model', type=str, default='stegaformer')
    parser.add_argument('--experiment_name', type=str, default='1_1_255_w16_learn_im')
    parser.add_argument('--roi', type=str, default='fit', 
                        choices=['fit', 'crop'])
    parser.add_argument('--img_size', type=int, default=256,
                        help='Size of the image before processing, used for cropping or fitting')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'])
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    face_recognizer_service = FaceNetRecognizer(device=args.device)
    
    # so the path to the datasets is facial_data
    dataset_path = Path(f'facial_data/{args.dataset}/processed/test')

    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        print("Please check the dataset structure.")

        return
    
    if args.dataset == 'facelab_london':
        ext = '.jpg'
        image_paths = list(dataset_path.glob(f'**/*{ext}'))
    elif args.dataset == 'CFD':
        ext = '.jpg'
        image_paths = list(dataset_path.glob(f'**/*{ext}'))
    elif args.dataset == 'ONOT':
        ext = '.png'
        image_paths = list(dataset_path.glob(f'**/*{ext}'))
    
    # Group images by identity (assuming identity is the parent folder name)
    images_by_identity = defaultdict(list)
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        identity = filename.split('_')[0]
        images_by_identity[identity].append(img_path)
        
    embeddings_by_identity = defaultdict(list)
    
    print("Generating embeddings for all images before watermarking...")
    # Generate embeddings for all images
    total_identities = len(images_by_identity)
    # add tqdm progreess
    for i, (identity, path) in tqdm(enumerate(images_by_identity.items()), 
                                   total=total_identities, desc="Processing identities before ..."):
        
        # Apply some operations before getting the embedding if needed, depend on the watermarking model
        img = Image.open(path[0]).convert('RGB')
        if args.roi == 'fit':
            img_cover = ImageOps.fit(img, (args.img_size, args.img_size))
        elif args.roi == 'crop':
            width, height = img.size   # Get dimensions
            left = (width - args.im_size[0]) / 2
            top = (height - args.im_size[1]) / 2
            right = (width + args.im_size[0]) / 2
            bottom = (height + args.im_size[1]) / 2
            # Crop the center of the image
            img_cover = img.crop((left, top, right, bottom))

        # send the image to the recognizer and get the embedding
        embedding = face_recognizer_service.get_embedding(img_cover)
        embeddings_by_identity[identity].append(embedding)
        
    # Based on the fact that there is only one embedding per identity,
    # we can only calculate impostor distances, at this point   
    print("Calculating impostor distances...")
    impostor_distances = []
    identities = list(embeddings_by_identity.keys())
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            identity1 = identities[i]
            identity2 = identities[j]
            
            # To keep it manageable, let's just compare the first embedding of each identity
            # A more thorough approach would be to compare all embeddings from identity1 vs all from identity2
            if embeddings_by_identity[identity1] and embeddings_by_identity[identity2]:
                emb1 = embeddings_by_identity[identity1][0]
                emb2 = embeddings_by_identity[identity2][0]
                
                distance = face_recognizer_service.get_distance(emb1, emb2, metric=args.metric)
                impostor_distances.append(distance)
      
    output_dir = Path(f'output/recognition/{args.watermarking_model}/{args.experiment_name}/{args.dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to {output_dir}...")
    if impostor_distances:
        impostor_df = pd.DataFrame(impostor_distances, columns=['distance'])
        impostor_df.to_csv(output_dir / f'{args.metric}_impostor_{args.metric}_distances_ideal.csv', index=False)
    
    # calculate average distances
    # for genuine distances, we will assume that is 0.0
    # for impostor distances, we will save the average distance
    avg_impostor_distance_before_watermark = np.mean(impostor_distances)
    avg_genuine_distance_before_watermark = 0.0
        
    print(f"Average impostor distance before watermarking: {avg_impostor_distance_before_watermark}")
    print(f"Average genuine distance before watermarking: {avg_genuine_distance_before_watermark}")
    
    ## Second part: After watermarking, we will need to run the same process again
    watermarked_dataset_path = Path(f'output/watermarking/{args.watermarking_model}/{args.experiment_name}/inference/{args.dataset}/watermarked_images')
    if not watermarked_dataset_path.exists():
        print(f"Watermarked dataset path not found: {watermarked_dataset_path}")
    else:
        watermarked_image_paths = list(watermarked_dataset_path.glob('**/*.jpg'))
        watermarked_embeddings_by_identity = defaultdict(list)

        print("Generating embeddings for all watermarked images...")
        for path in tqdm(watermarked_image_paths, desc="Processing watermarked images"):
            filename = os.path.basename(path)
            identity = filename.split('_')[0]
            img = Image.open(path).convert('RGB')
            embedding = face_recognizer_service.get_embedding(img)
            if embedding is not None:
                watermarked_embeddings_by_identity[identity].append(embedding)

        print("Calculating genuine distances (original vs watermarked)...")
        genuine_distances_wm = []
        for identity, original_embs in embeddings_by_identity.items():
            if identity in watermarked_embeddings_by_identity and original_embs and watermarked_embeddings_by_identity[identity]:
                original_emb = original_embs[0]
                watermarked_emb = watermarked_embeddings_by_identity[identity][0]
                distance = face_recognizer_service.get_distance(original_emb, watermarked_emb, metric=args.metric)
                genuine_distances_wm.append(distance)

        print("Calculating impostor distances for watermarked images...")
        impostor_distances_wm = []
        watermarked_identities = list(watermarked_embeddings_by_identity.keys())
        for i in range(len(watermarked_identities)):
            for j in range(i + 1, len(watermarked_identities)):
                identity1 = watermarked_identities[i]
                identity2 = watermarked_identities[j]
                if watermarked_embeddings_by_identity[identity1] and watermarked_embeddings_by_identity[identity2]:
                    emb1 = watermarked_embeddings_by_identity[identity1][0]
                    emb2 = watermarked_embeddings_by_identity[identity2][0]
                    distance = face_recognizer_service.get_distance(emb1, emb2, metric=args.metric)
                    impostor_distances_wm.append(distance)

        if genuine_distances_wm:
            genuine_wm_df = pd.DataFrame(genuine_distances_wm, columns=['distance'])
            genuine_wm_df.to_csv(output_dir / f'{args.metric}_genuine_distances_watermarked.csv', index=False)

        if impostor_distances_wm:
            impostor_wm_df = pd.DataFrame(impostor_distances_wm, columns=['distance'])
            impostor_wm_df.to_csv(output_dir / f'{args.metric}_impostor_distances_watermarked.csv', index=False)  

        avg_genuine_distance_watermarked = np.mean(genuine_distances_wm) 
        avg_impostor_distance_watermarked = np.mean(impostor_distances_wm) 

        print(f"Average genuine distance after watermarking: {avg_genuine_distance_watermarked}")
        print(f"Average impostor distance after watermarking: {avg_impostor_distance_watermarked}")

        # open the json file with the results and add the results for average distances
        # the results file is at:
        watermarking_summery_path = Path(f'/app/output/watermarking/{args.watermarking_model}') / args.experiment_name / "inference" / f"{args.dataset}"
        results_filepath = watermarking_summery_path / f"results_summary.json"

        with open(results_filepath, "r") as f:
            results_data = json.load(f)
        # Add the average distances to the results data
        results_data['average_distances'] = {
            f'avg_dist_{args.metric}_genuine_before_watermark': avg_genuine_distance_before_watermark,
            f'avg_dist_{args.metric}_impostor_before_watermark': avg_impostor_distance_before_watermark,
            f'avg_dist_{args.metric}_genuine_after_watermark': avg_genuine_distance_watermarked,
            f'avg_dist_{args.metric}_impostor_after_watermark': avg_impostor_distance_watermarked
        }

        # Save the updated results data back to the file
        #new_path = results_filepath.with_name("results_summary_with_recognition.json")
        with open(results_filepath, "w") as f:
            json.dump(results_data, f, indent=2)

if __name__ == '__main__':
    main()