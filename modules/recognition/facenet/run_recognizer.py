import os
from pathlib import Path
import argparse
from recognizer import FaceNetRecognizer
from collections import defaultdict
import pandas as pd
import json
from tqdm import tqdm  # For progress bar
from PIL import Image, ImageOps
import numpy as np

def load_and_preprocess_image(image_path, img_size, img_norm=False):
    # Apply some operations before getting the embedding if needed, depend on the watermarking model
    img = Image.open(image_path).convert('RGB')
    img_cover = ImageOps.fit(img, (img_size, img_size))
    
    return img_cover

def get_identity_from_filename(filename):
    return os.path.splitext(filename.split('_')[0])[0]

def get_embeddings(folder_path, image_files, img_size, face_recognizer_service):
    """Generate embeddings for all images in the folder."""
    embeddings_by_identity = defaultdict(list)
    for img_path in tqdm(image_files, desc=f"Generating embeddings for {folder_path.name}"):
        identity = get_identity_from_filename(img_path.name)
        img = load_and_preprocess_image(img_path, img_size)
        #print(identity, img_path, img.shape)
        embedding = face_recognizer_service.get_embedding(img)
        if embedding is not None:
            embeddings_by_identity[identity].append(embedding)
    return embeddings_by_identity

def calculate_metrics(genuine_distances, impostor_distances):
    """Compute EER, FAR, and FRR based on genuine and impostor distances."""
    distances = np.concatenate([genuine_distances, impostor_distances])
    labels = np.concatenate([np.ones_like(genuine_distances), np.zeros_like(impostor_distances)])

    # Sort distances and calculate metrics for each threshold
    sorted_distances = np.sort(distances)
    far_list, frr_list = [], []

    for threshold in sorted_distances:
        predictions = distances <= threshold
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        far_list.append(far)
        frr_list.append(frr)

    eer_index = np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2

    # Calculate AUC using the trapezoidal rule
    # The False Positive Rate (FPR) is the same as FAR
    # The True Positive Rate (TPR) is 1 - FRR
    tpr_list = 1 - np.array(frr_list)
    fpr_list = np.array(far_list)
    
    # Sort by FPR for AUC calculation
    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = fpr_list[sorted_indices]
    tpr_sorted = tpr_list[sorted_indices]
    
    auc = np.trapz(tpr_sorted, fpr_sorted)

    return {'EER': round(eer*100,3), 
            'FAR_at_EER': round(far_list[eer_index]*100,3), 
            'FRR_at_EER': round(frr_list[eer_index]*100,3),
            'AUC': round(auc, 3)}

def main() -> None:
    parser = argparse.ArgumentParser(description="Face recognition using FaceNet")
    parser.add_argument('--dataset', type=str, choices=['facelab_london', 'CFD', 'ONOT'], default='CFD')
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
    
    # Initialize the FaceNet recognizer
    face_recognizer_service = FaceNetRecognizer(device=args.device)
    
    # so the path to the datasets is facial_data
    test_path = Path(f'facial_data/{args.dataset}/processed/test')
    templates_path = Path(f'facial_data/{args.dataset}/processed/templates')
    watermarked_path = Path(f'output/watermarking/{args.watermarking_model}/{args.experiment_name}/inference/{args.dataset}/watermarked_images')

    if not test_path.exists():
        print(f"Dataset path not found: {test_path}")
        print("Please check the dataset structure.")

        return
    
    if not templates_path.exists():
        print(f"Template path not found: {templates_path}")
        print("Please check the dataset structure.")

        return
    
    if args.dataset == 'facelab_london':
        ext = '.jpg'
        image_paths = list(test_path.glob(f'**/*{ext}'))
        template_paths = list(templates_path.glob(f'**/*{ext}'))

    elif args.dataset == 'CFD':
        ext = '.jpg'
        image_paths = list(test_path.glob(f'**/*{ext}'))
        template_paths = list(templates_path.glob(f'**/*{ext}'))

    elif args.dataset == 'ONOT':
        ext = '.png'
        image_paths = list(test_path.glob(f'**/*{ext}'))
        template_paths = list(templates_path.glob(f'**/*{ext}'))
    
    # always the watermarked images are png
    watermarked_paths = list(watermarked_path.glob(f'**/*.png'))
    
    # get the embeddings
    templates_embs = get_embeddings(templates_path, template_paths, args.img_size, face_recognizer_service)
    template_identities = set(templates_embs.keys())

    # Filter image_paths and watermarked_paths to only include identities present in templates
    filtered_image_paths = [p for p in image_paths if get_identity_from_filename(p.name) in template_identities]
    filtered_watermarked_paths = [p for p in watermarked_paths if get_identity_from_filename(p.name) in template_identities]

    tests_embs = get_embeddings(test_path, filtered_image_paths, args.img_size, face_recognizer_service)
    watermarked_embs = get_embeddings(watermarked_path, filtered_watermarked_paths, args.img_size, face_recognizer_service)
    
    print(f"Number of identities in templates: {len(templates_embs)}")
    print(f"Example identities in templates: {list(templates_embs.keys())[:5]}")
    print(f"Dimension of an embedding: {next(iter(templates_embs.values()))[0].shape if templates_embs else 'N/A'} ")
    print(f"Number of identities in test images: {len(tests_embs)}") 
    print(f"Example identities in test images: {list(tests_embs.keys())[:5]}")   
    print(f"Number of identities in watermarked images: {len(watermarked_embs)}")
    print(f"Example identities in watermarked images: {list(watermarked_embs.keys())[:5]}") 

    # Face recognition evaluation
    genuine_distances_baseline = []
    impostor_distances_baseline = []
    genuine_distances_wm = []
    impostor_distances_wm = []
    genuine_variation_distances = []
    impostor_variation_distances = []
    genuine_raw_distances = []
    impostor_raw_distances  = []

    identities = list(templates_embs.keys())
    for i, identity_a in enumerate(identities):
        for j, identity_b in enumerate(identities):
            # Genuine pairs (same person)
            if identity_a == identity_b:
                #print(f"Calculating genuine distance for identity: {identity_a}")
                dist = face_recognizer_service.get_distance(templates_embs[identity_a][0], tests_embs[identity_b][0], metric=args.metric)
                dist_wm = face_recognizer_service.get_distance(templates_embs[identity_a][0], watermarked_embs[identity_b][0], metric=args.metric)
                raw_dist = face_recognizer_service.get_distance(tests_embs[identity_b][0], watermarked_embs[identity_b][0], metric=args.metric)
                variation_dist = abs(dist - dist_wm)
                
                #print(f"Distance: {dist}")
                #print(f"Distance WM: {dist_wm}")
                #print(f"Raw distance between original and watermarked: {raw_dist}")
                #print(f"Variation in distance due to watermarking: {variation_dist}")

                genuine_distances_baseline.append(dist)
                genuine_distances_wm.append(dist_wm)
                genuine_raw_distances.append(raw_dist)
                genuine_variation_distances.append(variation_dist)
            
            # Impostor pairs (different persons)
            elif i < j:
                if templates_embs[identity_a] and tests_embs[identity_b]:
                    #print(f"Calculating impostor distance for identities: {identity_a} and {identity_b}")
                    dist = face_recognizer_service.get_distance(templates_embs[identity_a][0], tests_embs[identity_b][0], metric=args.metric)
                    dist_wm = face_recognizer_service.get_distance(templates_embs[identity_a][0], watermarked_embs[identity_b][0], metric=args.metric)
                    raw_dist = face_recognizer_service.get_distance(tests_embs[identity_a][0], watermarked_embs[identity_b][0], metric=args.metric)
                    variation_dist = abs(dist - dist_wm)

                    #print(f"Distance: {dist}")
                    #print(f"Distance WM: {dist_wm}")
                    #print(f"Raw distance between original and watermarked: {raw_dist}")
                    #print(f"Variation in distance due to watermarking: {variation_dist}")

                    impostor_distances_baseline.append(dist)
                    impostor_distances_wm.append(dist_wm)
                    impostor_raw_distances.append(raw_dist)
                    impostor_variation_distances.append(variation_dist)
    
    # Compute metrics
    metrics_baseline = calculate_metrics(genuine_distances_baseline, impostor_distances_baseline)
    print("Metrics before watermarking:")
    print(metrics_baseline)

    metrics_baseline_wm = calculate_metrics(genuine_distances_wm, impostor_distances_wm)
    print("Metrics after watermarking:")
    print(metrics_baseline_wm)

    # Compute mean and std of the variation distances
    if genuine_variation_distances:
        mean_variation_genuine = np.round(np.mean(genuine_variation_distances) * 100, 3)
        std_variation_genuine = np.round(np.std(genuine_variation_distances) * 100,3)
        print(f"Mean variation in genuine distances due to watermarking: {mean_variation_genuine}")
        print(f"Std deviation of variation in genuine distances due to watermarking: {std_variation_genuine}")

    if impostor_variation_distances:
        mean_variation_impostor = np.round(np.mean(impostor_variation_distances),3)     
        std_variation_impostor = np.round(np.std(impostor_variation_distances),3)
        print(f"Mean variation in impostor distances due to watermarking: {mean_variation_impostor}")
        print(f"Std deviation of variation in impostor distances due to watermarking: {std_variation_impostor}")

    # Compute average and std raw distances
    if genuine_raw_distances:
        avg_genuine_raw_distance = np.round(np.mean(genuine_raw_distances),3)
        std_genuine_raw_distance = np.round(np.std(genuine_raw_distances),3)
        print(f"Average raw distance for genuine pairs: {avg_genuine_raw_distance}")
        print(f"Std deviation of raw distance for genuine pairs: {std_genuine_raw_distance}")
    
    if impostor_raw_distances:
        avg_impostor_raw_distance = np.round(np.mean(impostor_raw_distances),3)
        std_impostor_raw_distance = np.round(np.std(impostor_raw_distances),3)
        print(f"Average raw distance for impostor pairs : {avg_impostor_raw_distance}")
        print(f"Std deviation of raw distance for impostor pairs: {std_impostor_raw_distance}")

    # Store the distances in csv files
    output_dir = Path(f'output/recognition/{args.watermarking_model}/{args.experiment_name}/{args.dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to {output_dir}...")
    if genuine_distances_baseline:
        genuine_df = pd.DataFrame(genuine_distances_baseline, columns=['distance'])
        genuine_df.to_csv(output_dir / f'{args.metric}_genuine_distances_baseline.csv', index=False)
        
    if impostor_distances_baseline:
        impostor_df = pd.DataFrame(impostor_distances_baseline, columns=['distance'])
        impostor_df.to_csv(output_dir / f'{args.metric}_impostor_distances_baseline.csv', index=False)
    
    if genuine_distances_wm:
        genuine_wm_df = pd.DataFrame(genuine_distances_wm, columns=['distance'])
        genuine_wm_df.to_csv(output_dir / f'{args.metric}_genuine_distances_watermarked.csv', index=False)

    if impostor_distances_wm:
        impostor_wm_df = pd.DataFrame(impostor_distances_wm, columns=['distance'])
        impostor_wm_df.to_csv(output_dir / f'{args.metric}_impostor_distances_watermarked.csv', index=False)
    

    # Open the json file with the results and add the results for average distances
    # the results file is at:
    watermarking_summery_path = Path(f'/app/output/watermarking/{args.watermarking_model}') / args.experiment_name / "inference" / f"{args.dataset}"
    results_filepath = watermarking_summery_path / f"results_summary.json"

    with open(results_filepath, "r") as f:
        results_data = json.load(f)
    # Add the average distances to the results data
    results_data['average_distances'] = {
        f'facenet_avg_var_dist_{args.metric}_genuine_due_watermark': mean_variation_genuine if genuine_variation_distances else None,
        f'facenet_avg_var_dist_{args.metric}_impostor_due_watermark': mean_variation_impostor if impostor_variation_distances else None,
        f'facenet_avg_raw_dist_{args.metric}_genuine_raw_vs_watermark': avg_genuine_raw_distance if genuine_raw_distances else None,
        f'facenet_avg_raw_dist_{args.metric}_impostor_raw_vs_watermark': avg_impostor_raw_distance if impostor_raw_distances else None
    }

    # Add the std of distances to the results data
    results_data['std_distances'] = {
        f'facenet_std_var_dist_{args.metric}_genuine_due_watermark': std_variation_genuine if genuine_variation_distances else None,
        f'facenet_std_var_dist_{args.metric}_impostor_due_watermark': std_variation_impostor if impostor_variation_distances else None,
        f'facenet_std_raw_dist_{args.metric}_genuine_raw_vs_watermark': std_genuine_raw_distance if genuine_raw_distances else None,
        f'facenet_std_raw_dist_{args.metric}_impostor_raw_vs_watermark': std_impostor_raw_distance if impostor_raw_distances else None
    }

    # Add the recognition metrics to the results data
    results_data['recognition_metrics'] = {
        f'facenet_EER_baseline_{args.metric}': metrics_baseline['EER'],
        f'facenet_EER_watermarked_{args.metric}': metrics_baseline_wm['EER'],
        f'facenet_FAR_at_EER_baseline_{args.metric}': metrics_baseline['FAR_at_EER'],
        f'facenet_FAR_at_EER_watermarked_{args.metric}': metrics_baseline_wm['FAR_at_EER'],
        f'facenet_FRR_at_EER_baseline_{args.metric}': metrics_baseline['FRR_at_EER'],
        f'facenet_FRR_at_EER_watermarked_{args.metric}': metrics_baseline_wm['FRR_at_EER'],
        f'facenet_AUC_baseline_{args.metric}': metrics_baseline['AUC'],
        f'facenet_AUC_watermarked_{args.metric}': metrics_baseline_wm['AUC']
    }

    # Save the updated results data back to the file
    # new_path = results_filepath.with_name("results_summary_with_recognition.json")
    with open(results_filepath, "w") as f:
        json.dump(results_data, f, indent=2)

if __name__ == '__main__':
    main()