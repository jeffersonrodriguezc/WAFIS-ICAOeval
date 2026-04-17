import os
from pathlib import Path
import argparse
from recognizer import ArcFaceRecognizer
from collections import defaultdict
import pandas as pd
import json
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import torch


def load_and_preprocess_image(image_path, img_size, img_norm=False, image_format='png'):
    """Load and do a first-pass resize — mirrors FaceNet service exactly."""
    if image_format == 'png':
        img = Image.open(image_path).convert('RGB')
        img_cover = ImageOps.fit(img, (img_size, img_size))
    elif image_format == 'npy':
        img_cover = np.load(image_path).astype(np.float32)
        # if the image is in the range [0,1], convert to [0,255] float
        if img_cover.max() <= 1.0:
            img_cover = (img_cover * 255)
    return img_cover

def get_identity_from_filename(filename):
    return os.path.splitext(filename.split('_')[0])[0]

def get_embeddings_old(folder_path, image_files, img_size, face_recognizer_service,
                   image_format='png', debug_img=False):
    """Generate embeddings for all images in the folder."""
    embeddings_by_identity = defaultdict(list)
    for img_path in tqdm(image_files, desc=f"Generating embeddings for {folder_path.name}"):
        identity = get_identity_from_filename(img_path.name)
        img = load_and_preprocess_image(img_path, img_size, image_format=image_format)
        embedding = face_recognizer_service.get_embedding(img, debug_img=debug_img)
        
        if embedding is not None:
            embeddings_by_identity[identity].append(embedding)
    return embeddings_by_identity

def get_embeddings(folder_path, image_files, img_size, face_recognizer_service,
                   image_format='png', debug_img=False, precomputed_boxes=None):
    """
    Generate embeddings for all images in the folder.
    
    Args:
        precomputed_boxes: dict mapping identity -> bounding box (np.ndarray).
                           When provided, this box is reused instead of running
                           MTCNN detection, avoiding uint8 quantization on
                           watermarked images.
    """
    embeddings_by_identity = defaultdict(list)
    for img_path in tqdm(image_files, desc=f"Generating embeddings for {folder_path.name}"):
        identity = get_identity_from_filename(img_path.name)
        img = load_and_preprocess_image(img_path, img_size, image_format=image_format)
        #print(f"min and max pixel values for image {img_path.name}: {img.min()} - {img.max()}")
        
        box = precomputed_boxes.get(identity) if precomputed_boxes else None
        embedding = face_recognizer_service.get_embedding(img, debug_img=debug_img,
                                                          precomputed_box=box, origin='watermarked' if box is not None else 'original')
        
        if embedding is not None:
            embeddings_by_identity[identity].append(embedding)
        else:
            raise ValueError(f"Failed to get embedding for image: {img_path} with identity: {identity}")
        
    return embeddings_by_identity

def get_embeddings_and_boxes(folder_path, image_files, img_size, face_recognizer_service,
                              image_format='png', debug_img=False):
    """
    Generate embeddings AND store the detected bounding boxes for later reuse.
    Only meaningful when use_mtcnn=True.
    
    Returns:
        embeddings_by_identity: dict  identity -> [embedding, ...]
        boxes_by_identity:      dict  identity -> np.ndarray (bounding box)
    """
    embeddings_by_identity = defaultdict(list)
    boxes_by_identity = {}
    for img_path in tqdm(image_files, desc=f"Generating embeddings+boxes for {folder_path.name}"):
        identity = get_identity_from_filename(img_path.name)
        img = load_and_preprocess_image(img_path, img_size, image_format=image_format)
        
        embedding, box = face_recognizer_service.get_embedding_and_box(img, debug_img=debug_img, origin='original')
        
        if embedding is not None:
            embeddings_by_identity[identity].append(embedding)
            if box is not None:
                boxes_by_identity[identity] = box
            else:
                raise ValueError(f"Failed to get bounding box for image: {img_path} with identity: {identity}")
        else:
            raise ValueError(f"Failed to get embedding for image: {img_path} with identity: {identity}")
        
    return embeddings_by_identity, boxes_by_identity

def calculate_tar_at_far(far_list, frr_list, target_far=0.001): # Note: 0.01% = 0.0001, 0.1% = 0.001, 1% = 0.01
    """
    Calculates the TAR (1 - FRR) at a specific FAR threshold.
    """
    far_array = np.array(far_list)
    frr_array = np.array(frr_list)
    
    # Find the indices where the FAR is less than or equal to the target.
    indices = np.where(far_array <= target_far)[0]
    
    if len(indices) == 0:
        # If no threshold meets the requirement, it cannot be reported.
        # This can happen if your system is not good enough or if you have too little data.
        print(f"Warning: No threshold found for a FAR <= {target_far*100}%.")
        return None

    # Of all the thresholds that meet the requirement, choose the one with the highest FAR
    # (and therefore the lowest FRR), which corresponds to the last valid index.
    best_index = indices[-1] # or max(indices)
    
    tar = 1 - frr_array[best_index]
    
    return {'TAR_at_FAR': round(tar * 100, 3),
            'Actual_FAR': round(far_array[best_index] * 100, 5)}

def calculate_metrics(genuine_distances, impostor_distances, num_thresholds=None, target_far=0.001):
    """Compute EER, FAR, and FRR based on genuine and impostor distances."""
    distances = np.concatenate([genuine_distances, impostor_distances])
    labels = np.concatenate([np.ones_like(genuine_distances), np.zeros_like(impostor_distances)])

    # Sort distances and calculate metrics for each threshold
    
    if num_thresholds is None:
        thresholds_to_check = np.sort(distances)
    else:
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        thresholds_to_check = np.linspace(min_dist, max_dist, num_thresholds)

    far_list, frr_list = [], []

    for threshold in tqdm(thresholds_to_check, desc="Thresholding for metrics calculation"):
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

    TAR_metric = calculate_tar_at_far(far_list, frr_list, target_far=target_far)

    return {'EER': round(eer*100,3), 
            'FAR_at_EER': round(far_list[eer_index]*100,3), 
            'FRR_at_EER': round(frr_list[eer_index]*100,3),
            'AUC': round(auc, 3),
            'TAR_at_FAR': TAR_metric['TAR_at_FAR'],
            'Actual_FAR': TAR_metric['Actual_FAR']}


def main() -> None:
    parser = argparse.ArgumentParser(description="Face recognition using arcface backbone")
    parser.add_argument('--dataset', type=str, choices=['facelab_london', 'CFD', 'ONOT', 'LFW', 'SCface', 'ONOT_set1'], default='CFD')
    parser.add_argument('--train_dataset', type=str, choices=['celeba_hq', 'coco'], default='celeba_hq')
    parser.add_argument('--watermarking_model', type=str, default='stegaformer')
    parser.add_argument('--experiment_name', type=str, default='1_1_255_w16_learn_im')
    parser.add_argument('--roi', type=str, default='fit', 
                        choices=['fit', 'crop'])
    parser.add_argument('--img_size', type=int, default=256,
                        help='Size of the image before processing, used for cropping or fitting')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'])
    parser.add_argument('--thresholds', type=int, default=20000,
                        help='Number of thresholds to evaluate for metrics calculation. If not set, all unique distances are used.')
    parser.add_argument('--format_evaluation', type=str, default='offline', 
                        choices=['offline', 'online'],
                        help='Format of the evaluation, offline (all images in png format) or online (npy arrays stored during watermarking)')
    parser.add_argument('--backbone', type=str, default='r50', choices=['r50', 'r100'],
                        help='ArcFace backbone: iresnet50 (r50) | iresnet100 (r100)')
    parser.add_argument('--weight_path', type=str, default='models',
                        help='Path to pretrained ArcFace backbone .pth weights')
    parser.add_argument('--use_mtcnn', action='store_true', default=False)
    parser.add_argument('--target_far', type=float, default=0.001)
    parser.add_argument('--debug_img', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.format_evaluation == 'offline':
        image_format = 'png'    
    elif args.format_evaluation == 'online':
        image_format = 'npy'
    
    # so the path to the datasets is facial_data
    test_path = Path(f'facial_data/{args.dataset}/processed/test')
    templates_path = Path(f'facial_data/{args.dataset}/processed/templates')
    watermarked_path = Path(f'output/watermarking/{args.watermarking_model}/{args.experiment_name}/inference/{args.train_dataset}/{args.dataset}/watermarked_images')
    watermarked_templates = Path(f'output/watermarking/{args.watermarking_model}/{args.experiment_name}/inference/{args.train_dataset}/{args.dataset}/watermarked_templates')
    # path for output images
    output_images_path = Path(f'output/recognition/{args.watermarking_model}/{args.experiment_name}/{args.train_dataset}/{args.dataset}/arcface/images')
    # weight path for ArcFace backbone
    if args.backbone == 'r50':
        backbone_name = 'ms1mv3_arcface_r50_fp16_backbone.pth'
        weight_path = Path(args.weight_path, backbone_name)
    elif args.backbone == 'r100':
        backbone_name = 'ms1mv3_arcface_r100_fp16_backbone.pth'
        weight_path = Path(args.weight_path, backbone_name)

    # target far for reporting TAR, can be set to 0.0001 for 0.01% FAR, 0.001 for 0.1% FAR, or 0.01 for 1% FAR
    target_far = args.target_far

    print(f"Using ArcFace backbone: {args.backbone} with weights from {weight_path}")
    # Initialize the ArcFace recognizer
    face_recognizer_service = ArcFaceRecognizer(device=args.device, image_format=image_format, use_mtcnn=args.use_mtcnn, 
                                                save_images_path = output_images_path, weight_path=weight_path, network=args.backbone)

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

    elif args.dataset == 'ONOT' or args.dataset == 'ONOT_set1':
        ext = '.png'
        image_paths = list(test_path.glob(f'**/*{ext}'))
        template_paths = list(templates_path.glob(f'**/*{ext}'))

    elif args.dataset == 'LFW':
        ext = '.jpg'
        image_paths = list(test_path.glob(f'**/*{ext}'))
        template_paths = list(templates_path.glob(f'**/*{ext}'))
    elif args.dataset == 'SCface':
        ext = '.JPG'
        template_paths = list(templates_path.glob(f'**/*{ext}'))
        ext = '.jpg'
        image_paths = list(test_path.glob(f'**/*{ext}'))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    if args.format_evaluation == 'offline':
        # offline evaluation, all images are png
        watermarked_paths = list(watermarked_path.glob(f'**/*.{image_format}'))
        watermarked_templates_paths = list(watermarked_templates.glob(f'**/*.{image_format}'))
    elif args.format_evaluation == 'online':
        # online evaluation, all images are npy
        watermarked_paths = list(watermarked_path.glob(f'**/*.{image_format}'))
        watermarked_templates_paths = list(watermarked_templates.glob(f'**/*.{image_format}'))
    else:
        raise ValueError(f"Unsupported evaluation format: {args.format_evaluation}")
    
    # -----------------------------------------------------------------------
    # Generate embeddings
    # When use_mtcnn=True, we detect bounding boxes on the ORIGINAL images
    # and reuse them for the watermarked counterparts. This prevents MTCNN's
    # internal uint8 quantization from destroying subtle watermark differences 
    # and ensures a fair comparison between original and watermarked images.
    # -----------------------------------------------------------------------
    if args.use_mtcnn:
        # Original templates: detect + embed, save boxes
        templates_embs, template_boxes = get_embeddings_and_boxes(
            templates_path, template_paths, args.img_size, face_recognizer_service, debug_img=args.debug_img)
        
        template_identities = set(templates_embs.keys())
 
        # Filter paths to only include identities present in templates
        filtered_image_paths = [p for p in image_paths if get_identity_from_filename(p.name) in template_identities]
        filtered_watermarked_paths = [p for p in watermarked_paths if get_identity_from_filename(p.name) in template_identities]
 
        # Original probes: detect + embed, save boxes
        tests_embs, probe_boxes = get_embeddings_and_boxes(
            test_path, filtered_image_paths, args.img_size, face_recognizer_service, debug_img=args.debug_img)
 
        # Watermarked templates: reuse template_boxes
        watermarked_templates_embs = get_embeddings(
            watermarked_templates, watermarked_templates_paths, args.img_size, 
            face_recognizer_service, image_format=image_format,
            precomputed_boxes=template_boxes, debug_img=args.debug_img)
        
        # Watermarked probes: reuse probe_boxes
        watermarked_embs = get_embeddings(
            watermarked_path, filtered_watermarked_paths, args.img_size, 
            face_recognizer_service, image_format=image_format,
            debug_img=args.debug_img, precomputed_boxes=probe_boxes)    

    else:
        # No MTCNN: process everything independently (no box reuse needed)
        watermarked_templates_embs = get_embeddings(
            watermarked_templates, watermarked_templates_paths, args.img_size, 
            face_recognizer_service, image_format=image_format)
        
        templates_embs = get_embeddings(templates_path, template_paths, args.img_size, face_recognizer_service)
        template_identities = set(templates_embs.keys())
 
        filtered_image_paths = [p for p in image_paths if get_identity_from_filename(p.name) in template_identities]
        filtered_watermarked_paths = [p for p in watermarked_paths if get_identity_from_filename(p.name) in template_identities]
 
        tests_embs = get_embeddings(test_path, filtered_image_paths, args.img_size, face_recognizer_service)
        
        watermarked_embs = get_embeddings(
            watermarked_path, filtered_watermarked_paths, args.img_size, 
            face_recognizer_service, image_format=image_format, debug_img=args.debug_img)
    
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
    genuine_distances_wm_both = []
    impostor_distances_wm_both = []
    genuine_variation_distances = []
    impostor_variation_distances = []
    genuine_raw_distances = []
    impostor_raw_distances  = []
    genuine_raw_distances_template = []
    impostor_raw_distances_template = []
    genuine_variation_distances_template = []
    impostor_variation_distances_template = []

    identities = list(templates_embs.keys())
    genuine_pairs = []
    impostor_pairs = []
    for i, identity_a in tqdm(enumerate(identities), total=len(identities), desc="Calculating distances"):
        for j, identity_b in enumerate(identities):
            # Genuine pairs (same person)
            if identity_a == identity_b:
                genuine_pairs.append((identity_a, identity_b))
                #print(f"Calculating genuine distance for identity: {identity_a}") # original template - original probe
                dist = face_recognizer_service.get_distance(templates_embs[identity_a][0], tests_embs[identity_b][0], metric=args.metric)
                # original template - watermarked probe
                dist_wm = face_recognizer_service.get_distance(templates_embs[identity_a][0], watermarked_embs[identity_b][0], metric=args.metric)
                # watermarked template - watermarked probe
                dist_wm_template = face_recognizer_service.get_distance(watermarked_templates_embs[identity_a][0], watermarked_embs[identity_b][0], metric=args.metric)
                # raw distance between original probe and watermarked probe
                raw_dist = face_recognizer_service.get_distance(tests_embs[identity_b][0], watermarked_embs[identity_b][0], metric=args.metric)
                variation_dist = (dist_wm - dist) 
                # raw distance between original template and watermarked template
                raw_dist_template = face_recognizer_service.get_distance(templates_embs[identity_a][0], watermarked_templates_embs[identity_a][0], metric=args.metric)
                variation_dist_template = (dist_wm_template - dist) 
                
                #print(f"Distance: {dist}")
                #print(f"Distance WM: {dist_wm}")
                #print(f"Raw distance between original and watermarked: {raw_dist}")
                #print(f"Variation in distance due to watermarking: {variation_dist}")

                genuine_distances_baseline.append(dist)
                genuine_distances_wm.append(dist_wm)
                genuine_distances_wm_both.append(dist_wm_template)
                genuine_raw_distances.append(raw_dist)
                genuine_variation_distances.append(variation_dist)
                genuine_raw_distances_template.append(raw_dist_template)
                genuine_variation_distances_template.append(variation_dist_template)
            
            # Impostor pairs (different persons)
            elif i < j:
                if templates_embs[identity_a] and tests_embs[identity_b]:
                    if args.dataset != 'LFW':
                        impostor_pairs.append((identity_a, identity_b))
                    #print(f"Calculating impostor distance for identities: {identity_a} and {identity_b}")
                    # original template - original probe
                    dist = face_recognizer_service.get_distance(templates_embs[identity_a][0], tests_embs[identity_b][0], metric=args.metric)
                    # original template - watermarked probe
                    dist_wm = face_recognizer_service.get_distance(templates_embs[identity_a][0], watermarked_embs[identity_b][0], metric=args.metric)
                    # watermarked template - watermarked probe
                    dist_wm_template = face_recognizer_service.get_distance(watermarked_templates_embs[identity_a][0], watermarked_embs[identity_b][0], metric=args.metric)
                    raw_dist = face_recognizer_service.get_distance(tests_embs[identity_a][0], watermarked_embs[identity_b][0], metric=args.metric)
                    variation_dist = (dist_wm - dist) 
                    # raw distance between original template and watermarked template
                    raw_dist_template = face_recognizer_service.get_distance(templates_embs[identity_a][0], watermarked_templates_embs[identity_b][0], metric=args.metric)
                    variation_dist_template = (dist_wm_template - dist) 

                    #print(f"Distance: {dist}")
                    #print(f"Distance WM: {dist_wm}")
                    #print(f"Raw distance between original and watermarked: {raw_dist}")
                    #print(f"Variation in distance due to watermarking: {variation_dist}")

                    impostor_distances_baseline.append(dist)
                    impostor_distances_wm.append(dist_wm)
                    impostor_distances_wm_both.append(dist_wm_template)
                    impostor_raw_distances.append(raw_dist)
                    impostor_variation_distances.append(variation_dist)
                    impostor_raw_distances_template.append(raw_dist_template)
                    impostor_variation_distances_template.append(variation_dist_template)
    
    # Compute metrics
    metrics_baseline = calculate_metrics(genuine_distances_baseline, impostor_distances_baseline, 
                                         num_thresholds=args.thresholds, target_far=target_far)
    print("Metrics before watermarking:")
    print(metrics_baseline)

    metrics_baseline_wm = calculate_metrics(genuine_distances_wm, impostor_distances_wm, num_thresholds=args.thresholds, target_far=target_far)
    print("Metrics after watermarking:")
    print(metrics_baseline_wm)

    # Compute metrics for template watermarked
    metrics_baseline_wm_template = calculate_metrics(genuine_distances_wm_both, impostor_distances_wm_both, num_thresholds=args.thresholds,
                                                     target_far=target_far)
    print("Metrics after watermarking both:")
    print(metrics_baseline_wm_template)


    # Compute mean and std of the variation distances
    if genuine_variation_distances:
        mean_variation_genuine = np.round(np.mean(genuine_variation_distances) * 100, 3)
        std_variation_genuine = np.round(np.std(genuine_variation_distances) * 100,3)
        print(f"Mean variation in genuine distances due to watermarking: {mean_variation_genuine}")
        print(f"Std deviation of variation in genuine distances due to watermarking: {std_variation_genuine}")

    if impostor_variation_distances:
        mean_variation_impostor = np.round(np.mean(impostor_variation_distances)*100,3)     
        std_variation_impostor = np.round(np.std(impostor_variation_distances)*100,3)
        print(f"Mean variation in impostor distances due to watermarking: {mean_variation_impostor}")
        print(f"Std deviation of variation in impostor distances due to watermarking: {std_variation_impostor}")

    # Compute mean and std of the variation distances for watermarked templates and watermarked probes
    if genuine_variation_distances_template:
        mean_variation_genuine_template = np.round(np.mean(genuine_variation_distances_template) * 100, 3)
        std_variation_genuine_template = np.round(np.std(genuine_variation_distances_template) * 100,3)
        print(f"Mean variation in genuine distances due to watermarking process on both: {mean_variation_genuine_template}")
        print(f"Std deviation of variation in genuine distances due to watermarking process on both: {std_variation_genuine_template}")

    if impostor_variation_distances_template:
        mean_variation_impostor_template = np.round(np.mean(impostor_variation_distances_template)*100,3)     
        std_variation_impostor_template = np.round(np.std(impostor_variation_distances_template)*100,3)
        print(f"Mean variation in impostor distances due to watermarking process on both: {mean_variation_impostor_template}")
        print(f"Std deviation of variation in impostor distances due to watermarking process on both: {std_variation_impostor_template}")

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

    # Compute average and std raw distances for watermarked templates and watermarked probes
    if genuine_raw_distances_template:
        avg_genuine_raw_distance_template = np.round(np.mean(genuine_raw_distances_template),3)
        std_genuine_raw_distance_template = np.round(np.std(genuine_raw_distances_template),3)
        print(f"Average raw distance for genuine pairs templates due to WM: {avg_genuine_raw_distance_template}")
        print(f"Std deviation of raw distance for genuine pairs templates due to WM: {std_genuine_raw_distance_template}")

    if impostor_raw_distances_template:
        avg_impostor_raw_distance_template = np.round(np.mean(impostor_raw_distances_template),3)
        std_impostor_raw_distance_template = np.round(np.std(impostor_raw_distances_template),3)
        print(f"Average raw distance for impostor pairs templates due to WM: {avg_impostor_raw_distance_template}")
        print(f"Std deviation of raw distance for impostor pairs templates due to WM: {std_impostor_raw_distance_template}")

    # Store the distances in csv files
    output_dir = Path(f'output/recognition/{args.watermarking_model}/{args.experiment_name}/{args.train_dataset}/{args.dataset}/arcface/distances')
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(genuine_pairs, columns=['id_a', 'id_b']).to_excel(output_dir / f'{args.dataset}_genuine_pairs.xlsx', index=False)
    if args.dataset != 'LFW':
        pd.DataFrame(impostor_pairs, columns=['id_a', 'id_b']).to_excel(output_dir / f'{args.dataset}_impostor_pairs.xlsx', index=False)
    
    print(f"Saving results to {output_dir}...")
    mtcnn_tag = 'mtcnn' if args.use_mtcnn else 'no-mtcnn'
    if genuine_distances_baseline:
        genuine_df = pd.DataFrame(genuine_distances_baseline, columns=['distance'])
        if args.format_evaluation == 'offline':
            # Save as csv
            genuine_df.to_csv(output_dir / f'{args.metric}_genuine_distances_baseline_{mtcnn_tag}.csv', index=False)
        elif args.format_evaluation == 'online':
            # Save with npy tag
            genuine_df.to_csv(output_dir / f'{args.metric}_genuine_distances_baseline_online_{mtcnn_tag}.csv', index=False)
        
    if impostor_distances_baseline:
        impostor_df = pd.DataFrame(impostor_distances_baseline, columns=['distance'])
        if args.format_evaluation == 'offline':
            # Save as csv
            impostor_df.to_csv(output_dir / f'{args.metric}_impostor_distances_baseline_{mtcnn_tag}.csv', index=False)
        elif args.format_evaluation == 'online':
            # Save with npy tag
            impostor_df.to_csv(output_dir / f'{args.metric}_impostor_distances_baseline_online_{mtcnn_tag}.csv', index=False)
    
    if genuine_distances_wm:
        genuine_wm_df = pd.DataFrame(genuine_distances_wm, columns=['distance'])
        if args.format_evaluation == 'offline':
            genuine_wm_df.to_csv(output_dir / f'{args.metric}_genuine_distances_watermarked_{mtcnn_tag}.csv', index=False)
        elif args.format_evaluation == 'online':
            genuine_wm_df.to_csv(output_dir / f'{args.metric}_genuine_distances_watermarked_online_{mtcnn_tag}.csv', index=False)

    if impostor_distances_wm:
        impostor_wm_df = pd.DataFrame(impostor_distances_wm, columns=['distance'])
        if args.format_evaluation == 'offline':
            impostor_wm_df.to_csv(output_dir / f'{args.metric}_impostor_distances_watermarked_{mtcnn_tag}.csv', index=False)
        elif args.format_evaluation == 'online':
            impostor_wm_df.to_csv(output_dir / f'{args.metric}_impostor_distances_watermarked_online_{mtcnn_tag}.csv', index=False)

    if genuine_distances_wm_both:
        # both watermarked
        genuine_wm_both_df = pd.DataFrame(genuine_distances_wm_both, columns=['distance'])
        if args.format_evaluation == 'offline': 
            genuine_wm_both_df.to_csv(output_dir / f'{args.metric}_genuine_distances_watermarked_both_{mtcnn_tag}.csv', index=False)
        elif args.format_evaluation == 'online':
            genuine_wm_both_df.to_csv(output_dir / f'{args.metric}_genuine_distances_watermarked_both_online_{mtcnn_tag}.csv', index=False)

    if impostor_distances_wm_both:
        impostor_wm_both_df = pd.DataFrame(impostor_distances_wm_both, columns=['distance'])
        if args.format_evaluation == 'offline':
            impostor_wm_both_df.to_csv(output_dir / f'{args.metric}_impostor_distances_watermarked_both_{mtcnn_tag}.csv', index=False)
        elif args.format_evaluation == 'online':
            impostor_wm_both_df.to_csv(output_dir / f'{args.metric}_impostor_distances_watermarked_both_online_{mtcnn_tag}.csv', index=False)

    # Store the results in a new summary json file for recognition
    recognition_summary_path = Path(f'output/recognition/{args.watermarking_model}/{args.experiment_name}/{args.train_dataset}/{args.dataset}/arcface')
    results_filepath = recognition_summary_path / f"results_summary_{mtcnn_tag}.json"

    results_data = {}
    results_data['average_distances'] = {
        f'arcface_avg_var_dist_{args.metric}_genuine_due_watermark': mean_variation_genuine if genuine_variation_distances else None,
        f'arcface_avg_var_dist_{args.metric}_impostor_due_watermark': mean_variation_impostor if impostor_variation_distances else None,
        f'arcface_avg_raw_dist_{args.metric}_genuine_raw_vs_watermark': avg_genuine_raw_distance if genuine_raw_distances else None,
        f'arcface_avg_raw_dist_{args.metric}_impostor_raw_vs_watermark': avg_impostor_raw_distance if impostor_raw_distances else None,
        f'arcface_avg_raw_dist_{args.metric}_genuine_raw_vs_watermark_template': avg_genuine_raw_distance_template if genuine_raw_distances_template else None,
        f'arcface_avg_raw_dist_{args.metric}_impostor_raw_vs_watermark_template': avg_impostor_raw_distance_template if impostor_raw_distances_template else None,
        f'arcface_avg_var_dist_{args.metric}_genuine_due_watermark_both': mean_variation_genuine_template if genuine_variation_distances_template else None,
        f'arcface_avg_var_dist_{args.metric}_impostor_due_watermark_both': mean_variation_impostor_template if impostor_variation_distances_template else None
    }

    # Add the std of distances to the results data
    results_data['std_distances'] = {
        f'arcface_std_var_dist_{args.metric}_genuine_due_watermark': std_variation_genuine if genuine_variation_distances else None,
        f'arcface_std_var_dist_{args.metric}_impostor_due_watermark': std_variation_impostor if impostor_variation_distances else None,
        f'arcface_std_raw_dist_{args.metric}_genuine_raw_vs_watermark': std_genuine_raw_distance if genuine_raw_distances else None,
        f'arcface_std_raw_dist_{args.metric}_impostor_raw_vs_watermark': std_impostor_raw_distance if impostor_raw_distances else None,
        f'arcface_std_raw_dist_{args.metric}_genuine_raw_vs_watermark_template': std_genuine_raw_distance_template if genuine_raw_distances_template else None,
        f'arcface_std_raw_dist_{args.metric}_impostor_raw_vs_watermark_template': std_impostor_raw_distance_template if impostor_raw_distances_template else None,
        f'arcface_std_var_dist_{args.metric}_genuine_due_watermark_both': std_variation_genuine_template if genuine_variation_distances_template else None,
        f'arcface_std_var_dist_{args.metric}_impostor_due_watermark_both': std_variation_impostor_template if impostor_variation_distances_template else None
    }

    # Add the recognition metrics to the results data
    results_data['recognition_metrics'] = {
        f'arcface_EER_baseline_{args.metric}': metrics_baseline['EER'],
        f'arcface_EER_watermarked_{args.metric}': metrics_baseline_wm['EER'],
        f'arcface_FAR_at_EER_baseline_{args.metric}': metrics_baseline['FAR_at_EER'],
        f'arcface_FAR_at_EER_watermarked_{args.metric}': metrics_baseline_wm['FAR_at_EER'],
        f'arcface_FRR_at_EER_baseline_{args.metric}': metrics_baseline['FRR_at_EER'],
        f'arcface_FRR_at_EER_watermarked_{args.metric}': metrics_baseline_wm['FRR_at_EER'],
        f'arcface_AUC_baseline_{args.metric}': metrics_baseline['AUC'],
        f'arcface_AUC_watermarked_{args.metric}': metrics_baseline_wm['AUC'],
        f'arcface_TAR_at_FAR_baseline_{args.metric}': metrics_baseline['TAR_at_FAR'],
        f'arcface_TAR_at_FAR_watermarked_{args.metric}': metrics_baseline_wm['TAR_at_FAR'],
        f'arcface_Actual_FAR_baseline_{args.metric}': metrics_baseline['Actual_FAR'],
        f'arcface_Actual_FAR_watermarked_{args.metric}': metrics_baseline_wm['Actual_FAR'],
        f'arcface_EER_watermarked_both_{args.metric}': metrics_baseline_wm_template['EER'],
        f'arcface_FAR_at_EER_watermarked_both_{args.metric}': metrics_baseline_wm_template['FAR_at_EER'],
        f'arcface_FRR_at_EER_watermarked_both_{args.metric}': metrics_baseline_wm_template['FRR_at_EER'],
        f'arcface_AUC_watermarked_both_{args.metric}': metrics_baseline_wm_template['AUC'],
        f'arcface_TAR_at_FAR_watermarked_both_{args.metric}': metrics_baseline_wm_template['TAR_at_FAR'],
        f'arcface_Actual_FAR_watermarked_both_{args.metric}': metrics_baseline_wm_template['Actual_FAR']
    }

    # Save the updated results data back to the file
    # new_path = results_filepath.with_name("results_summary_with_recognition.json")
    if args.format_evaluation == 'offline':
        with open(results_filepath, "w") as f:
            json.dump(results_data, f, indent=2)
    elif args.format_evaluation == 'online':
        new_path = results_filepath.with_name(f"results_summary_online_{mtcnn_tag}.json")
        with open(new_path, "w") as f:
            json.dump(results_data, f, indent=2)

if __name__ == '__main__':
    main()