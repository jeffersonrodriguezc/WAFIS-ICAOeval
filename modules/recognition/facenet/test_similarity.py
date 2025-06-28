import argparse
from PIL import Image, ImageOps
from recognizer import FaceNetRecognizer

def apply_roi_transform(img, roi_type='fit', img_size=256):
    if roi_type == 'fit':
        return ImageOps.fit(img, (img_size, img_size))
    elif roi_type == 'crop':
        width, height = img.size
        left = (width - img_size) / 2
        top = (height - img_size) / 2
        right = (width + img_size) / 2
        bottom = (height + img_size) / 2
        return img.crop((left, top, right, bottom))
    else:
        return img

def main():
    parser = argparse.ArgumentParser(description="Compare embeddings between original and watermarked image")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--roi", type=str, choices=["fit", "crop"], default="fit")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()


    original_path = '/app/facial_data/CFD/processed/test/AF253_130_N.jpg'
    watermarked_path = '/app/output/watermarking/stegaformer/1_1_255_w16_learn_im/inference/CFD/watermarked_images/AF253_130_N.jpg' 

    recognizer = FaceNetRecognizer(device=args.device)

    # Load the images
    img1 = apply_roi_transform(Image.open(original_path).convert("RGB"), args.roi, args.img_size)
    img2 = Image.open(watermarked_path).convert("RGB")

    # get embeddings
    emb1 = recognizer.get_embedding(img1)
    emb2 = recognizer.get_embedding(img2)

    if emb1 is None or emb2 is None:
        print("❌ Not was possible to get embeddings from the images.")
        return

    # Calcular distancia
    dist = recognizer.get_distance(emb1, emb2, metric=args.metric)
    print(f"✅ {args.metric} distance between images: {dist:.4f}")

if __name__ == "__main__":
    main()
