"""
Crop image regions based on YOLO format annotations
Format: class_id x_center y_center width height (normalized 0-1)
"""

import cv2
import sys
import os

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO format (normalized) to absolute pixel coordinates
    
    Args:
        x_center, y_center, width, height: YOLO normalized values (0-1)
        img_width, img_height: Image dimensions in pixels
    
    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    # Convert to pixels
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate corners
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    return (x1, y1, x2, y2)

def crop_by_yolo(image_path, yolo_txt_path, output_dir="cropped", padding=0):
    """
    Crop image regions based on YOLO annotations
    
    Args:
        image_path: Path to image file
        yolo_txt_path: Path to YOLO format txt file
        output_dir: Output directory for cropped images
        padding: Extra padding around bbox (pixels)
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return False
    
    img_height, img_width = image.shape[:2]
    
    print("="*60)
    print("CROP IMAGE BY YOLO BOXES (EXACT)")
    print("="*60)
    print(f"Image: {image_path} ({img_width}x{img_height})")
    print(f"Annotations: {yolo_txt_path}")
    print(f"Padding: {padding} pixels (0 = exact YOLO bbox)")
    print()
    
    # Read YOLO annotations
    if not os.path.exists(yolo_txt_path):
        print(f"❌ Annotation file not found: {yolo_txt_path}")
        return False
    
    with open(yolo_txt_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 5:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        annotations.append({
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        })
    
    print(f"Found {len(annotations)} annotations")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Crop each region
    for i, ann in enumerate(annotations):
        # Convert to pixel coordinates
        x1, y1, x2, y2 = yolo_to_bbox(
            ann['x_center'], ann['y_center'],
            ann['width'], ann['height'],
            img_width, img_height
        )
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        # Save
        output_path = f"{output_dir}/class_{ann['class_id']}_crop_{i}.png"
        cv2.imwrite(output_path, cropped)
        
        print(f"  Crop {i}: class={ann['class_id']}, "
              f"bbox=({x1},{y1},{x2},{y2}), "
              f"size={x2-x1}x{y2-y1} → {output_path}")
    
    print()
    print("="*60)
    print(f"✅ Cropped {len(annotations)} regions to {output_dir}/")
    print("="*60)
    
    # Also create visualization
    vis_image = image.copy()
    for i, ann in enumerate(annotations):
        x1, y1, x2, y2 = yolo_to_bbox(
            ann['x_center'], ann['y_center'],
            ann['width'], ann['height'],
            img_width, img_height
        )
        
        color = (0, 255, 0)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        label = f"#{i} c{ann['class_id']}"
        cv2.putText(vis_image, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    vis_path = f"{output_dir}/visualization.png"
    cv2.imwrite(vis_path, vis_image)
    print(f"📊 Visualization: {vis_path}")
    
    return True

def main():
    if len(sys.argv) < 3:
        print("Crop Image by YOLO Annotations")
        print("\nUsage:")
        print("  python crop_by_yolo.py <image> <yolo_txt> [output_dir] [padding]")
        print("\nExample:")
        print("  python crop_by_yolo.py output_png/page_0.png output_png/page_0.txt cropped")
        print("  python crop_by_yolo.py output_png/page_0.png output_png/page_0.txt cropped 5  # with padding")
        print()
        print("YOLO format: class_id x_center y_center width height (normalized 0-1)")
        print("Default padding: 0 (exact YOLO bbox)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    yolo_txt_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "cropped"
    padding = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    success = crop_by_yolo(image_path, yolo_txt_path, output_dir, padding)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
