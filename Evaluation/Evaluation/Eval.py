import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


LOWER_BLUE = np.array([100, 0, 0])
UPPER_BLUE = np.array([255, 100, 100])
LOWER_RED = np.array([0, 0, 100])
UPPER_RED = np.array([100, 100, 255])


MIN_AREA_FOR_GT_CIRCLE = 500
MIN_AREA_FOR_PREDICTION_CIRCLE = 400

MIN_CIRCULARITY_THRESHOLD = 0.40

MATCHING_DISTANCE_THRESHOLD  = 250

def create_color_mask(image, color_ranges):
    final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lower_bound, upper_bound in color_ranges:
        current_mask = cv2.inRange(image, lower_bound, upper_bound)
        final_mask = cv2.bitwise_or(final_mask, current_mask)
    return final_mask


def find_circles(mask, min_area, min_circularity):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    valid_contours = []
    for contour in contours:

        area = cv2.contourArea(contour)
        if area > min_area:

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > min_circularity:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroids.append((cX, cY))
                    valid_contours.append(contour)
                    
    return valid_contours, centroids

def evaluate_and_visualize(gt_folder, pred_folder, output_folder, max_distance_for_match):
 
    os.makedirs(output_folder, exist_ok=True)
    gt_image_paths = sorted(glob.glob(os.path.join(gt_folder, '*')))
    
    if not gt_image_paths:
        print("Error: Ground truth folder is empty or does not exist.")
        return

    total_f1_scores = []
    print(f"Found {len(gt_image_paths)} images to process.")
    print("-" * 40)

    for gt_path in gt_image_paths:
        filename = os.path.basename(gt_path)
        base_name = os.path.splitext(filename)[0]
        search_pattern = os.path.join(pred_folder, f'*{base_name}*')
        matching_preds = glob.glob(search_pattern)
        
        if not matching_preds:
            print(f"Skipping {filename}: No corresponding prediction file found containing '{base_name}'.")
            continue
        elif len(matching_preds) > 1:
            print(f"Skipping {filename}: Found multiple possible prediction files: {matching_preds}. Please clean up the prediction folder.")
            continue
        
        pred_path = matching_preds[0]
        print(f"Processing: {filename}  <-->  {os.path.basename(pred_path)}")

        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_COLOR)

        gt_h, gt_w, _ = gt_img.shape
        if gt_img.shape != pred_img.shape:
            pred_img = cv2.resize(pred_img, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
        
        
        crop_width = int(gt_w * 0.85)  # Keep the left 85% of the image
        
        gt_img_for_detection = gt_img[:, :crop_width]
        pred_img_for_detection = pred_img[:, :crop_width]
        # --- END OF NEW CODE ---
        
        COLOR_RANGES_TO_DETECT = [(LOWER_BLUE, UPPER_BLUE), (LOWER_RED, UPPER_RED)]
        
        # Use the CROPPED images for creating the mask
        gt_mask = create_color_mask(gt_img_for_detection, COLOR_RANGES_TO_DETECT)
        pred_mask = create_color_mask(pred_img_for_detection, COLOR_RANGES_TO_DETECT)
        
        # find_circles will now only operate on the main map area
        gt_contours, gt_centroids = find_circles(gt_mask, MIN_AREA_FOR_GT_CIRCLE, MIN_CIRCULARITY_THRESHOLD)
        pred_contours, pred_centroids = find_circles(pred_mask, MIN_AREA_FOR_PREDICTION_CIRCLE, MIN_CIRCULARITY_THRESHOLD)
        
        num_gt = len(gt_centroids)
        num_pred = len(pred_centroids)
        
        true_positives, matched_pairs = 0, []
        if num_gt > 0 and num_pred > 0:
            unmatched_preds = list(range(num_pred))
            for gt_c in gt_centroids:
                distances = [np.linalg.norm(np.array(gt_c) - np.array(p_c)) for p_c in pred_centroids]
                if not distances: continue
                min_dist_idx = np.argmin(distances)
                if np.min(distances) < max_distance_for_match and min_dist_idx in unmatched_preds:
                    true_positives += 1
                    matched_pairs.append((gt_c, pred_centroids[min_dist_idx]))
                    unmatched_preds.remove(min_dist_idx)
                    
        false_positives = num_pred - true_positives
        false_negatives = num_gt - true_positives
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        total_f1_scores.append(f1_score)
        
        print(f"  - GT Circles Detected: {num_gt}")
        print(f"  - Predicted Circles Detected: {num_pred}")
        print(f"  - TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
        print(f"  - Object F1-Score (Accuracy): {f1_score * 100:.2f}%")


        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        # Use the ORIGINAL, full images for display
        gt_display, pred_display = gt_img.copy(), pred_img.copy()
        cv2.drawContours(gt_display, gt_contours, -1, (0, 255, 0), 3) 
        axes[0].imshow(cv2.cvtColor(gt_display, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Ground Truth (Detected Circles: {num_gt})')
        axes[0].axis('off')
        cv2.drawContours(pred_display, pred_contours, -1, (255, 0, 255), 3)
        axes[1].imshow(cv2.cvtColor(pred_display, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Prediction (Detected Circles: {num_pred})')
        axes[1].axis('off')
        overlay_display = gt_img.copy()
        cv2.drawContours(overlay_display, gt_contours, -1, (0, 255, 0), 3) 
        cv2.drawContours(overlay_display, pred_contours, -1, (0, 0, 255), 3) 
        for gt_c, pred_c in matched_pairs:

            cv2.line(overlay_display, gt_c, pred_c, (255, 255, 0), 2)
        axes[2].imshow(cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB))

        axes[2].set_title(f'Overlay (GT=Green, Pred=Red, Match=Cyan)')
        axes[2].axis('off')
        fig.suptitle(f'Comparison for {filename}\nObject F1-Score: {f1_score*100:.2f}% (TP:{true_positives}, FP:{false_positives}, FN:{false_negatives})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = os.path.join(output_folder, f'comparison_{filename}')
        plt.savefig(output_path)
        plt.close(fig)

    print("-" * 40)
    print("Evaluation Summary (Averages):")
    if total_f1_scores:
        avg_f1 = np.mean(total_f1_scores)
        print(f"Average Object-level F1-Score: {avg_f1 * 100:.2f}%")
    else:
        print("No images were processed.")

if __name__ == '__main__':

    GROUND_TRUTH_FOLDER = 'C:/Users/shriv/OneDrive/Desktop/oos/PSPNet/ground_truth'
    PREDICTIONS_FOLDER = 'C:/Users/shriv/OneDrive/Desktop/oos/PSPNet/Yearwithoutlabel'
    OUTPUT_FOLDER = 'C:/Users/shriv/OneDrive/Desktop/oos/PSPNet/Eval_Dis'
    
    evaluate_and_visualize(
        gt_folder=GROUND_TRUTH_FOLDER,
        pred_folder=PREDICTIONS_FOLDER,
        output_folder=OUTPUT_FOLDER,
        max_distance_for_match=MATCHING_DISTANCE_THRESHOLD
    )