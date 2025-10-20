import os
import numpy as np
import pandas as pd
from preprocessing import pipeline, get_contour
from alignement import expend_fourier, standardize_contour, alignment_pipeline
import tqdm
import multiprocessing

path_to_data = 'C://Users//andre//Desktop//Stage 2A//Projet//data//'

# === Single image processing function ===
def process_single_image(args):
    filename, image_path, label, standardized_ref_coefs, n_deltas, M, segmentation_tech= args
    try:
        hair_removed, final_mask = pipeline(image_path, segmentation_tech)
        coefs = alignment_pipeline(
            final_mask, standardized_ref_coefs, n_deltas=n_deltas, M=M
        )

        segmented = hair_removed
        segmented[final_mask == 0] = 0
        segmented = segmented.astype(np.uint8)
        final_mask = final_mask.astype(np.uint8)

        return segmented, coefs, final_mask, label
    except Exception as e:
        print(f"Error processing {filename}: {e}")  # Return the filename in case of failure
        return filename

#=== Main script ===
if __name__ == "__main__":
    df = pd.read_csv(path_to_data + 'HAM10000_metadata.csv')
    df.set_index('image_id', inplace=True)

    filepath = {}
    for filename in os.listdir(path_to_data + "HAM10000_images_part_1"):
        filepath[filename] = os.path.join(path_to_data + "HAM10000_images_part_1", filename)
    for filename in os.listdir(path_to_data + "HAM10000_images_part_2"):
         filepath[filename] = os.path.join(path_to_data + "HAM10000_images_part_2", filename)

    M = 101
    n_deltas = 100
    segmentation_tech = "ratio_otsu"
    classes_chosen = ['nv', 'mel']
    #=== Reference contour for alignment ===
    ref_image_path = path_to_data + "//HAM10000_images_part_1//ISIC_0024803.jpg"
    ref_hair_removed, ref_mask = pipeline(ref_image_path, segmentation_tech)
    ref_contour = get_contour(ref_mask)
    ref_X, ref_Y = ref_contour[:, 0, 0], ref_contour[:, 0, 1]
    ref_coefs, _ = expend_fourier(ref_X, ref_Y, M=M)
    standardized_ref_coefs = standardize_contour(ref_coefs)

    #=== Prepare multiprocessing tasks ===
    print("Preparing image processing tasks...")
    tasks = []
    
    for filename, image_path in filepath.items():
        image_id = filename.split('.')[0]
        if image_id in df.index:
            label = df.loc[image_id]['dx']
            if label not in classes_chosen: #only keep classes we want
                continue
            else:
                tasks.append((filename, image_path, label, standardized_ref_coefs, n_deltas, M, segmentation_tech))

    #=== Process in parallel ===
    print(f"Processing {len(tasks)} images using multiprocessing...")
    X_images, X_align, masks, y = [], [], [], []

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap(process_single_image, tasks), total=len(tasks)))

    #=== Collect results ===
    failed = [] #saves the names of images we failed to process
    for result in results:
        if isinstance(result, str):  #Check if it's a failed task
            failed.append(result)
        else:
            segmented, align_info, mask, label = result
            X_images.append(segmented)
            X_align.append(align_info)
            masks.append(mask)
            y.append(label)

    #=== Save results ===
    save_path = "C://Users//andre//Desktop//Stage 2A//Projet//preprocessed_data//nv_mel_ratio//"
    os.makedirs(save_path, exist_ok=True)

    print("Saving preprocessed data...")
    np.save(save_path + "X_segmented", np.array(X_images, dtype=np.uint8))
    np.save(save_path + "X_align", np.array(X_align, dtype=np.float32))
    np.save(save_path + "masks", np.array(masks, dtype=np.uint8))
    np.save(save_path + "y", np.array(y))

    #save the names of failed images
    with open(save_path + "failed.txt", "w") as f:
        for filename in failed:
            f.write(f"{filename}\n")

    print("Data saved successfully.")
    print(f"Number of images processed: {len(X_images)}")
