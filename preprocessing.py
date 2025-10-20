import numpy as np
from scipy import ndimage #pour fill holes
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.ndimage import label, sum

import numpy as np
from scipy.ndimage import label, sum as ndi_sum, center_of_mass


def get_largest_connected_component_centered(binary_image, use_centroid_weighting=True, center_weight=0.5):
    """
    Optimized version: Finds the largest connected component in a binary image,
    optionally weighted by proximity to image center.
    """
    # Label components
    labeled_array, num_features = label(binary_image)
    if num_features == 0:
        return np.zeros_like(binary_image, dtype=np.uint8)

    # Compute component sizes with bincount (faster than ndi_sum)
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # skip background

    if not use_centroid_weighting:
        # Select largest by size only
        largest_component_label = np.argmax(component_sizes) + 1
    else:
        # Vectorized centroid computation for all components
        centroids = np.array(center_of_mass(binary_image, labeled_array, range(1, num_features + 1)))

        # Image center
        image_center = np.array(binary_image.shape) / 2
        max_distance = np.linalg.norm(image_center)

        # Distance of each centroid to image center (vectorized)
        distances = np.linalg.norm(centroids - image_center, axis=1)
        normalized_distances = distances / max_distance

        # Normalize size and centrality scores
        size_scores = component_sizes / component_sizes.max()
        centrality_scores = 1 - normalized_distances

        # Combined score
        combined_scores = (1 - center_weight) * size_scores + center_weight * centrality_scores

        # Select the component with the best score
        largest_component_label = np.argmax(combined_scores) + 1

    # Create binary mask for the selected component
    return (labeled_array == largest_component_label).astype(np.uint8) * 255


def get_largest_connected_component(binary_image):
    """
    Finds the largest connected component in a binary image.

    Args:
        binary_image (np.array): A 2D NumPy array representing the binary image.

    Returns:
        np.array: A binary image containing only the largest connected component.
    """
    labeled_array, num_features = label(binary_image)
    if num_features == 0:
        return np.zeros_like(binary_image)

    component_sizes = sum(binary_image, labeled_array, range(1, num_features + 1))
    largest_component_label = np.argmax(component_sizes) + 1  # +1 because labels start from 1

    largest_component_mask = (labeled_array == largest_component_label)
    return largest_component_mask

def cleaning_mask(mask_otsu):
    """
    Applies morphological operations to clean the mask by removing small objects and filling holes.
    Parameters:
        mask (numpy.ndarray): The binary mask to be cleaned.
        steps (dict, optional): Dictionary to store intermediate steps for debugging.
    Returns:
        cleaned_mask (numpy.ndarray): The cleaned binary mask.
    """
    ####Trouver la plus grande composante connexe####
    #on trouve la plus grande composante (celle de plus grande surface)
    main_region = get_largest_connected_component_centered(mask_otsu, use_centroid_weighting=True, center_weight=0.65)

    #####FillHoles######
    main_region_bool = main_region.astype(bool) #conversion en booléen pour le remplissage des trous
    #On remplit les trous UNIQUEMENT dans la région principale
    filled_main_region = ndimage.binary_fill_holes(main_region_bool).astype(np.uint8) * 255
    return filled_main_region


def plot_treatment(function, df, n_samples, seed, filepath):
    """ Plot a grid of images after applying a transformation function.

    Parameters:
    - function: A function to apply to each image
    - df: DataFrame containing image metadata
    - n_samples: Number of samples to plot per class
    - seed: Random seed for reproducibility
    """
    n_classes = len(df['dx'].unique())
    
    # Créer une figure avec 2 colonnes par échantillon (avant/après)
    plt.figure(figsize=(4 * n_samples * 2, 4 * n_classes))
    
    plot_idx = 1
    
    for i, class_name in enumerate(df['dx'].unique()):
        class_samples = df[df['dx'] == class_name].sample(n=n_samples, random_state=seed)
        
        for j, (_, row) in enumerate(class_samples.iterrows()):
            #Image après suppression des poils (avant transformation)
            img_hair_removed = remove_hair(filepath[row['image_id'] + '.jpg'], steps_record=False)
            
            plt.subplot(n_classes, n_samples * 2, plot_idx)
            plt.imshow(img_hair_removed, cmap='gray')
            plt.axis('off')
            plt.title(f"{row['dx']} - Avant")
            
            #Image après transformation
            img_transformed = function(img_hair_removed)
            
            plt.subplot(n_classes, n_samples * 2, plot_idx + 1)
            plt.imshow(img_transformed, cmap='gray')
            plt.axis('off')
            plt.title(f"{row['dx']} - Après")
            
            plot_idx += 2
    
    plt.tight_layout()
    plt.show()

def plot_contour_samples(function, df, n_samples, seed, filepath, target_classes=None):
    """Plot a grid of contour images for selected classes.

    Parameters:
    - function: transformation function applied after hair removal
    - df: DataFrame containing image metadata
    - n_samples: Number of samples to plot per class
    - seed: Random seed for reproducibility
    - filepath: Path to the image files
    - target_classes: List of class names to plot (default: all classes in df)
    """
    if target_classes is None:
        target_classes = df['dx'].unique()

    n_classes = len(target_classes)
    plt.figure(figsize=(8 * n_samples, 6 * n_classes))

    plot_idx = 1

    for i, class_name in enumerate(target_classes):
        if class_name not in df['dx'].unique():
            print(f"⚠️ Classe '{class_name}' absente du DataFrame, ignorée.")
            continue

        class_samples = df[df['dx'] == class_name].sample(n=n_samples, random_state=seed)
        for _, row in class_samples.iterrows():
            img_hair_removed = remove_hair(filepath[row['image_id'] + '.jpg'], steps_record=False)
            img_transformed = function(img_hair_removed)
            contour = get_contour(img_transformed)
            
            plt.subplot(n_classes, n_samples, plot_idx)
            plt.imshow(img_hair_removed)
            plt.axis('off')
            plt.title(f"{row['dx']} - Contour")
            
            if contour is not None and len(contour) > 0:
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], color='red', linewidth=2)
            
            plot_idx += 1

    plt.tight_layout()
    plt.show()


##Equalize histograms in ROI
#Most optimized version - combines best of both approaches
def central_roi_mask(img, radius_ratio=0.9):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w//2, h//2)
        axes = (int(w*0.52), int(h*0.67))  #Ajuste le rayon de l'ellipse
        cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
        return mask.astype(bool)

def channels_equalized(img, combine=True):
    ellipse_mask = central_roi_mask(img)
    result = img.copy()
    
    #Process all channels in one operation
    roi_pixels = img[ellipse_mask]
    for c in range(3):
        roi_pixels[:, c] = cv2.equalizeHist(roi_pixels[:, c]).flatten()
    
    #Replace the pixels in the original image
    result[ellipse_mask] = roi_pixels
    return result

def remove_hair(image_path, steps_record=False):
    """
    Applies the dull razor hair removal algorithm to an RGB image.

    Args:
        image_path (string): the path to the image file.

    Returns:
        numpy.ndarray: The image with the hair removed (450, 600, 3) shaped.

    Example:
        remove_hair(image_rgb) returns an rgb image with the hair removed.
    """


    image=cv2.imread(image_path, cv2.IMREAD_COLOR) #image is read in BGR
    if steps_record==True:
        steps = {}
        steps['original_image'] = image.copy()

    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #####Black hat filter####
    #The black hat filter is used to extract the dark regions of the image (hair)
    kernel = cv2.getStructuringElement(1,(9,9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    ####Gaussian filter (to smoothen the blackhat image)####
    bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)

    #####Binary thresholding (MASK)#####
    ret,mask = cv2.threshold(bhg,4,255,cv2.THRESH_BINARY) #Otsu's thresholding

    ######Fill holes in the mask#####
    mask_bool = ~mask.astype(bool) #invert the mask to fill the holes
    filled_outside = ndimage.binary_fill_holes(mask_bool).astype(np.uint8) * 255


    #Replace pixels of the mask
    dst = cv2.inpaint(image, ~filled_outside, 6, cv2.INPAINT_TELEA) #have to invert the mask because inpaint removes the pixels where the mask is 0

    #retransform the image to RGB
    hair_removed = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    ### Record steps if required
    if steps_record == True:
        steps['gray_scale'] = grayScale
        steps['blackhat'] = blackhat
        steps['bhg'] = bhg
        steps['hair_removal_mask'] = mask
        steps['hair_filled_outside'] = filled_outside
        steps['hair_removed'] = hair_removed
        return hair_removed, steps
    else:
        return hair_removed
    


def ratio_otsu(hair_removed, steps=None):
    """
    Segmentation function that extracts the red channel, computes the ratio of red to (blue + green),
    applies Otsu's thresholding, fills holes, and returns the final mask.

    Parameters:
        hair_removed (numpy.ndarray): The image with hair removed.
        steps (dict, optional): Dictionary to store intermediate steps for debugging.
    Returns:
        filled_inside (numpy.ndarray): The final mask with holes filled.
        steps (dict): Dictionary containing intermediate steps if provided.
    """
    #Extraction du canal rouge
    red_channel = hair_removed[:, :, 0]
    
    #Ratio rouge / (bleu + vert) pour intensifier le canal rouge
    ratio = red_channel / (np.sum(hair_removed[:, :, 1:], axis=2) + 1e-6) #on évite la division par zéro en ajoutant une petite constante
    ratio_clipped = np.clip(ratio, 0, 1)
    
    #Conversion en image 8-bit pour Otsu (valeurs entre 0 et 255)
    ratio_8bit = (ratio_clipped * 255).astype(np.uint8)

    ####Seuillage d’Otsu####
    _, mask_otsu = cv2.threshold(ratio_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ####Trouver la plus grande composante connexe####
    #calculs des composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_otsu, connectivity=8)

    #On ignore le background (label 0) et on trouve la plus grande composante (celle de plus grande surface)
    if num_labels > 1:
        largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        main_region = (labels == largest_component_label).astype(np.uint8) * 255
    else:
        main_region = mask_otsu.copy()

    #####FillHoles######
    main_region_bool = main_region.astype(bool) #conversion en booléen pour le remplissage des trous
    #On remplit les trous UNIQUEMENT dans la région principale
    filled_main_region = ndimage.binary_fill_holes(main_region_bool).astype(np.uint8) * 255

    if steps is not None:
        steps['red_channel'] = red_channel
        steps['ratio red/(blue + green)'] = ratio
        steps['ratio red/(blue + green) clipped'] = ratio_clipped
        steps['mask_otsu'] = mask_otsu
        steps['main_region'] = main_region
        steps['filled_main_region'] = filled_main_region
        return filled_main_region, steps

    return filled_main_region

def pca_otsu_opencv(img):
    #Mask the ROI directly
    mask = central_roi_mask(img)
    roi_pixels = img[mask].astype(np.float32)  # OpenCV travaille en float32

    #Calcul du PCA via OpenCV
    mean, eigenvectors = cv2.PCACompute(roi_pixels, mean=None, maxComponents=1)
    first_component = cv2.PCAProject(roi_pixels, mean, eigenvectors)[:, 0]

    #Normalisation 0–255
    norm_pca = cv2.normalize(first_component, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #Seuillage Otsu
    _, binary = cv2.threshold(norm_pca, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #invert mask
    binary = cv2.bitwise_not(binary)

    # Reconstruction du masque final
    mask_out = np.zeros_like(mask, dtype=np.uint8)
    mask_out[mask] = binary.ravel()

    return cleaning_mask(mask_out)


def get_contour(mask):
    """
    Plots the largest contour found in the mask on the image, using Marching Squares algorithm.
    Args:
        image: RGB image (numpy array)
        mask: Binary mask (numpy array, same size as image)
        ax: matplotlib axis to plot on (optional)
        show_label: whether to show 'Contour' label
    """
    #Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Find largest contour
    sizes = [contour.shape[0] for contour in contours]
    k_out = np.argmax(sizes)
    contour = contours[k_out]
    return contour

def pipeline(image_path, segmentation_tech):
    """
    Full pipeline for hair removal and segmentation.
    
    Args:
        image_path (str): Path to the input image.
        steps_record (bool): Whether to record intermediate steps for debugging.
    
    Returns:
        tuple: Hair removed image, final mask, and optionally steps dictionary.
    """
    tech_dict = {
        'pca_otsu_opencv': pca_otsu_opencv,
        'ratio_otsu': ratio_otsu
    }

    if segmentation_tech in tech_dict:
        segmentation_func = tech_dict[segmentation_tech]
    else:
        raise ValueError(f"Unknown segmentation technique: {segmentation_tech}")

    hair_removed = remove_hair(image_path)
    final_mask = segmentation_func(hair_removed)
    return hair_removed, final_mask