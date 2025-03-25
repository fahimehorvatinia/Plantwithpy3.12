from plantcv import plantcv as pcv
import cv2
import numpy as np
from skimage.filters import threshold_li


def get_plant_phenotypes(original_image, offset):
    # --- Preprocessing ---
    # Set debug to None and clear previous outputs.
    pcv.params.debug = None
    pcv.outputs.clear()
    # Set a sample label (this will cause measurements to be stored under "plant_1")
    pcv.params.sample_label = "plant"

    # Convert to grayscale
    gray_image = pcv.rgb2gray(rgb_img=original_image)

    # Use threshold_li to compute a threshold and generate a binary image.
    binary_threshold = threshold_li(gray_image)
    binary_image = (gray_image > binary_threshold).astype(np.uint8)

    # Fill gaps in the binary image.
    filled_img = pcv.fill(bin_img=binary_image, size=10)

    # Define ROI and filter the filled binary image.
    roi = pcv.roi.rectangle(img=original_image, x=95, y=5, h=500, w=350)
    mask = pcv.roi.filter(mask=filled_img, roi=roi, roi_type="partial")

    # Check that the mask has valid pixels.
    if np.sum(mask) == 0:
        print("⚠ No valid plant detected in ROI mask — skipping size analysis.")
        return None

    # Create a labeled mask for analyze.size using connectedComponents.
    ret, labeled_mask = cv2.connectedComponents(mask.astype(np.uint8))
    labeled_mask = labeled_mask.astype(np.uint8)
    # Force all nonzero labels to 1 (ensuring one object)
    labeled_mask[labeled_mask > 0] = 1

    # --- Analyze Size and Shape ---
    # Add metadata (optional but helps to track outputs)
    pcv.outputs.add_metadata(term="plantbarcode", datatype=str, value="SamplePlant")
    pcv.outputs.add_metadata(term="camera", datatype=str, value="MockCam")
    pcv.outputs.add_metadata(term="timestamp", datatype=str, value="2025-03-23 00:00:00")

    # Run size analysis using the labeled mask. In the new PlantCV, analyze.size
    # stores observations under the key "plant_1" (since pcv.params.sample_label = "plant").
    try:
        _ = pcv.analyze.size(img=original_image, labeled_mask=labeled_mask, n_labels=1, label=pcv.params.sample_label)
    except Exception as e:
        print(f"⚠ Error in analyze.size: {e}")

    # --- Morphological Analysis ---
    # Skeletonize the same mask.
    try:
        skeleton = pcv.morphology.skeletonize(mask=mask)
        if np.count_nonzero(skeleton) == 0:
            raise ValueError("No skeleton pixels found")
    except Exception as e:
        print(f"⚠️ Skeletonization failed: {e}")
        skeleton = None

    # Segment the skeleton if available.
    segmented_image, segmented_objects = None, []
    if skeleton is not None:
        try:
            segmented_image, segmented_objects = pcv.morphology.segment_skeleton(skel_img=skeleton, mask=mask)
        except Exception as e:
            print(f"⚠️ Segmenting skeleton failed: {e}")

    # Sort the segmented skeleton objects if valid segments were obtained.
    primary_objects, secondary_objects = [], []
    if skeleton is not None and segmented_objects and len(segmented_objects) > 0:
        try:
            secondary_objects, primary_objects = pcv.morphology.segment_sort(skel_img=skeleton,
                                                                             objects=segmented_objects,
                                                                             mask=mask,
                                                                             first_stem=True)
        except Exception as e:
            print(f"⚠ Sorting skeleton segments failed: {e}")
            secondary_objects, primary_objects = [], []
    else:
        print("⚠ No valid segments found; skipping segment sorting.")

    if skeleton is None or (len(primary_objects) == 0 and len(secondary_objects) == 0):
        print("⚠ Morphological analysis did not yield valid features.")
    else:
        print(" Morphological analysis completed.")

    num_branches = len(primary_objects)
    num_leaves = len(secondary_objects)

    # --- Retrieve Measurements ---
    # In the new PlantCV, analyze.size stores outputs under pcv.outputs.observations
    # with a key based on the sample_label (here, "plant_1").
    try:
        plant_height_pixels = pcv.outputs.observations['plant_1']['height']['value']
        plant_width_pixels = pcv.outputs.observations['plant_1']['width']['value']
        plant_area_pixels = pcv.outputs.observations['plant_1']['area']['value']
        plant_perimeter_pixels = pcv.outputs.observations['plant_1']['perimeter']['value']
        plant_solidity = pcv.outputs.observations['plant_1']['solidity']['value']
    except Exception as e:
        print(f"⚠️ Error retrieving outputs: {e}")
        plant_height_pixels = plant_width_pixels = plant_area_pixels = plant_perimeter_pixels = plant_solidity = 0

    image_height, image_width, image_channels = original_image.shape

    actual_greenhouse_height = plant_height_pixels * offset
    actual_greenhouse_width = plant_width_pixels * offset
    actual_greenhouse_area = plant_area_pixels * (offset ** 2)
    actual_greenhouse_perimeter = plant_perimeter_pixels * offset

    # Create and return a dictionary with all phenotypes.
    data = {
        'Image Height (pixels)': image_height,
        'Image Width (pixels)': image_width,
        'Pixel Size (cm)': offset,
        'Plant Height (pixels)': plant_height_pixels,
        'Plant Height (cm)': actual_greenhouse_height,
        'Plant Width (pixels)': plant_width_pixels,
        'Plant Width (cm)': actual_greenhouse_width,
        'Plant Area (pixels)': plant_area_pixels,
        'Plant Area (square cm)': actual_greenhouse_area,
        'Plant Perimeter (pixels)': plant_perimeter_pixels,
        'Plant Perimeter (cm)': actual_greenhouse_perimeter,
        'Plant Solidity': plant_solidity,
        'Number of Branches': num_branches,
        'Number of Leaves': num_leaves
    }
    return data
