import os
import time
from Plant_Analysis import Plant_Analysis


def main():
    # Define dataset path and check its existence.
    dataset_path = "Raw_Images"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Create an instance of Plant_Analysis and update the input path.
    analysis = Plant_Analysis(session=1)
    analysis.update_input_path(dataset_path)

    # Set segmentation model weights and load the model if not loaded.
    analysis.segmentation_model_weights_path = "yolo_segmentation_model.pt"
    if analysis.segmentation_model is None:
        analysis.load_segmentation_model()

    # Run the plant analysis pipeline.
    start_time = time.time()
    analysis.do_plant_analysis()
    total_analysis_time = time.time() - start_time
    print(f"Plant analysis completed in {total_analysis_time:.2f} seconds.")

    # Define the main output folder.
    output_folder = "PlantAnalysisResults"
    os.makedirs(output_folder, exist_ok=True)

    # For each plant, create a subfolder and inside that a "Correlation_Matrix" folder.
    for plant_name in analysis.get_plant_names():
        plant_folder = os.path.join(output_folder, plant_name)
        os.makedirs(plant_folder, exist_ok=True)
        corr_folder = os.path.join(plant_folder, "Correlation_Matrix")
        os.makedirs(corr_folder, exist_ok=True)

        # Run the correlation analysis, saving all VI data and correlation matrices to corr_folder.
        corr_matrix = analysis.calculate_single_plant_vi_correlation(plant_name, output_folder=corr_folder)
        if corr_matrix is not None:
            output_csv_path = os.path.join(corr_folder, f"{plant_name}_VI_Correlation.csv")
            corr_matrix.to_csv(output_csv_path)

    # Save all analysis results (this method creates subfolders for Color Images,
    # Vegetation Indices, and also saves other images in the plant folder).
    analysis.save_results(output_folder)
    print(f"Results have been saved to: {output_folder}")


if __name__ == "__main__":
    main()
