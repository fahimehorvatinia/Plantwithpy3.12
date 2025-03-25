import pandas as pd
from itertools import product
from Connect_Components_Preprocessing import CCA_Preprocess
from Image_Stitching import *
from matplotlib import cm
from Plant_Phenotypes import *
from Image_Segmentation import *
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import pickle
import json
import inspect
import shutil
from plantcv import plantcv as pcv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yaml
import seaborn as sns
import cv2

VEG_INDEX_FORMULAS = {
    "MCARIOSAVI": lambda red_edge, red, green, nir: np.divide(((red_edge - red) - 0.2*(red_edge - green))*(red_edge/red), (nir - red)/(nir+red+0.16+1e-10), out=np.zeros_like(nir), where=(nir+red+0.16)!=0),
    "MCARI": lambda red_edge, red, green: ((red_edge - red) - 0.2*(red_edge - green)) * np.divide(red_edge, red+1e-10, out=np.zeros_like(red_edge), where=red!=0),
    "GRNDVI": lambda nir, green, red: np.divide(nir-(green+red), nir+(green+red)+1e-10, out=np.zeros_like(nir), where=(nir+green+red)!=0),
    "CVI": lambda nir, red, green: np.divide(nir*red, (green**2.0)+1e-10, out=np.zeros_like(nir), where=(green**2.0)!=0),
    "ARI2": lambda nir, green, red_edge: nir*np.divide(1, green+1e-10, out=np.zeros_like(nir), where=green!=0) - nir*np.divide(1, red_edge+1e-10, out=np.zeros_like(nir), where=red_edge!=0),
    "BIXS": lambda green, red, epsilon=1e-10: (((green**2.0)+(red**2.0))/2.0)**0.5,
    "CIRE": lambda nir, red_edge, epsilon=1e-10: (nir/(red_edge+epsilon))-1.0,
    "DSWI4": lambda green, red, epsilon=1e-10: green/(red+epsilon),
    "DVI": lambda nir, red, epsilon=1e-10: nir-red,
    "ExR": lambda red, green, epsilon=1e-10: 1.3*red - green,
    "GEMI": lambda nir, red, epsilon=1e-10: ((2.0*((nir**2.0)-(red**2.0))+1.5*nir+0.5*red)/(nir+red+0.5+epsilon)),
    "GNDVI": lambda nir, green, epsilon=1e-10: (nir-green)/(nir+green+epsilon),
    "GOSAVI": lambda nir, green, epsilon=1e-10: (nir-green)/(nir+green+0.16+epsilon),
    "GRVI": lambda nir, green, epsilon=1e-10: nir/(green+epsilon),
    "IPVI": lambda nir, red, epsilon=1e-10: nir/(nir+red+epsilon),
    "MCARI1": lambda nir, red, green, epsilon=1e-10: 1.2*(2.5*(nir-red)-1.3*(nir-green)),
    "MCARI2": lambda nir, red, green, epsilon=1e-10: (1.5*(2.5*(nir-red)-1.3*(nir-green)))/np.sqrt((2*nir+1)**2-(6*nir-5*np.sqrt(red+epsilon))),
    "MGRVI": lambda green, red, epsilon=1e-10: (green**2.0-red**2.0)/(green**2.0+red**2.0+epsilon),
    "MSAVI": lambda nir, red, epsilon=1e-10: 0.5*(2.0*nir+1-np.sqrt((2*nir+1)**2-8*(nir-red))),
    "MSR": lambda nir, red, epsilon=1e-10: (nir/(red+epsilon)-1)/np.sqrt(nir/(red+epsilon)+1),
    "MTVI1": lambda nir, green, red, epsilon=1e-10: 1.2*(1.2*(nir-green)-2.5*(red-green)),
    "MTVI2": lambda nir, green, red, epsilon=1e-10: (1.5*(1.2*(nir-green)-2.5*(red-green)))/np.sqrt((2*nir+1)**2-(6*nir-5*np.sqrt(red+epsilon))),
    "NDVI": lambda nir, red, epsilon=1e-10: (nir-red)/(nir+red+epsilon),
    "NDRE": lambda nir, red_edge, epsilon=1e-10: (nir-red_edge)/(nir+red_edge+epsilon),
    "NDWI": lambda green, nir, epsilon=1e-10: (green-nir)/(green+nir+epsilon),
    "NLI": lambda nir, red, epsilon=1e-10: ((nir**2)-red)/((nir**2)+red+epsilon),
    "OSAVI": lambda nir, red, soil_factor=0.16: np.divide(nir-red, nir+red+soil_factor+1e-10, out=np.zeros_like(nir), where=(nir+red+soil_factor)!=0),
    "RDVI": lambda nir, red, epsilon=1e-10: (nir-red)/np.sqrt(nir+red+epsilon),
    "PVI": lambda nir, red, a=0.5, b=0.3, epsilon=1e-10: (nir-a*red-b)/(np.sqrt(1+a**2)+epsilon),
    "SR": lambda nir, red, epsilon=1e-10: nir/(red+epsilon),
    "TCARIOSAVI": lambda red_edge, red, green, nir: np.divide(3*((red_edge-red)-0.2*(red_edge-green)*np.divide(red_edge,red+1e-10, out=np.zeros_like(red_edge), where=red!=0)), np.divide(nir-red, nir+red+0.16+1e-10, out=np.zeros_like(nir), where=(nir+red+0.16)!=0), out=np.zeros_like(nir), where=np.divide(nir-red, nir+red+0.16+1e-10, where=(nir+red+0.16)!=0)!=0),
    "TNDVI": lambda nir, red: np.sqrt(np.divide(nir-red, nir+red+1e-10, out=np.zeros_like(nir), where=(nir+red)!=0)+0.5),
    "TSAVI": lambda nir, red, a=0.5, x=0.08, b=0.03, epsilon=1e-10: (a*(nir-a*red-b))/(a*nir+red-a*b+x*(1+a**2)+epsilon),
    "GSAVI": lambda nir, green, l=0.5: (1+l)*np.divide(nir-green, nir+green+l+1e-10, out=np.zeros_like(nir), where=(nir+green+l)!=0),
    "RI": lambda red, green, epsilon=1e-10: (red-green)/(red+green+epsilon),
    "TCARI": lambda red_edge, red, green: 3*((red_edge-red)-0.2*(red_edge-green)*np.divide(red_edge,red+1e-10, out=np.zeros_like(red_edge), where=red!=0)),
    "IRECI": lambda nir, red_edge, red: np.divide(nir-red_edge, red_edge-red+1e-10, out=np.zeros_like(nir), where=(red_edge-red)!=0),
    "LCI": lambda nir, red_edge, epsilon=1e-10: (nir-red_edge)/(nir+red_edge+epsilon),
    "CIgreen": lambda nir, green, epsilon=1e-10: (nir/(green+epsilon))-1,
    "NGRDI": lambda green, red, epsilon=1e-10: (green-red)/(green+red+epsilon),
}

class Plant_Analysis:
    def __init__(self, session):
        try:
            with open('pipeline_config.yaml', 'r') as file:
                self.pipeline_config = yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load pipeline config: {e}")
        self.plants = {}
        self.segmentation_model_weights_path = self.pipeline_config.get('segmentation_model_weights_path', 'default_model.pth')
        self.batch_size = self.pipeline_config.get('pipeline_batch_size', 1)
        self.service_type = 0  # 0: Multi Plant Analysis, 1: Single Plant Analysis
        self.input_folder_path = None
        self.output_folder_path = None
        self.show_raw_images = False
        self.show_color_images = False
        self.plant_paths = {}
        self.plant_stats = {}
        self.interm_result_folder = f'Interm_Results_{session}'
        self.segmentation_model = None
        self.variable_k = self.pipeline_config.get('cca_variable_k', 5)
        self.raw_channel_names = self.pipeline_config.get('raw_channel_names', ['Red', 'Green', 'Red Edge', 'NIR'])
        self.device = self.pipeline_config.get('device', 'cpu')
        self.LBP_radius = self.pipeline_config.get('LBP_radius', 1)
        self.LBP_n_points = 8 * self.LBP_radius
        self.offset = self.pipeline_config.get('offset', 1.0)
        self.analysis_items = [
            'stitched_image', 'cca_image', 'segmented_image', 'tips', 'branches',
            'tips_and_branches', 'sift_features', 'lbp_features', 'ndvi_image'
        ]
        self.statistics_items = [
            'Height', 'Width', 'Area', 'Perimeter', 'Solidity', 'Number of Branches', 'Number of Leaves',
            'NDVI (Maximum)', 'NDVI (Minimum)', 'NDVI (Average)', 'NDVI (Positive Average)',
            'ARI (Maximum)', 'ARI (Minimum)', 'ARI (Average)',
            'TCARI (Maximum)', 'TCARI (Minimum)', 'TCARI (Average)',
            'PRI (Maximum)', 'PRI (Minimum)', 'PRI (Average)', 'OSAVI', 'MCARI2', 'GARI',
            'VARI', 'MTVI2', 'CIgreen', 'MTCI', 'EVI2', 'SIPI2', 'CVI2'
        ]
        self.statistics_units = [
            ' cm' if 'Height' in item or 'Width' in item or 'Perimeter' in item
            else ' square cm' if 'Area' in item
            else ''
            for item in self.statistics_items
        ]

    def update_service_type(self, service):
        self.service_type = service

    def check_for_ipynb(self, input_list):
        if '.ipynb_checkpoints' in input_list:
            input_list.remove('.ipynb_checkpoints')
        return input_list

    def parse_folders(self):
        if self.service_type == 0:
            folder_path = self.input_folder_path
            plant_folders = sorted(os.listdir(folder_path))
            for plant_folder in plant_folders:
                self.plant_paths[plant_folder] = {}
                self.plant_stats[plant_folder] = {}
                self.plant_paths[plant_folder]['raw_images'] = []
                plant_folder_path = os.path.join(folder_path, plant_folder)
                image_names = self.check_for_ipynb(sorted(os.listdir(plant_folder_path)))
                for image_name in image_names:
                    image_path = os.path.join(plant_folder_path, image_name)
                    self.plant_paths[plant_folder]['raw_images'].append(image_path)
        if self.service_type == 1:
            plant_folder_path = self.input_folder_path
            plant_name = plant_folder_path.split('/')[-1]
            self.plant_paths[plant_name] = {}
            self.plant_stats[plant_name] = {}
            self.plant_paths[plant_name]['raw_images'] = []
            image_names = self.check_for_ipynb(sorted(os.listdir(plant_folder_path)))
            for image_name in image_names:
                image_path = os.path.join(plant_folder_path, image_name)
                self.plant_paths[plant_name]['raw_images'].append(image_path)

    def update_input_path(self, input_path):
        self.input_folder_path = input_path
        self.parse_folders()

    def update_check_RI_option(self, check_RI):
        self.show_raw_images = check_RI

    def update_check_CI_option(self, check_CI):
        self.show_color_images = check_CI

    def load_segmentation_model(self):
        self.segmentation_model = load_yolo_model(self.segmentation_model_weights_path)

    def get_plant_names(self):
        return sorted(list(self.plant_paths.keys()))

    def get_raw_images(self, plant):
        return [(Image.open(image_path), image_path.split('/')[-1].split('.')[0]) for image_path in self.plant_paths[plant]['raw_images']]

    def get_color_images(self, plant):
        if plant not in self.plant_paths or 'color_images_pickle' not in self.plant_paths[plant]:
            return []
        try:
            with open(self.plant_paths[plant]['color_images_pickle'], 'rb') as handle:
                color_images = pickle.load(handle).get('color_images', [])
            return [(image.astype(np.uint8), image_name) for image, image_name in color_images]
        except Exception:
            return []

    def get_segmented_image(self, plant):
        return cv2.imread(self.plant_paths[plant]['segmented_image'])

    def get_plant_analysis_images(self, plant):
        with open(self.plant_paths[plant]['plant_analysis_pickle'], 'rb') as handle:
            plant_analysis_dict = pickle.load(handle)
        return [(image.astype(np.uint8), image_name) for item in self.analysis_items
                if item in plant_analysis_dict and plant_analysis_dict[item] is not None
                for image, image_name in [plant_analysis_dict[item]]]

    def get_plant_height(self, plant):
        return str(round(self.plant_stats[plant]['Height'], 2)) + ' cm'

    def get_plant_statistics_df_plantwise(self, plant):
        return pd.DataFrame({'Phenotypic trait': self.statistics_items,
                             'Value': [str(round(self.plant_stats[plant][self.statistics_items[index]], 2)) + self.statistics_units[index]
                                       for index in range(len(self.statistics_items))]})

    def tile(self, image, d=2):
        w, h = image.size
        grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
        boxes = []
        for i, j in grid:
            boxes.append((j, i, j + d, i + d))
        return boxes

    def make_batches(self):
        self.batches = []
        plant_names = self.get_plant_names()
        num_plants = len(plant_names)
        num_batches = num_plants // self.batch_size
        for i in range(num_batches + 1):
            begin = self.batch_size * i
            end = min(num_plants, self.batch_size * (i + 1))
            if end > begin:
                self.batches.append(plant_names[begin:end])

    def do_plant_analysis(self):
        start_time = time.time()
        self.make_batches()
        if self.segmentation_model is None:
            self.load_segmentation_model()
        for batch in self.batches:
            self.plants = {}
            self.load_raw_images(batch)
            self.get_ndvi_image_indices(batch)
            self.make_color_images(batch)
            self.stitch_color_images(batch)
            self.calculate_connected_components(batch)
            self.run_segmentation(batch)
            self.calculate_plant_phenotypes(batch)
            self.calculate_tips_and_branches(batch)
            self.calculate_sift_features(batch)
            self.calculate_lbp_features(batch)
            self.calculate_hog_features(batch)
            self.calculate_vegetation_indices(batch)
        total_time = time.time() - start_time
        print(f"Plant analysis completed in {total_time:.2f} seconds.")

    def load_raw_images(self, batch):
        for plant_name in batch:
            self.plants[plant_name] = {}
            self.plants[plant_name]['raw_images'] = []
            for image_path in self.plant_paths[plant_name]['raw_images']:
                image_name = image_path.split('/')[-1].split('.')[0]
                self.plants[plant_name]['raw_images'].append((Image.open(image_path), image_name))

    def get_ndvi_image_indices(self, batch):
        for plant_name in batch:
            num_images = len(self.plants[plant_name]['raw_images'])
            index = (num_images // 2) - 1 if num_images % 2 == 0 else (num_images // 2)
            self.plant_paths[plant_name]['ndvi_image_index'] = index

    def make_color_images(self, batch):
        for plant_name in batch:
            if 'raw_images' not in self.plants[plant_name] or not self.plants[plant_name]['raw_images']:
                continue
            self.plants[plant_name]['color_images'] = []
            image_index = 0
            for raw_image, image_name in self.plants[plant_name]['raw_images']:
                size = raw_image.size[0] // 2
                slices = self.tile(raw_image, d=size)
                image_stack = np.zeros((size, size, len(slices)))
                for idx, box in enumerate(slices):
                    image_stack[:, :, idx] = np.array(raw_image.crop(box))
                red = np.expand_dims(image_stack[:, :, 1], axis=-1)
                green = np.expand_dims(image_stack[:, :, 0], axis=-1)
                red_edge = np.expand_dims(image_stack[:, :, 2], axis=-1)
                NIR = np.expand_dims(image_stack[:, :, -1], axis=-1)
                composite_image = np.concatenate((green, red_edge, red), axis=-1) * 255
                normalized_image = ((composite_image - composite_image.min()) * 255 / (composite_image.max() - composite_image.min())).astype(np.uint8)
                self.plants[plant_name]['color_images'].append((normalized_image, image_name))
                if self.plant_paths[plant_name].get('ndvi_image_index', None) == image_index:
                    self.plants[plant_name]['ndvi_inputs'] = {
                        'red': red,
                        'NIR': NIR,
                        'red_edge': red_edge,
                        'green': green,
                        'color': normalized_image
                    }
                image_index += 1

    def calculate_vegetation_indices(self, batch):
        pcv.params.debug = None
        epsilon = self.pipeline_config['ndvi_epsilon']
        L = 0.5
        soil_factor = 0.16
        ndvi_min, ndvi_max = -1.0, 1.0
        input_images = [self.plants[plant_name]['ndvi_inputs']['color'] for plant_name in batch]
        results = self.segmentation_model.predict(input_images, conf=self.pipeline_config['segmentation_confidence'], device=self.device)
        for result_index, result in enumerate(results):
            if not result:
                continue
            plant_name = batch[result_index]
            if result.masks.data.shape[0] > 4:
                result.masks.data = result.masks.data[:4]
            mask = preprocess_mask(result.masks.data)
            binary_mask_np = generate_binary_mask(mask)
            segmented_color_image = overlay_mask_on_image(binary_mask_np, input_images[result_index])
            original_image = segmented_color_image
            gray_image = pcv.rgb2gray(rgb_img=original_image)
            binary_threshold = threshold_li(gray_image)
            binary_image = (gray_image > binary_threshold).astype(np.uint8)
            filled_binary_image = pcv.fill(bin_img=binary_image, size=10)
            roi = pcv.roi.rectangle(img=original_image, x=95, y=5, h=500, w=350)
            filtered_mask = pcv.roi.filter(mask=filled_binary_image, roi=roi, roi_type='partial')
            if np.count_nonzero(filtered_mask) == 0:
                continue
            labeled_mask = pcv.create_labels(mask=filtered_mask)
            composed_mask = filtered_mask
            bands = self.plants[plant_name]['ndvi_inputs']
            red = pcv.apply_mask(img=bands['red'], mask=composed_mask, mask_color='black')
            nir = pcv.apply_mask(img=bands['NIR'], mask=composed_mask, mask_color='black')
            green = pcv.apply_mask(img=bands['green'], mask=composed_mask, mask_color='black')
            red_edge = pcv.apply_mask(img=bands['red_edge'], mask=composed_mask, mask_color='black')
            for index_name, formula in VEG_INDEX_FORMULAS.items():
                try:
                    num_params = len(inspect.signature(formula).parameters)
                    if num_params == 2:
                        params = [nir, red]
                    elif num_params == 3:
                        if index_name == "SAVI":
                            params = [nir, red, L]
                        elif index_name == "OSAVI":
                            params = [nir, red, soil_factor]
                        elif index_name in ["RECI", "NDREI", "MTCI"]:
                            params = [nir, red_edge, epsilon]
                        elif index_name in ["GNDVI", "CIGREEN", "SIPI2", "CVI2"]:
                            params = [nir, green, epsilon]
                        else:
                            params = [nir, red, epsilon]
                    elif num_params == 4:
                        params = [nir, red, green, red_edge]
                    else:
                        params = [nir, red, green, red_edge, epsilon]
                    index_image = formula(*params)
                    index_image = np.nan_to_num(index_image, nan=0.0, posinf=0.0, neginf=0.0)
                    max_val = np.nanmax(index_image)
                    min_val = np.nanmin(index_image)
                    avg_val = np.nanmean(index_image[index_image != 0]) if np.any(index_image != 0) else 0
                    pos_avg_val = np.nanmean(index_image[index_image > 0]) if np.any(index_image > 0) else 0
                    self.plant_stats[plant_name][f"{index_name} (Maximum)"] = max_val
                    self.plant_stats[plant_name][f"{index_name} (Minimum)"] = min_val
                    self.plant_stats[plant_name][f"{index_name} (Average)"] = avg_val
                    self.plant_stats[plant_name][f"{index_name} (Positive Average)"] = pos_avg_val
                    # Generate visualization (if needed, can be commented out)
                    fig, ax = plt.subplots()
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    im = ax.imshow(index_image, vmin=ndvi_min, vmax=ndvi_max, cmap=cm.get_cmap("RdYlGn"))
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    plt.suptitle(index_name)
                    ax.axis('off')
                    fig.tight_layout(pad=0)
                    ax.margins(0)
                    canvas = FigureCanvas(fig)
                    canvas.draw()
                    image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.plants[plant_name][f"{index_name.lower()}_image"] = (image_from_plot, f"{index_name} Image")
                    plt.close(fig)
                except Exception:
                    continue
    def stitch_color_images(self, batch):
        for plant_name in batch:
            input_images = [ci for ci, _ in self.plants[plant_name]['color_images']]
            stitched_image = image_stitching(input_images)
            self.plants[plant_name]['stitched_image'] = (stitched_image, 'Whole Plant Image')

    def calculate_connected_components(self, batch):
        for plant_name in batch:
            gray_image, binary = CCA_Preprocess(self.plants[plant_name]['stitched_image'][0], k=self.variable_k)
            preprocessed_image = np.repeat(np.expand_dims(binary, axis=-1), 3, axis=-1) * self.plants[plant_name]['stitched_image'][0]
            cca_image = 255 * (preprocessed_image - preprocessed_image.min()) / (preprocessed_image.max() - preprocessed_image.min())
            cca_image = cca_image.astype(np.uint8)
            self.plants[plant_name]['cca_image'] = (cca_image, 'Background Separated Using Connected Component Analysis')

    def run_segmentation(self, batch):
        try:
            input_images = [self.plants[p]['stitched_image'][0] for p in batch if p in self.plants]
        except KeyError:
            return
        try:
            results = self.segmentation_model.predict(input_images, conf=self.pipeline_config.get('segmentation_confidence', 0.5), device=self.device)
        except Exception:
            return
        for i, plant_name in enumerate(batch):
            try:
                result = results[i]
                if result and hasattr(result, 'masks') and hasattr(result.masks, 'data'):
                    max_masks = self.pipeline_config.get('max_masks', 4)
                    if result.masks.data.shape[0] > max_masks:
                        result.masks.data = result.masks.data[:max_masks]
                    mask = preprocess_mask(result.masks.data)
                    binary_mask_np = generate_binary_mask(mask)
                    original_image = self.plants[plant_name]['stitched_image'][0]
                    overlayed_image = overlay_mask_on_image(binary_mask_np, original_image)
                    self.plants[plant_name]['segmented_image'] = (overlayed_image, 'Background Separated Using Image Segmentation')
                else:
                    continue
            except IndexError:
                continue
            except Exception:
                continue

    def run_segmentation_corn(self, batch):
        input_images = [self.plants[p]['stitched_image'][0] for p in batch]
        plant_names = batch
        results = self.segmentation_model.predict(input_images, conf=self.pipeline_config['segmentation_confidence'], device=self.device)
        for i in range(len(results)):
            result = results[i]
            if result:
                if result.masks.data.shape[0] > 4:
                    result.masks.data = result.masks.data[:4]
                mask = preprocess_mask(result.masks.data)
                binary_mask_np = generate_binary_mask(mask)
                overlayed_image = overlay_mask_on_image(binary_mask_np, self.plants[plant_names[i]]['stitched_image'][0])
                self.plants[plant_names[i]]['segmented_image'] = (overlayed_image, 'Background Separated Using Image Segmentation')

    def run_segmentation_yolo(self, batch):
        input_images = [self.plants[p]['stitched_image'][0] for p in batch]
        plant_names = batch
        results = self.segmentation_model.predict(input_images, conf=self.pipeline_config['segmentation_confidence'], device=self.device)
        for i in range(len(results)):
            result = results[i]
            if result:
                if result.masks.data.shape[0] > 4:
                    result.masks.data = result.masks.data[:4]
                mask = preprocess_mask(result.masks.data)
                binary_mask_np = generate_binary_mask(mask)
                overlayed_image = overlay_mask_on_image(binary_mask_np, self.plants[plant_names[i]]['stitched_image'][0])
                self.plants[plant_names[i]]['segmented_image'] = (overlayed_image, 'Background Separated Using Image Segmentation')

    def calculate_plant_phenotypes(self, batch):
        for plant_name in batch:
            try:
                phenotypes = get_plant_phenotypes(self.plants[plant_name]['segmented_image'][0], offset=self.offset)
            except Exception:
                continue
            if not phenotypes:
                continue
            self.plant_stats[plant_name]['Height'] = phenotypes.get('Plant Height (cm)', 0)
            self.plant_stats[plant_name]['Width'] = phenotypes.get('Plant Width (cm)', 0)
            self.plant_stats[plant_name]['Area'] = phenotypes.get('Plant Area (square cm)', 0)
            self.plant_stats[plant_name]['Perimeter'] = phenotypes.get('Plant Perimeter (cm)', 0)
            self.plant_stats[plant_name]['Solidity'] = phenotypes.get('Plant Solidity', 0)
            self.plant_stats[plant_name]['Number of Branches'] = phenotypes.get('Number of Branches', 0)
            self.plant_stats[plant_name]['Number of Leaves'] = phenotypes.get('Number of Leaves', 0)

    def calculate_tips_and_branches(self, batch):
        for plant_name in batch:
            gray_image = cv2.cvtColor(self.plants[plant_name]['segmented_image'][0], cv2.COLOR_RGB2GRAY)
            skeleton = pcv.morphology.skeletonize(mask=gray_image)
            tips = pcv.morphology.find_tips(skel_img=skeleton, mask=None, label=plant_name)
            branches = pcv.morphology.find_branch_pts(skel_img=skeleton, mask=None, label=plant_name)
            tips_and_branches = np.zeros_like(skeleton)
            tips_and_branches[tips > 0] = 255
            tips_and_branches[branches > 0] = 128
            kernel = np.ones((5, 5), np.uint8)
            tips = cv2.dilate(tips, kernel, iterations=1)
            branches = cv2.dilate(branches, kernel, iterations=1)
            tips_and_branches = cv2.dilate(tips_and_branches, kernel, iterations=1)
            self.plants[plant_name]['tips'] = (tips, 'Plant Tips')
            self.plants[plant_name]['branches'] = (branches, 'Plant Branch Points')
            self.plants[plant_name]['tips_and_branches'] = (tips_and_branches, 'Plant Tips and Branch Points')
            self.plants[plant_name]['gray_image'] = (gray_image, 'Gray Segmented Image')
            self.plants[plant_name]['skeleton'] = (skeleton, 'Morphology Skeleton')

    def calculate_sift_features(self, batch):
        for plant_name in batch:
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(self.plants[plant_name]['skeleton'][0], None)
            sift_image = cv2.drawKeypoints(self.plants[plant_name]['skeleton'][0], kp, des)
            self.plants[plant_name]['sift_features'] = (sift_image, 'SIFT Features')

    def calculate_lbp_features(self, batch):
        for plant_name in batch:
            lbp = local_binary_pattern(self.plants[plant_name]['gray_image'][0], self.LBP_n_points, self.LBP_radius)
            self.plants[plant_name]['lbp_features'] = (lbp, 'Local Binary Patterns')

    def calculate_hog_features(self, batch):
        for plant_name in batch:
            # Get the stored grayscale image for this plant
            gray_image = self.plants[plant_name]['gray_image'][0]
            # Convert to grayscale if the image has three channels
            if len(gray_image.shape) == 3:
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
            # Compute HOG features using parameters from self.pipeline_config
            fd, hog_image = hog(
                gray_image,
                orientations=self.pipeline_config.get('HOG_orientations', 9),
                pixels_per_cell=(
                    self.pipeline_config.get('HOG_pixels_per_cell', 16),
                    self.pipeline_config.get('HOG_pixels_per_cell', 16)
                ),
                cells_per_block=(
                    self.pipeline_config.get('HOG_cells_per_block', 1),
                    self.pipeline_config.get('HOG_cells_per_block', 1)
                ),
                visualize=True
            )
            # Rescale the HOG visualization image to a 0-255 range
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, self.pipeline_config.get('HOG_orientations', 9)))
            hog_image_rescaled = np.asarray(hog_image_rescaled * 255).astype(np.uint8)
            self.plants[plant_name]['hog_features'] = (hog_image_rescaled, 'Histogram of Oriented Gradients')
    def clear(self):
        self.service_type = 0
        self.input_folder_path = None
        self.output_folder_path = None
        self.show_raw_images = False
        self.show_color_images = False
        del self.plant_paths
        del self.plant_stats
        shutil.rmtree(self.interm_result_folder)

    def get_plant_statistics_df(self):
        plant_names = self.get_plant_names()
        df_dict = {'Plant_Name': plant_names}
        for item in self.statistics_items:
            df_dict[item] = [round(self.plant_stats[plant_name][item], 2) for plant_name in plant_names]
        return pd.DataFrame(df_dict)

    @staticmethod
    def make_dir(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

    def show_vegetation_index_matrix(self, plant_name, vi_name):
        vi_key = f"{vi_name.lower()}_image"
        if plant_name not in self.plants or vi_key not in self.plants[plant_name]:
            return
        vi_matrix = self.plants[plant_name][vi_key][0]
        plt.figure(figsize=(8, 6))
        plt.imshow(vi_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label=f"{vi_name} Value")
        plt.title(f"{vi_name} Pixel Matrix for {plant_name}")
        plt.axis("off")
        plt.show()

    def calculate_single_plant_vi_correlation(self, plant_name, output_folder="PlantAnalysisResults"):
        if plant_name not in self.plants:
            return None
        vi_data = {}
        for vi_name in VEG_INDEX_FORMULAS.keys():
            vi_key = f"{vi_name.lower()}_image"
            if vi_key in self.plants[plant_name]:
                vi_image = self.plants[plant_name][vi_key][0]
                vi_data[vi_name] = vi_image.flatten()
        vi_df = pd.DataFrame(vi_data)
        os.makedirs(output_folder, exist_ok=True)
        vi_df.to_csv(os.path.join(output_folder, f"{plant_name}_VI_Data_before_remove.txt"), sep="\t", index=False)
        vi_df_filtered = vi_df[((vi_df != 255).all(axis=1) & (vi_df != 254).all(axis=1))]  # type: ignore
        vi_df_filtered.to_csv(os.path.join(output_folder, f"{plant_name}_VI_Data_after_remove.txt"), sep="\t", index=False)
        correlation_matrix = vi_df_filtered.corr()
        correlation_matrix.to_csv(os.path.join(output_folder, f"{plant_name}_VI_Correlation.csv"))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, mask=mask, cmap="coolwarm", cbar=True, annot=False, linewidths=0.5)
        plt.title("Correlation Heatmap (Lower Triangle)")
        plt.savefig(os.path.join(output_folder, f"{plant_name}_correlation_lower_triangle.png"), dpi=300, bbox_inches='tight')
        plt.close()
        filtered_correlation_matrix = correlation_matrix.copy()
        filtered_correlation_matrix[np.abs(filtered_correlation_matrix) < 0.75] = np.nan
        plt.figure(figsize=(12, 10))
        sns.heatmap(filtered_correlation_matrix, mask=mask, cmap="coolwarm", cbar=True, annot=False, linewidths=0.5)
        plt.title("Correlation Heatmap (Only |r| >= 0.75)")
        plt.savefig(os.path.join(output_folder, f"{plant_name}_correlation_filtered.png"), dpi=300, bbox_inches='tight')
        plt.close()
        return correlation_matrix

    def save_results(self, folder_path):
        self.make_dir(folder_path)
        if not hasattr(self, 'plants') or not self.plants:
            return
        result_dict = {'statistics_items': self.statistics_items, 'statistics_units': self.statistics_units}
        plant_names = self.get_plant_names()
        with open(os.path.join(folder_path, 'plants_features_and_statistics.txt'), 'w') as f:
            for plant_name in plant_names:
                if plant_name not in self.plants:
                    continue
                result_dict[plant_name] = {}
                for item in self.statistics_items:
                    if item in self.plant_stats[plant_name]:
                        f.write(f"{plant_name},{item},{self.plant_stats[plant_name][item]}\n")
                        result_dict[plant_name][item] = self.plant_stats[plant_name][item]
        flattened_data = []
        for plant_name, plant_data in result_dict.items():
            if plant_name in ['statistics_items', 'statistics_units']:
                continue
            flat_dict = {'plant_name': plant_name}
            if isinstance(plant_data, dict):
                for key, value in plant_data.items():
                    flat_dict[key] = value
            else:
                flat_dict['data'] = plant_data
            flattened_data.append(flat_dict)
        result_df = pd.DataFrame(flattened_data)
        result_df.to_excel(os.path.join(folder_path, 'plants_features_and_statistics.xlsx'), index=False)
        with open(os.path.join(folder_path, 'plant_features_and_statistics.json'), 'w') as fp:
            json.dump(result_dict, fp, indent=4)

        # For each plant, create subfolders for Color Images, Vegetation Indices, and Correlation Matrix
        for plant_name in plant_names:
            if plant_name not in self.plants:
                continue
            plant_folder = os.path.join(folder_path, plant_name)
            self.make_dir(plant_folder)
            color_images_folder = os.path.join(plant_folder, 'Color_Images')
            self.make_dir(color_images_folder)
            veg_indices_folder = os.path.join(plant_folder, 'Vegetation_Indices')
            self.make_dir(veg_indices_folder)
            corr_matrix_folder = os.path.join(plant_folder, 'Correlation_Matrix')
            self.make_dir(corr_matrix_folder)
            # Save color images
            color_images = self.get_color_images(plant_name)
            for image, image_name in color_images:
                save_path = os.path.join(color_images_folder, image_name.split('.')[0] + '.jpg')
                cv2.imwrite(save_path, image)
            # Save all images that have a key ending with "_image"
            for key in self.plants[plant_name]:
                if key.endswith("_image"):
                    base_name = key.replace('_image', '').upper() + "_Image.jpg"
                    # If the key is one of the vegetation indices, save in veg_indices_folder;
                    # if it contains "CORRELATION", save in corr_matrix_folder;
                    # otherwise, save in the main plant folder.
                    veg_keys = [k.lower() + "_image" for k in VEG_INDEX_FORMULAS.keys()]
                    if key in veg_keys:
                        target_folder = veg_indices_folder
                    elif "correlation" in key.lower():
                        target_folder = corr_matrix_folder
                    else:
                        target_folder = plant_folder
                    image_path = os.path.join(target_folder, base_name)
                    if self.plants[plant_name][key][0] is not None:
                        cv2.imwrite(image_path, self.plants[plant_name][key][0])
        # End of save_results

