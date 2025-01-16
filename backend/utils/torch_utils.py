import numpy as np
import cv2
import torch
import torch.nn as nn
import pickle
from models.feedforward import SimpleNN
from sklearn.preprocessing import MinMaxScaler

def calculate_fractal_dimension(contour, scales=[2, 4, 8, 16]):
        """Calculate the fractal dimension of a contour using coastline approximation.
        Args:
        contour (numpy array): Contour points.
        scales (list): List of segment lengths to approximate the contour.
    
        Returns:
        float: Fractal dimension of the contour."""
        try:
            if len(contour) ==0:
                return 0
            perimeter=cv2.arcLength(contour,True)
            if perimeter==0:
                return 0
            lengths=[]
            for scale in scales:
                downsampled=contour[::scale]
                approx_length=np.sum(np.sqrt(np.diff(downsampled.squeeze())**2))
                lengths.append(approx_length) 
            if lengths:
                log_scales=-np.log(scales[:len(lengths)])
                log_lengths=np.log(np.array(lengths) +1e-10 )
                slope,_=np.polyfit(log_scales,log_lengths,1)
                return slope+1
            else:
                return 0   
            
                
            
        except Exception as e:
            print(f"error in calculate fractal dimension:{e}")
            return 0
def stats_calculator(property):
        try:
            if isinstance(property, (list, np.ndarray)):
                property = [p for p in property if np.isscalar(p)]  # Filter non-scalars
                property = np.array(property,dtype=np.float64)
                if property.size>0:
                    mean_property=np.mean(property)
                    std_property=np.std(property)
                    max_property=max(property)
                else:
                    mean_property,std_property,max_property=0.0,0.0,0.0
            else:
                mean_property, std_property, max_property = 0.0, 0.0, 0.0            
            return mean_property,std_property,max_property
        except Exception as e:
            print(f"Error in stats_calculator: {e}")
            return 0.0,0.0,0.0
def transform_image(images_bytes):
    
      
            
    img_array=np.frombuffer(images_bytes,dtype=np.uint8)
    image=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    if len(image.shape) == 2:  # Check if the image is already grayscale
        grayscale_image = image
    else:    
        grayscale_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    threshold_value = 127  # Adjust this value as necessary
    max_value = 255
    _, thresholded_image = cv2.threshold(grayscale_image, threshold_value, max_value, cv2.THRESH_BINARY_INV)

    contours,_=cv2.findContours(thresholded_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(grayscale_image)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(grayscale_image, grayscale_image, mask=mask)
    nuclei_values = masked_image[masked_image > 0]
    try:
        mean_texture,std_dev_texture,max_texture_value = stats_calculator(nuclei_values)
    except Exception as e:
        print(f"Error calculating texture stats: {e}")
        mean_texture, std_dev_texture, max_texture_value = 0.0, 0.0, 0.0
    id_value=123456    
    radii=[]
    features=[id_value]
    compactness_values=[]
    concavity_values=[]
    areas=[]
    perimeters=[]
    concave_points_list = []
    symmetry_values=[]
    fractal_dimensions=[]
    for contour in contours:
         
         
        if len(contour) > 1:  # Valid contour
            # Calculate contour area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            moments=cv2.moments(contour)
            try:
                fractal_dim=calculate_fractal_dimension(contour)
            except Exception as e:
                print(f"error calculating fractal dimension:{e}")
                fractal_dim=0   
            fractal_dimensions.append(fractal_dim)
            if moments['m00'] !=0:
                centroid_x=int(moments['m10']/moments['m00'])
                centroid_y=int(moments['m01']/moments['m00'])
                reflected_contour=[]
                for point in contour:
                    x,y=point[0]
                    reflected_contour.append([[2*centroid_x-x,y]])
                reflected_contour=np.array(reflected_contour)   
                original_polygon=cv2.convexHull(contour)
                reflected_polygon=cv2.convexHull(reflected_contour)
                _,intersection_area=cv2.intersectConvexConvex(original_polygon,reflected_polygon)
                try:
                     # Ensure `area` and `intersection_area` are scalars or properly reduced
                    if isinstance(area, np.ndarray):
                        area = np.sum(area)  # Replace with .item() if array is single-valued
                    if isinstance(intersection_area, np.ndarray):
                        intersection_area = np.sum(intersection_area)

                    if area>0 and intersection_area is not None:
                        max_value = 1e6  # Define a maximum limit to prevent overflows
                        symmetry=intersection_area/area
                        intersection_area = min(intersection_area, max_value)
                        # Use safe division to prevent overflow or division by zero
                        
                        symmetry = intersection_area / (area + 1e-10)
                        
                    else:
                        symmetry=0
                    symmetry_values.append(symmetry)
                except cv2.error as e:
                    print(f"Error calculating symmetry: {e}")
                    symmetry = 0
                symmetry_values.append(symmetry)    
            else:
                symmetry_values.append(0)            
            if area > 0:  # Avoid division by zero
                # Minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * (radius ** 2)

                # Append radius
                radii.append(radius)

                # Compactness
                if area>0:
                    compactness = (perimeter ** 2 / area) - 1
                else:
                    compactness = 0    
                compactness_values.append(compactness)

                # Concavity (severity of concave portions)
                if circle_area>0 and area>0:

                    concavity = area / circle_area
                else:
                    concavity=0    
                concavity_values.append(concavity)

                # Append contour area and perimeter
                areas.append(area)
                perimeters.append(perimeter)

        else:  # For contours with zero area
            radii.append(0)
            compactness_values.append(0)
            concavity_values.append(0)
            areas.append(0)
            perimeters.append(0)

        
        # Concave points
        # Simplify the contour to ensure it's valid
        epsilon = 0.04 * cv2.arcLength(contour, True)  # Approximation accuracy factor
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        try:
            if len(approx_contour) >=4:
                hull = cv2.convexHull(approx_contour, returnPoints=False)
                hull[::-1].sort(axis=0)
                if len(hull) >=3: 
                    defects = cv2.convexityDefects(contour, hull)
                else:
                    defects=None
            else:
                defects=None
        except cv2.error as e:
            print(f"Error in convexityDefects:{e}")
            defects=None        
        
        if defects is not None:
            concave_points = defects.shape[0]
        else:
            concave_points = 0
        concave_points_list.append(concave_points)

    else:  # For invalid contours
        radii.append(0)
        compactness_values.append(0)
        concavity_values.append(0)
        areas.append(0)
        perimeters.append(0)
        concave_points_list.append(0)
        symmetry_values.append(0)
        fractal_dimensions.append(0)

 

      
    symmetry_mean, symmetry_std, symmetry_max = stats_calculator(symmetry_values)
    mean_compactness,std_compactness,max_compactness=stats_calculator(compactness_values)        
    mean_concavity,std_concavity,max_concavity= stats_calculator(concavity_values)
    mean_area,std_area,max_area=stats_calculator(areas)
    mean_perimeter,std_perimeter,max_perimeter=stats_calculator(perimeters)
    mean_concave_points,std_concave_points,max_concave_points=stats_calculator(concave_points_list)
    mean_fractal_dimension,std_fractal_dimension,max_fractal_dimension=stats_calculator(fractal_dimensions)
    if radii:
        local_smoothness = []
        window_size = 5 
        for i in range(len(radii) - window_size + 1):
            window = radii[i:i + window_size]
            local_std_dev = np.std(window)  # Calculate standard deviation of the local window
            local_smoothness.append(local_std_dev)

        
        mean_radius,std_radius ,max_radius = stats_calculator(radii) 
        mean_smoothness,std_smoothness,max_smoothness=stats_calculator(local_smoothness)
        
    
    else:
    # If radii cannot be calculated, append default values
        mean_radius,std_radius,max_radius,mean_smoothness,std_smoothness,max_smoothness=0.0,0.0,0.0,0.0,0.0,0.0
    features.extend([mean_radius,std_radius,max_radius,mean_texture,std_dev_texture,max_texture_value,mean_area,std_area,max_area,mean_perimeter,std_perimeter,max_perimeter,mean_smoothness,std_smoothness,max_smoothness,mean_compactness,std_compactness,max_compactness,mean_concavity,std_concavity,max_concavity,mean_concave_points,std_concave_points,max_concave_points,symmetry_mean,symmetry_std,symmetry_max,mean_fractal_dimension,std_fractal_dimension,max_fractal_dimension])
    
    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return feature_tensor


# Load the model
with open('.//models//trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)    
def get_prediction(tensor):
    # Make sure the tensor is on the correct device
    try:
        tensor = tensor.to(device)
    
    # Get the output from the model
        outputs = model(tensor)
    
    # Apply torch.max to get the predicted class (index of max logit)
        _, predicted = torch.max(outputs.data, 1)
    
    # Return 'M' for Malignant (class 1) and 'N' for Benign (class 0)
        if predicted.item() == 1:
            return 'M'  # Malignant
        else:
            return 'N'  # Benign
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
