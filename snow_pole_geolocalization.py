import cv2
import numpy as np
import pandas as pd
import matplotlib
from pyproj import Transformer
import torch
from bagpy import bagreader
from ouster import client
from geographiclib.geodesic import Geodesic
import csv
import math
from geopy.distance import distance
from geopy.point import Point
from matplotlib.animation import FuncAnimation
import json
from geopy.distance import distance as geopy_distance
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Geod
import pickle
import threading
from PIL import ImageGrab
import time
from geoloc_utils import *
import numpy as np
from bagpy import bagreader
from scipy.interpolate import interp1d
import contextily as ctx
import numpy as np



model = load_custom_model('./model/pole_best_signal.pt', confidence_threshold=0.7, backend='TkAgg') # Load the custom YOLOv5 model

matplotlib.pyplot.matplotlib.pyplot.ion() # Initialize matplotlib for interactive mode

# Setup the figure and axis for dynamic plotting
fig, ax = matplotlib.pyplot.subplots(figsize=(10, 8)) # Set figure size
ax.set_title('GNSS Data Visualization') # Set title
ax.set_xlabel('Easting') # Set x-axis label
ax.set_ylabel('Northing') # Set y-axis label



dataRef_north, dataRef_east, min_north, max_north, min_east, max_east, dataRef_latitudes, dataRef_longitudes, min_latitude, max_latitude, min_longitude, max_longitude= load_gnss_data_and_calculate_utm_range('Groundtruth_pole_location_at_test_site_E39_Hemnekjølen.csv') # Load reference GNSS data and calculate UTM range
print(f"min_north: {min_north}, max_north: {max_north}, min_east: {min_east}, max_east: {max_east}")

ground_truth_poles = [(dataRef_east, dataRef_north) for dataRef_east, dataRef_north in zip(dataRef_east, dataRef_north)] # Create a list of ground truth pole coordinates
unused_poles = ground_truth_poles.copy() # Create a copy of ground truth poles for unused poles tracking

# Plot Reference GNSS data statically
ax.scatter(dataRef_east, dataRef_north, c='green', label='Reference GNSS Data', marker='x') # 'gx' for green x markers


# Placeholder for dynamically updated object GNSS data plot
corresponding_gnss_plot = ax.scatter([], [], c='blue', label='Vehicle GNSS Data', marker='^') # 'bo' for blue dots
detected_objects_plot, = ax.plot([], [], 'ro', label='Detected Objects GNSS Data')  # 'ro' for red dots

# Add the map background
ctx.add_basemap(ax, crs='EPSG:32633', source=ctx.providers.OpenStreetMap.Mapnik) # Using UTM Zone 33N for the map background

# Display legend
ax.legend() # Show legend
fig.canvas.draw() # Draw the canvas
matplotlib.pyplot.pause(0.001)  # Pause briefly to ensure the plot displays

metadata, xyzlut = load_lidar_configuration('Trip068.json', client) # Load LiDAR configuration

object_gnss_list = [] # List to store object GNSS data
corresponding_gnss_data = [] # List to store corresponding GNSS data
object_gnss_data = [] # List to store object GNSS data
object_gnss_data_1 = [] # List to store object GNSS data
object_gnss_data_2 = [] # List to store object GNSS data
min_distance_list = [] # List to store minimum distances
nearest_poles = [] # List to store nearest poles  
azimuth_gnss_list = [] # List to store azimuth values
new_distance_list = [] # List to store new distances
heading_list = [] # List to store heading values
vehicle_lats = [] # List to store vehicle latitudes
vehicle_lons = [] # List to store vehicle longitudes
vehicle_eastings = [] # List to store vehicle eastings
vehicle_northings = [] # List to store vehicle northings
proj_latlon = "EPSG:4326"  # WGS84 latitude/longitude
proj_utm33 = "EPSG:32633"  # UTM Zone 33N 


csv_data = [] # List to store data for CSV

vehicle_gnss_average_data_vec = [] # Initialize lists to store data

# Define the GNSS and LiDAR sensor offsets
# left_gnss_offset = np.array([-0.32, -0.51, 1.24])
# right_gnss_offset = np.array([-0.32, 0.51, 1.24])
gnss_offset = np.array([-0.32, 0.0, 1.24]) # average of left and right gnss offset
lidar_offset = np.array([0.7, 0.0, 1.8])  # LiDAR sensor offset relative to the same reference point
gnss_to_lidar_offset = gnss_offset - lidar_offset  # Calculate GNSS to LiDAR offset
print('gnss_to_lidar_offset', gnss_to_lidar_offset) # Print GNSS to LiDAR offset




signal_image_data, nearir_image_data, reflec_image_data, range_image_data, vehicle_left_gnss_data, vehicle_right_gnss_data, imu_data, point_cloud_data, vehicle_heading_data, timestamps_signal, timestamps_nearir, timestamps_reflec, timestamps_range, timestamps_left_gnss, timestamps_right_gnss, timestamps_imu = process_ros_bag_data('./2024-02-28-12-59-51_no_unwanted_topics.bag') # Process ROS bag data

# first ten time stamps
print(f'timestamps_signal: {timestamps_signal[:10]}') # print first ten timestamps of signal
print(f'timestamps_reflec: {timestamps_reflec[:10]}') # print first ten timestamps of reflec
print(f'timestamps_left_gnss: {timestamps_left_gnss[:10]}') # print first ten timestamps of left gnss
print(f'timestamps_right_gnss: {timestamps_right_gnss[:10]}') # print first ten timestamps of right gnss
print(f'timestamps_range: {timestamps_range[:10]}') # print first ten timestamps of range

print(f'length of range_image_data: {len(range_image_data)}') # print length of range_image_data

print(f'length of vehicle_left_gnss_data: {len(vehicle_left_gnss_data)}') # print length of vehicle_left_gnss_data

print(f'length of vehicle_right_gnss_data: {len(vehicle_right_gnss_data)}') # print length of vehicle_right_gnss_data





# Process data and calculate vehicle's latitude and longitude
for i, _ in enumerate(range_image_data): 
    print(f'Processing frame: {i}')
    print(f'vehicle_left_gnss_data: {vehicle_left_gnss_data[i]}')
    print(f'vehicle_right_gnss_data: {vehicle_right_gnss_data[i]}')
    lat = (vehicle_left_gnss_data[i][0] + vehicle_right_gnss_data[i][0]) / 2 # Average latitude from left and right GNSS
    lon = (vehicle_left_gnss_data[i][1] + vehicle_right_gnss_data[i][1]) / 2 # Average longitude from left and right GNSS
    vehicle_lats.append(lat) # Append latitude to the list
    vehicle_lons.append(lon)  # Append longitude to the list
    # Convert latitude and longitude to UTM33 coordinates
    easting, northing = transform_coordinates(proj_latlon, proj_utm33, lon, lat) # Transform coordinates from lat/lon to UTM33
    vehicle_eastings.append(easting) # Append easting to the list
    vehicle_northings.append(northing) # Append northing to the list     
    # print(f'Vehicle GNSS: Easting: {easting}, Northing: {northing}')
print(f'vehicle_lats: {vehicle_lats[:10]}') # print first ten vehicle latitudes
print(f'vehicle_lons: {vehicle_lons[:10]}') # print first ten vehicle longitudes



predicted_latitude, predicted_longitude = kriging_interpolation(timestamps_range, timestamps_left_gnss[:-2], vehicle_lats, vehicle_lons) # Perform Kriging interpolation to predict latitude and longitude
print(f' length of timestamps_left_gnss: {len(timestamps_left_gnss)}') # print length of timestamps_left_gnss
print(f' length of timestamps_range: {len(timestamps_range)}') # print length of timestamps_range


# Iterate over each image/frame for object detection and geolocation
for i, (range_image, timestamp) in enumerate(zip(range_image_data, timestamps_range)): # Iterate through range images and their timestamps

    vehicle_easting, vehicle_northing = transform_coordinates(proj_latlon, proj_utm33, predicted_longitude[i], predicted_latitude[i]) # Transform predicted lat/lon to UTM33 coordinates
    if min_north <= vehicle_northing <= max_north and min_east <= vehicle_easting <= max_east: # Check if the vehicle GNSS data is within the reference range       
        print(f'Processing frame: {i}') # Print the current frame being processed
        # Calculate heading and draw on the plot
        if i > 0:  # Ensure there is a previous point to calculate heading

            range_image_vis = (range_image - range_image.min()) / (range_image.max() - range_image.min()) # Normalize range image for visualization
            # create rgb image of range_image using range_image as all 3 channels 
            range_image_vis = np.stack((range_image_vis, range_image_vis, range_image_vis), axis=-1) # Stack range image into 3 channels for RGB
            cv2.imshow('range_image', range_image_vis) # Display the range image using cv2
            xyz = xyzlut(range_image) # Generate point cloud from range image using lookup table
            # print(f'xyz table pointcloud shape: {xyz.shape}')
            xyz = xyz * 4 # mutiply by 4(mm) to get the actual xyz values
    
            # range_lookup_table, range_vals_scaled_lookup_table = display_range_from_xyz(xyz)
            # Display the range image using cv2
            # cv2.imshow('Range Image from lookup table point cloud ', range_lookup_table)

            
            rgb_image = np. stack((signal_image_data[i], signal_image_data[i], signal_image_data[i]), axis=-1) # Create RGB image by stacking signal image data into 3 channels
            # rgb_image = np. stack((signal_image_data[i], nearir_image_data[i], reflec_image_data[i]), axis=-1)

            
            rgb_image = ((rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255).astype(np.uint8) # Normalize RGB image to 0-255 and convert to uint8
            results = model(rgb_image) # Perform object detection using the loaded model
            rgb_annotated_img = results.render()[0] # Render the detection results on the RGB image
            bboxes = results.xyxy[0] # Extract bounding boxes from the detection results

            for bbox in bboxes: # Iterate through each detected bounding box
                
                print(f'sequence number used for geo localization: {i}') # Print the sequence number for geolocation
                vehicle_gnss_average_data_vec.append((vehicle_easting, vehicle_northing)) # Append vehicle GNSS data to the list
                x_min, y_min, x_max, y_max = map(int, bbox[:4]) # Extract bounding box coordinates
                bbox_range = range_image_data[i][y_min:y_max, x_min:x_max] # Extract the range values within the bounding box

                global_x, global_y, nearest_distance = extract_nearest_point_in_bounding_box_region(x_min, x_max, y_min, y_max, rgb_image, range_image_data, i) # Extract the nearest point within the bounding box region
                print(f'Object: Global X: {global_x}, Global Y: {global_y} for range_image_data') # Print the global coordinates of the detected object

                # Check if the values are None and skip the rest of the calculations if so
                if global_x is None or global_y is None or nearest_distance is None:
                  print('Skipping calculations as no valid nearest point found.')
                  continue  # Skip to the next iteration of the loop

                center_xyz   = xyz[int(global_y), int(global_x), :] # Get the XYZ coordinates of the nearest point
                print('center_xyz before transformation', center_xyz) # Print the center XYZ coordinates before transformation
                # xyz_distance = np.linalg.norm(center_xyz)
                # print('xyz_distance_table_pointcloud in mts', xyz_distance)

                nearest_distance_range = (range_image_data[i][global_y, global_x] / (1000))*4 # Convert to meters
                print('nearest distance in mts from range image', nearest_distance_range) # Print the nearest distance from the range image

                x, y, z = center_xyz

                # calculate the distance from the center_xyz to the gnss_offset
                new_distance = calculate_distance(center_xyz, gnss_to_lidar_offset) # Calculate new distance from the GNSS location
                print('new_distance from the  gnss location w.r.to gnss', new_distance) # Print the new distance from the GNSS location
                # append the new_distance to the list
                
                new_distance_list.append(new_distance) # Append new distance to the list


                easting_vec, northing_vec = zip(*vehicle_gnss_average_data_vec) # Unzip vehicle GNSS data into separate lists

                easting1, northing1 = vehicle_eastings[i - 1], vehicle_northings[i - 1] # Previous vehicle GNSS coordinates
                easting2, northing2 = vehicle_easting, vehicle_northing # Current vehicle GNSS coordinates
                heading = calculate_vehicle_heading_from_two_utm(easting1, northing1, easting2, northing2) # Calculate vehicle heading from two UTM coordinates
                
                heading_list.append(heading) # Append heading to the list
                end_easting, end_northing = calculate_vehicle_heading_direction_utm(easting1, northing1, heading) # Calculate the end point based on heading


                azimuth_gnss, elevation_gnss = calculate_azimuth_elevation_from_gnss(x, y, z, gnss_to_lidar_offset) # Calculate azimuth and elevation from GNSS
                print(f'Azimuth: {azimuth_gnss}, Elevation: {elevation_gnss} from the GNSS sensor') # Print azimuth and elevation from GNSS sensor

                print(f'Vehcile Heading from GNSS: {heading}') # Print vehicle heading from GNSS
                # print vehicle heading
                # vehicle_heading = math.degrees(vehicle_heading_data[i])
                # print(f'Vehicle Heading from the vehicle: {vehicle_heading}')

                azimuth_gnss = -(azimuth_gnss)
                # append the azimuth_gnss to the list
                azimuth_gnss_list.append(azimuth_gnss)


                target_easting, target_northing, adjusted_azimuth = local_to_utm33_with_offset_proj(predicted_latitude[i], predicted_longitude[i], heading, (azimuth_gnss), (new_distance))

                print(f'Target GNSS: Easting: {target_easting}, Northing: {target_northing}')
                
                


                object_gnss_data.append((target_easting, target_northing))
                # Extract latitudes and longitudes for all detected objects
                target_easting_vec, target_northing_vec = zip(*object_gnss_data)



                min_distance = float('inf')
                nearest_pole = None


                for pole_easting, pole_northing in ground_truth_poles:
                    dist = np.sqrt((target_easting - pole_easting) ** 2 + (target_northing - pole_northing) ** 2)
                    if dist < min_distance:
                        min_distance = dist
                        nearest_pole = (pole_easting, pole_northing)


                if nearest_pole is not None:
                    nearest_poles.append(nearest_pole)


                ground_truth_easting, ground_truth_northing = nearest_pole
                print(f'Nearest Pole: {nearest_pole}, Distance: {min_distance}')
                # append the min_distance to the list
                if min_distance > 10:
                    print(f'Nearest Pole: {nearest_pole}, Distance: {min_distance}')
                    print('Skipping the pole as distance is greater than 10 meters')
                    continue
                min_distance_list.append(min_distance)
                # calculate the avegage of min_distance
                average_min_distance = np.mean(min_distance_list)
                print(f'Average Min Distance: {average_min_distance}')

                # ax.annotate('', xy=(end_easting, end_northing), xytext=(easting1, northing1), arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=3))
                # ax.annotate('', xy=(end_northing, end_easting), xytext=(northing1, easting1), arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=3))


                print(f'Target GNSS: Easting: {target_easting}, Northing: {target_northing}')
                

                ####################Update the dynamic plot with new data #########################
                # # chnage marker color to light blue

                # corresponding_gnss_plot = ax.scatter(easting_vec, northing_vec, c='cornflowerblue', label='Vehicle GNSS Data', marker='^')
                # # corresponding_gnss_plot = ax.scatter(northing_vec, easting_vec , c='cornflowerblue', label='Vehicle GNSS Data', marker='^')

                # corresponding_gnss_plot = ax.scatter(vehicle_easting, vehicle_northing, c='blue', label='Vehicle GNSS Data', marker='^') # Update the plot with the latest detected object GNSS location
                # # corresponding_gnss_plot = ax.scatter(vehicle_northing, vehicle_easting , c='blue', label='Vehicle GNSS Data', marker='^')


                # detected_objects_plot = ax.scatter(target_easting_vec, target_northing_vec, c='red', marker='o') # Update the plot with the latest detected object GNSS location
                # # detected_objects_plot = ax.scatter(target_northing_vec, target_easting_vec , c='orange', marker='o') # Update the plot with the latest detected object GNSS location

                # detected_objects_plot = ax.scatter(target_easting, target_northing, c='red', marker='o') # Update the plot with the latest detected object GNSS location
                # # detected_objects_plot = ax.scatter(target_northing, target_easting , c='red', marker='o') # Update the plot with the latest detected object GNSS location
 
                # fig.canvas.draw_idle()
                # matplotlib.pyplot.pause(1)  # A short pause to ensure the plot updates and remains responsive
                #################### End of dynamic plot update #########################

             

                

                # Resize the image before displaying it
                new_size = (1024, 128)
                rgb_annotated_img = cv2.resize(rgb_annotated_img, new_size)

                #  Display the image without text annotations
                # cv2.imshow("Geo referenced poles", rgb_annotated_img)
                # cv2.waitKey(1)  # Refresh the display with the current image
                # time.sleep(3)  # Wait for 3 seconds
                # Step 2: Add the text annotations after 3 seconds
                # Define an initial vertical offset for annotations
                vertical_offset = 30
                line_height = 20  # Height of each line of text with extra space for padding

                # Distance w.r.t GNSS coordinates

                # Azimuth text with blue background and white text at the bottom left (xmin, ymin)
                annotation_text_azimuth = f"Azimuth: {azimuth_gnss:.6f} deg"
                (text_width_azimuth, text_height_azimuth), _ = cv2.getTextSize(annotation_text_azimuth, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw background rectangle for azimuth at (xmin, ymin) (blue background)
                cv2.rectangle(rgb_annotated_img, 
                            (x_min, y_min - text_height_azimuth),  # Bottom left corner
                            (x_min + text_width_azimuth, y_min),    # Top right corner
                            (255, 0, 0),  # Blue background
                            cv2.FILLED)

                # Place azimuth text (white text) at (xmin, ymin)
                cv2.putText(rgb_annotated_img, annotation_text_azimuth, (x_min, y_min), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Adjust vertical offset for next annotation with equal space
                vertical_offset += text_height_azimuth + line_height

                # Distance text with green background and white text
                annotation_text_distance = f"Distance: {nearest_distance_range:.6f} m"
                (text_width_distance, text_height_distance), _ = cv2.getTextSize(annotation_text_distance, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw background rectangle for distance (green background)
                cv2.rectangle(rgb_annotated_img, 
                            (x_max, y_max - vertical_offset - text_height_distance), 
                            (x_max + text_width_distance, y_max - vertical_offset + text_height_distance), 
                            (0, 255, 0),  # Green background
                            cv2.FILLED)

                # Draw circle (as originally intended)
                cv2.circle(rgb_annotated_img, (global_x, global_y), radius=1, color=(0, 0, 255), thickness=3)

                # Place distance text (white text)
                cv2.putText(rgb_annotated_img, annotation_text_distance, (x_max, y_max - vertical_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Adjust vertical offset for next annotation with equal space
                vertical_offset += text_height_distance + line_height

                # GNSS text with red background and white text
                annotation_text = f"GNSS: ({target_easting:.6f}, {target_northing:.6f})"
                (text_width_gnss, text_height_gnss), _ = cv2.getTextSize(annotation_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw background rectangle for GNSS (red background)
                cv2.rectangle(rgb_annotated_img, 
                            (x_max, y_max - vertical_offset - text_height_gnss), 
                            (x_max + text_width_gnss, y_max - vertical_offset + text_height_gnss), 
                            (0, 0, 255),  # Red background
                            cv2.FILLED)

                # Place GNSS text (white text)
                cv2.putText(rgb_annotated_img, annotation_text, (x_max, y_max - vertical_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                #  Redisplay the image with the text annotations
                # cv2.imshow("Geo referenced poles", rgb_annotated_img)

                # Assuming original image size and new image size
                original_size = (1024, 128)  # Original image size (width, height)
                new_size = (2048, 300)       # New image size (width, height)

                # Calculate scaling factors
                scale_x = new_size[0] / original_size[0]
                scale_y = new_size[1] / original_size[1]

                # Adjust the coordinates for the circle
                scaled_global_x = int(global_x * scale_x)
                scaled_global_y = int(global_y * scale_y)
                scaled_radius = int(1 * scale_x)  # Scale the circle radius if necessary

                # Adjust the text coordinates
                scaled_x_max = int(x_max * scale_x)
                scaled_y_max = int(y_max * scale_y)

                # Initial vertical offset and line height for annotations
                vertical_offset = int(20 * scale_y)  # Adjust the offset based on the scaling
                line_height = int(10 * scale_y)  # Adjust line height similarly

                # Distance annotation text
                annotation_text_distance = f"Distance: {nearest_distance_range:.6f} m"
                cv2.circle(rgb_annotated_img, (scaled_global_x, scaled_global_y), radius=scaled_radius, color=(0, 0, 255), thickness=3)
                cv2.putText(rgb_annotated_img, annotation_text_distance, (scaled_x_max, scaled_y_max - vertical_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_x, (0, 100, 255), 2)

                # GNSS annotation text
                vertical_offset += line_height
                annotation_text = f"GNSS: ({target_easting:.6f}, {target_northing:.6f})"
                cv2.putText(rgb_annotated_img, annotation_text, (scaled_x_max, scaled_y_max - vertical_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_x, (0, 255, 100), 2)

                # Resize the image to new dimensions
                rgb_annotated_img = cv2.resize(rgb_annotated_img, new_size)

                # Display the annotated image
                cv2.imshow("Geo referenced poles", rgb_annotated_img)

                range_image_vis = cv2.resize(range_image_vis, new_size) 
                # cv2.imshow('range_image', range_image_vis)
                
                

                cv2.waitKey(1)



# Remove all nearest poles from the unused poles list               
for pole in nearest_poles:
    if pole in unused_poles:
        unused_poles.remove(pole)

# Count the number of unused poles
num_unused_poles = len(unused_poles)
total_poles = len(ground_truth_poles)


# Output the number of unused poles and total poles
print(f"Number of detected poles at the test location: {total_poles- num_unused_poles} out of {total_poles} total ground truth poles.")


# Calculate the standard deviation of the min_distance_list
std_dev_min_distance = np.std(min_distance_list)

# Plotting the histogram of min_distance with bin numbers, mean, median, and standard deviation
plt.figure(figsize=(8, 6))  # Set the size of the figure

# Histogram for min_distance
n, bins, patches = plt.hist(min_distance_list, bins=10, color='green', alpha=0.7, edgecolor='black')

# Adding bin labels on top of each bar
for i in range(len(patches)):
    bin_height = patches[i].get_height()
    plt.text(patches[i].get_x() + patches[i].get_width() / 2, bin_height, f'{int(bin_height)}', 
             ha='center', va='bottom', fontsize=12)

# Calculate and plot the mean line
mean_min_distance = np.mean(min_distance_list)
plt.axvline(mean_min_distance, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_min_distance:.2f}')

# Calculate and plot the median line
median_min_distance = np.median(min_distance_list)
plt.axvline(median_min_distance, color='red', linestyle='solid', linewidth=2, label=f'Median: {median_min_distance:.2f}')

# Adding the standard deviation text
plt.text(0.75, 0.75, f'Std Dev: {std_dev_min_distance:.2f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', horizontalalignment='right', color='purple')

# Adding titles and labels
plt.title('Histogram of Errors Distances', fontsize=16)
plt.xlabel('Error Distance', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Display the legend
plt.legend()


# Cleanup OpenCV windows
cv2.destroyAllWindows()
# Disable interactive mode
matplotlib.pyplot.ioff()

# Keep the plot open until it is manually closed
matplotlib.pyplot.show(block=True)



