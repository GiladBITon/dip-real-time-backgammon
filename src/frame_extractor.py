import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def extract_frames_from_webcam(camera_index=0, output_dir="webcam_frames", frame_interval=1):
    """
    Captures video from a webcam and extracts frames at the specified interval.

    Parameters:
    - camera_index (int): The index of the webcam to use.
    - output_dir (str): The directory to save the frames.
    - frame_interval (int): Extract every nth frame.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the webcam
    camera = cv2.VideoCapture(camera_index)

    fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"Webcam FPS: {fps}")

    # Check if the webcam is opened successfully
    if not camera.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}.")
        return

    frame_count = 0
    saved_count = 0

    print("Press 'q' to stop capturing video.")

    while True:
        # Read a frame from the webcam
        success_capture, frame = camera.read()

        if not success_capture:
            print("Failed to capture frame. Exiting...")
            break

        # Display the frame
        cv2.imshow("Webcam Feed", frame)

        # Save the frame at the specified interval
        # if cv2.waitKey(1) & 0xFF == ord('c'):
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")
            saved_count += 1

        frame_count += 1

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()
    print(f"Frames saved in '{output_dir}'. Total frames saved: {saved_count}")

def find_intersections_by_proximity(points, threshold=10):
    """
    Finds intersections by grouping points that are close to each other.

    Args:
        points: List of all endpoints as (x, y).
        threshold: Maximum distance between two points to consider them as intersecting.

    Returns:
        intersections: List of unique intersection points as (x, y).
    """
    intersections = []
    visited_point = [False] * len(points)  # Track if a point has already been clustered

    for i, point1 in enumerate(points):
        if not visited_point[i]:
            visited_point[i] = True

            for j, point2 in enumerate(points):
                if not visited_point[j]:
                    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
                    if distance < threshold:
                        avg_x = (point1[0] + point2[0]) // 2
                        avg_y = (point1[1] + point2[1]) // 2
                        intersections.append((avg_x, avg_y))
                        visited_point[j] = True
                        # print(f"Found intersection at ({avg_x}, {avg_y})")
                        break

    return intersections

def plot_hough_space(lines, max_rho):
    """
    Plots the Hough space for a set of lines.

    Args:
        lines: List of lines as [(rho, theta)] or [(x1, y1, x2, y2)].
        max_rho: Maximum possible rho value for the image dimensions.
    """
    rhos = []
    thetas = []
    hough_vars = []

    # Check if the input lines are in (rho, theta) or (x1, y1, x2, y2) format
    line_lst = []
    for x1, y1, x2, y2 in lines:
        # Convert to (rho, theta)
        theta = np.arctan2(y2 - y1, x2 - x1)
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        # if 490 < rho < 570:
        rhos.append(rho)
        thetas.append(theta)
        hough_vars.append((rho, theta))
        # y2 = avg_lower_y
        line_lst.append((x1, y1, x2, y2))
            # print(f'y1={y1} ; y2={y2} ; x1={x1} ; x2={x2}')
    # print(f'Thetas: {len(thetas)} ; {sorted(thetas)}')
    # print(f'Rho: {len(rhos)} ; {sorted(rhos)}')
    # Plot the Hough space
    hough_vars.sort(key=lambda var: var[0])
    # for i in range(len(rhos)):
    #     print(f'Rho={hough_vars[i][0]} ; Theta={hough_vars[i][1]}')
    # avg_y1 = int(np.mean([y1 for x1, y1, x2, y2 in line_lst]))
    # avg_y2 = int(np.mean([y2 for x1, y1, x2, y2 in line_lst]))
    # rho_lst, theta_lst, lst = [], [], []
    # for x1, y1, x2, y2 in line_lst:
    #     y1 = avg_y1
    #     y2 = avg_y2
    #     theta = np.arctan2(y2 - y1, x2 - x1)
    #     rho = x1 * np.cos(theta) + y1 * np.sin(theta)
    #     rho_lst.append(rho)
    #     theta_lst.append(theta)
    #     lst.append((x1, y1, x2, y2))


    # plt.figure(figsize=(10, 10))
    # plt.scatter(thetas, rhos, color="red", marker="o", s=20, label="Detected Lines")
    # plt.title("Hough Space")
    # plt.xlabel("Theta (radians)")
    # plt.ylabel("Rho (pixels)")
    # plt.xticks(np.linspace(-np.pi, np.pi, 10))  # From 0 to pi, with 6 intervals
    # plt.yticks(np.linspace(-max_rho, max_rho, 10))  # From -max_rho to +max_rho
    # plt.grid(color="gray", linestyle="--", linewidth=0.5)
    # plt.legend()
    # plt.show()

    return line_lst

def board2tiles(img):
    # Step 1: Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)  # Kernel size of 3
    edges = cv2.Canny(blurred, 0, 150)
    return edges
    # Step 2: Detect triangle lines
    line_img = np.zeros_like(edges)
    line_img, lines = detect_and_visualize_triangle_lines(line_img, edges, min_line_length=edges.shape[0] * 0.15) # Proportional length of triangle height compared to board height

    height, width = edges.shape
    max_rho = int(np.sqrt(height ** 2 + width ** 2))

    # Visualize Hough space
    line_lst = plot_hough_space(lines, max_rho)
    # return line_img

    line_lst.sort(key=lambda line: line[2])  # Sort by x2

    dx1_lst, dx2_lst = [], []
    for i in range(1, len(line_lst)):
        dx1 = line_lst[i][0] - line_lst[i - 1][0]
        dx2 = line_lst[i][2] - line_lst[i - 1][2]
        dx1_lst.append(dx1)
        dx2_lst.append(dx2)
        # print(f'dx1={dx1} ; dx2={dx2}')
    # print(f'dx1_median={np.median(dx1_lst)} ; dx2_median={np.median(dx2_lst)}')
    median_dist = (np.median(dx1_lst) + np.median(dx2_lst)) / 2

    new_line_lst = []
    i = 0
    while i < len(line_lst) - 1:
        dist = np.sqrt(dx1_lst[i]**2 + dx2_lst[i]**2)
        new_line_lst.append(line_lst[i])
        if dist < median_dist * 0.2:
            i += 1
        if dist > median_dist * 1.5:
            new_line = (
                line_lst[i][0]+int(np.median(dx1_lst)), line_lst[i][1],
                line_lst[i][2]+int(np.median(dx2_lst)), line_lst[i][3]
            )
            new_line_lst.append(new_line)
        i += 1

    x1 = line_lst[0][0]
    while 0 < x1:
        if (x1 - 2*np.median(dx1_lst)) > 0: # Change 1.5 to add width
            new_line = (
                line_lst[0][0] - int(np.median(dx1_lst)), line_lst[0][1],
                line_lst[0][2] - int(np.median(dx2_lst)), line_lst[0][3]
            )
            new_line_lst.insert(0, new_line)
        x1 -= np.median(dx1_lst)

    x2 = line_lst[-1][2]
    while x2 < width:
        if (x2 + np.median(dx2_lst)) < width:
            new_line = (
                line_lst[-1][0] + int(np.median(dx1_lst)), line_lst[-1][1],
                line_lst[-1][2] + int(np.median(dx2_lst)), line_lst[-1][3]
            )
            new_line_lst.append(new_line)
        x2 += np.median(dx2_lst)

    dx1_lst, dx2_lst = [], []
    for i in range(1, len(new_line_lst)):
        dx1 = new_line_lst[i][0] - new_line_lst[i - 1][0]
        dx2 = new_line_lst[i][2] - new_line_lst[i - 1][2]
        dx1_lst.append(dx1)
        dx2_lst.append(dx2)
        print(f'dx1={dx1} ; dx2={dx2}')


    for x1, y1, x2, y2 in new_line_lst:
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return line_img

    # # Extract all endpoints
    # points = []
    # for x1, y1, x2, y2 in lines:
    #     points.append((x1, y1))
    #     points.append((x2, y2))
    #
    # intersections = find_intersections_by_proximity(points, threshold=edges.shape[1] * 0.05) # Proportional length of half triangle base compared to board width
    # # return edges
    # # intersections = find_all_intersections(lines)
    # for point in intersections:
    #     x, y = point
    #     if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:  # Ensure points are within image bounds
    #         cv2.circle(edges, (x, y), 5, (0, 255, 0), -1)  # Green circle
    # return edges

# def find_triangles(img):
#     # Step 1: Preprocessing
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.medianBlur(gray, 3)  # Kernel size of 3
#     edges = cv2.Canny(blurred, 50, 150)
#
#     # Step 2: Detect triangle lines
#     line_img = np.zeros_like(edges)
#     line_img, lines = detect_and_visualize_triangle_lines(line_img, edges, min_line_length=edges.shape[0] * 0.15)  # Proportional length of triangle height compared to board height
#
#     height, width = edges.shape
#     max_rho = int(np.sqrt(height ** 2 + width ** 2))
#     print(width)
#     # Visualize Hough space
#     line_lst = plot_hough_space(lines, max_rho)
#
#     # Align same y values for each cluster
#     upper_lines, lower_lines = [], []
#     for x1,y1,x2,y2 in line_lst:
#         if y1 <= height/2 and y2 <= height/2:
#             upper_lines.append((x1,y1,x2,y2)) if y1<y2 else upper_lines.append((x2,y2,x1,y1))
#         else:
#             lower_lines.append((x1,y1,x2,y2)) if y1<y2 else lower_lines.append((x2,y2,x1,y1))
#
#     def align_y_val(lines):
#         y1_median = int(np.median([line[1] for line in lines]))
#         y2_median = int(np.median([line[3] for line in lines]))
#         aligned_lines = []
#         for x1, y1, x2, y2 in lines:
#             theta = np.arctan2(y2 - y1, x2 - x1)
#             x1_new = x1 + int(np.cos(theta) * (y1_median - y1))
#             x2_new = x2 + int(np.cos(theta) * (y2_median - y2))
#             aligned_lines.append((x1_new, y1_median, x2_new, y2_median))
#         return aligned_lines
#
#     def align_x_val(lines):
#         for i in range(1, len(lines)):
#             print(f'x1={lines[i][0]} ; x2={lines[i][2]}')
#
#
#     def remove_duplicate_lines(lines, board_width):
#         lines.sort(key=lambda line: line[0])
#         dx_lst = []
#         for i in range(1, len(lines)):
#             dx1 = lines[i][0] - lines[i - 1][0]
#             dx2 = lines[i][2] - lines[i - 1][2]
#             dx_lst.append(np.sqrt(dx1 ** 2 + dx2 ** 2))
#         median_dist = int(np.median(dx_lst))
#
#         filtered_lines = [lines[0]]
#         for i in range(1, len(lines)):
#             dx1 = lines[i][0] - filtered_lines[-1][0]
#             dx2 = lines[i][2] - filtered_lines[-1][2]
#             dist = np.sqrt(dx1 ** 2 + dx2 ** 2)
#             if dist > median_dist * 0.5:
#                 filtered_lines.append(lines[i])
#
#         # Align X values
#         for i in range(1, len(filtered_lines)):
#             dx1 = filtered_lines[i][0] - filtered_lines[i - 1][0]
#             dx2 = filtered_lines[i][2] - filtered_lines[i - 1][2]
#             if abs(dx1) < median_dist * 0.5:
#                 filtered_lines[i - 1] = (
#                     int((filtered_lines[i][0] + filtered_lines[i - 1][0]) / 2),
#                     filtered_lines[i - 1][1],
#                     filtered_lines[i - 1][2],
#                     filtered_lines[i - 1][3])
#                 filtered_lines[i] = (
#                     int((filtered_lines[i][0] + filtered_lines[i - 1][0]) / 2),
#                     filtered_lines[i][1],
#                     filtered_lines[i][2],
#                     filtered_lines[i][3])
#
#             if abs(dx2) < median_dist * 0.5:
#                 filtered_lines[i - 1] = (
#                     filtered_lines[i - 1][0],
#                     filtered_lines[i - 1][1],
#                     int((filtered_lines[i][2] + filtered_lines[i - 1][2]) / 2),
#                     filtered_lines[i - 1][3])
#                 filtered_lines[i] = (
#                     filtered_lines[i][0],
#                     filtered_lines[i][1],
#                     int((filtered_lines[i][2] + filtered_lines[i - 1][2]) / 2),
#                     filtered_lines[i][3])
#
#         # Fill triangles between already created
#         left_point = median_dist * 1.5 + 1
#         while left_point > median_dist * 1.5:
#             x1, y1, x2, y2 = filtered_lines[0]
#             if x1 > x2:
#                 new_x1 = x1 - median_dist
#                 filtered_lines.insert(0, (new_x1, y1, x2, y2))
#                 left_point = new_x1
#             if x2 > x1:
#                 new_x2 = x2 - median_dist
#                 filtered_lines.insert(0, (x1, y1, new_x2, y2))
#                 left_point = new_x2
#
#         right_point = board_width - median_dist - 1
#         while right_point < board_width - median_dist:
#             x1, y1, x2, y2 = filtered_lines[-1]
#             if x1 < x2:
#                 new_x1 = x1 + median_dist
#                 filtered_lines.append((new_x1, y1, x2, y2))
#                 right_point = new_x1
#             if x2 < x1:
#                 new_x2 = x2 + median_dist
#                 filtered_lines.append((x1, y1, new_x2, y2))
#                 right_point = new_x2
#
#         i = 0
#         right_inner_point = filtered_lines[0][0]
#         while right_inner_point < board_width / 2 - median_dist:
#             x1, y1, x2, y2 = filtered_lines[i]
#             if abs(x1 - filtered_lines[i + 1][0]) < 2 or abs(x2 - filtered_lines[i + 1][2]) < 2:
#                 right_inner_point = filtered_lines[i + 1][0] if x1 > x2 else filtered_lines[i + 1][2]
#             elif x1 < x2:
#                 if abs(x1 - filtered_lines[i + 1][0]) < 1.5 * median_dist: # 1 line gap
#                     new x1 = (x1 + filtered_lines[i + 1][0]) / 2
#                 new_x1 = x1 + median_dist
#                 filtered_lines.insert(i+1, (new_x1, y1, x2, y2))
#                 right_inner_point = new_x1
#             elif x2 < x1:
#                 new_x2 = x2 + median_dist
#                 filtered_lines.insert(i+1, (x1, y1, new_x2, y2))
#                 right_inner_point = new_x2
#             i += 1
#
#         # while True:
#         #     x1, y1, x2, y2 = filtered_lines[i]
#         #     if abs(x1 - filtered_lines[i + 1][0]) < 2 or abs(x2 - filtered_lines[i + 1][2]) < 2: # Same point
#         #         right_inner_point = filtered_lines[i + 1][0] if x1 > x2 else filtered_lines[i + 1][2]
#         #     elif x1 < x2:
#         #         new_x1 = x1 + median_dist
#         #         if new_x1 < board_width / 2 - median_dist:
#         #             filtered_lines.insert(i+1, (new_x1, y1, x2, y2))
#         #             right_inner_point = new_x1
#         #         else:
#         #             break
#         #     elif x2 < x1:
#         #         new_x2 = x2 + median_dist
#         #         if new_x2 < board_width / 2 - median_dist:
#         #             filtered_lines.insert(i+1, (x1, y1, new_x2, y2))
#         #             right_inner_point = new_x2
#         #         else:
#         #             break
#         #     i += 1
#
#         i = -1
#         left_inner_point = filtered_lines[-1][0]
#         while left_inner_point > board_width / 2 + median_dist:
#             x1, y1, x2, y2 = filtered_lines[i]
#             if abs(x1 - filtered_lines[i - 1][0]) < 2 or abs(x2 - filtered_lines[i - 1][2]) < 2:
#                 left_inner_point = filtered_lines[i - 1][0] if x1 < x2 else filtered_lines[i - 1][2]
#             elif x1 > x2:
#                 new_x1 = x1 - median_dist
#                 filtered_lines.insert(i, (new_x1, y1, x2, y2))
#                 left_inner_point = new_x1
#             elif x2 > x1:
#                 new_x2 = x2 - median_dist
#                 filtered_lines.insert(i, (x1, y1, new_x2, y2))
#                 left_inner_point = new_x2
#             i -= 1
#
#         # added_lines = []
#         # for i in range(1, len(filtered_lines)):
#         #     dx1 = filtered_lines[i][0] - filtered_lines[i - 1][0]
#         #     dx2 = filtered_lines[i][2] - filtered_lines[i - 1][2]
#         #     print(f'dx1={dx1} ; dx2={dx2}')
#         #     if dx1 < 2 and dx2 > median_dist * 0.5:
#         #
#         # if filtered_lines[1][0] - filtered_lines[0][0] < 2:
#
#
#         return filtered_lines
#
#     upper_lines = align_y_val(upper_lines)
#     lower_lines = align_y_val(lower_lines)
#
#     upper_lines = remove_duplicate_lines(upper_lines, width)
#     lower_lines = remove_duplicate_lines(lower_lines, width)
#
#     # upper_lines = align_x_val(upper_lines)
#     # lower_lines = align_x_val(lower_lines)
#
#
#     for x1, y1, x2, y2 in upper_lines + lower_lines:
#         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     # Draw center line
#     cv2.line(img, (width//2, 0), (width//2, height - 1), (0, 255, 255), 2)
#     return img
#
#
#
#     # # Extract all endpoints
#     # points = []
#     # for x1, y1, x2, y2 in lines:
#     #     points.append((x1, y1))
#     #     points.append((x2, y2))
#     #
#     # intersections = find_intersections_by_proximity(points, threshold=edges.shape[1] * 0.05) # Proportional length of half triangle base compared to board width
#     # # return edges
#     # # intersections = find_all_intersections(lines)
#     # for point in intersections:
#     #     x, y = point
#     #     if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:  # Ensure points are within image bounds
#     #         cv2.circle(edges, (x, y), 5, (0, 255, 0), -1)  # Green circle
#     # return edges

def detect_and_visualize_triangle_lines(image, edges, rho=1, theta=np.pi/180, threshold=50, min_line_length=30, max_line_gap=10, line_color=(0, 0, 255), line_thickness=2):
    line_segments = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Create a copy of the original image for visualization
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

    # Draw the detected line segments
    angle_range = [(5, 85), (-85, -5)] # Don't take horizontal and vertical lines
    filtered_lines = []
    if line_segments is not None:
        for x1, y1, x2, y2 in line_segments[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # Angle in degrees

            for min_angle, max_angle in angle_range:
                if min_angle <= angle <= max_angle:
                    filtered_lines.append((x1, y1, x2, y2))
                    cv2.line(output, (x1, y1), (x2, y2), line_color, line_thickness)
                    break

    return output, filtered_lines

def segment_dice(img):
    # Step 1: Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)  # Kernel size of 3

    # Step 2: Edge Detection
    edges = cv2.Canny(blurred, 0, 150)
    edges, line_lst = detect_and_visualize_triangle_lines(edges, edges, min_line_length=edges.shape[0] * 0.1)
    return edges
    # return edges
    # Step 1: Preprocessing
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Enhance contrast (optional)
    # gray = cv2.equalizeHist(gray)
    # return gray
    # Step 2: Thresholding
    # Use adaptive thresholding for better results in uneven lighting
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY_INV, 3, 5)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)
    # remove_triangles(binary)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    return edges
    # Dilate the edges to close gaps
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # edges_dilated = cv2.dilate(binary, kernel, iterations=1)
    return binary
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Experiment with kernel size
    # binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Step 3: Apply Gaussian Blur to Reduce Noise
    # binary_blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    # Apply Otsu's thresholding
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_closed
    # Step 3: Morphological Operations to Remove Noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary
    h, s, v = cv2.split(hsv_img)
    # Normalize each channel independently
    # h_normalized = cv2.normalize(h, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # s_normalized = cv2.normalize(s, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # v_normalized = cv2.normalize(v, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Merge normalized channels back into HSV
    # hsv_normalized = cv2.merge([h_normalized, s_normalized, v_normalized])

    lower_bound = np.array([0, 0, 200])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    # gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, gs_img = cv2.threshold(gs_img, 127, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(gs_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # mask = gs_img.copy()
    # cv2.drawContours(mask, contours, -1, (255), -1)


    return mask

def straighten_and_crop_board(img):
    # Step 1: Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)  # Kernel size of 3

    # Step 2: Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 3: Morphological Closing to connect lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)  # Expand edges slightly
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closed
    # Step 4: Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board_corners = None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in sorted_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            board_corners = approx
            break

    # Debugging: Draw the detected polygon
    if board_corners is None:
        print("No polygon found!")
        return img
    else:
        cv2.drawContours(gray, [board_corners], -1, (0, 255, 0), 3)  # Green line
        # print("Polygon found and drawn!")

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    # Reshape and order the corners
    board_corners = board_corners.reshape(4, 2)  # Convert from (4, 1, 2) to (4, 2)
    ordered_corners = order_points(board_corners)

    # Compute the width and height of the new image
    (tl, tr, br, bl) = ordered_corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)

    # Use the maximum width and height for the new image
    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst)

    # Apply the perspective warp
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))
    return warped

def segment_checkers(img):
    # Step 1: Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    mask_hue = cv2.inRange(h, np.array([95]), np.array([110]))
    cv2.imshow("Mask Hue", mask_hue)
    # cv2.imshow("H", h)
    # cv2.imshow("S", s)
    # cv2.imshow("V", v)
    cv2.waitKey(0)
    # Step 2: Define color ranges for pieces
    # Adjust these ranges based on the colors of the pieces in your image
    lower_black = np.array([0, 0, 0])  # Lower bound for black
    upper_black = np.array([180, 255, 50])  # Upper bound for black

    lower_white = np.array([0, 0, 200])  # Lower bound for white
    upper_white = np.array([180, 30, 255])  # Upper bound for white

    # Step 3: Create masks for each color
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Debugging Stage 1: Display the masks
    cv2.imshow("Mask - Black", mask_black)
    cv2.imshow("Mask - White", mask_white)
    cv2.waitKey(0)

    # Step 4: Find contours for each mask
    def find_pieces(mask, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter out small noise
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(img, center, radius, color, 2)  # Draw a circle around the piece

    # Step 5: Detect and mark pieces on the board
    find_pieces(mask_black, (0, 0, 0))  # Black pieces
    find_pieces(mask_white, (255, 255, 255))  # White pieces
    return img

def detect_hsv_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the callback function
    def get_hsv_value(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
            pixel = hsv[y, x]
            hue, sat, val = pixel
            print(f"HSV at ({x}, {y}): H={hue}, S={sat}, V={val}")

    # Display the image and set the mouse callback
    cv2.imshow("Click to Inspect HSV", img)
    cv2.setMouseCallback("Click to Inspect HSV", get_hsv_value)

    # Wait until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set parameters
    main_dir = '/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/'
    os.listdir('/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/') # Project directory
    camera_index = 0  # Change this to the index of your webcam if needed
    output_dir = "webcam_frames_7"  # Directory to save frames
    frame_interval = 7.5  # Save every n-th frame (fps = 7.5)

    # Extract frames from the webcam
    extract_frames_from_webcam(camera_index, output_dir, frame_interval)

    # Segment
    # img_path = os.path.join(main_dir, output_dir, "dice_6.jpg")
    # board_img = cv2.imread(img_path)
    # # img_path = os.path.join(main_dir, output_dir, "aligned_board.jpg")
    # # aligned_board = cv2.imread(img_path)
    # aligned_board = straighten_and_crop_board(board_img)
    # # cv2.imwrite(os.path.join(output_dir, "aligned_board.jpg"), aligned_board)
    # # board_with_tiles = find_triangles(aligned_board)
    # plt.figure()
    # plt.imshow(aligned_board, cmap='gray')
    # # plt.axis('off')
    # plt.show()
