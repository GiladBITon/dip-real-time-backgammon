import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from itertools import permutations

"""Image Processing -------------------------------------------------------------------------------------------------"""
def show_game(last_verified_state, real_time_state, grid, text_prompt, dice_images):
    def fit_to_window(window,img):
        # Define fixed window size
        window_width, window_height = window
        # Get image dimensions
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
        h, w = img.shape[:2]
        # Create a black canvas
        canvas = np.zeros((window_width, window_height, 3), dtype=np.uint8)
        # Resize image while maintaining aspect ratio
        scale = min(window_width / w, window_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(img, (new_w, new_h))
        # Compute top-left corner for centering
        x_offset = (window_width - new_w) // 2
        y_offset = (window_height - new_h) // 2
        # Place resized image on canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        return canvas

    last_verified_state = fit_to_window((500, 500), last_verified_state)
    real_time_state = fit_to_window((500, 500), real_time_state)
    grid = fit_to_window((500, 500), grid)
    dice_images = fit_to_window((500, 500), dice_images)
    text_prompt = fit_to_window((500, 500), text_prompt)
    canvas = concat_images(last_verified_state, real_time_state)
    canvas = concat_images(canvas, grid)
    canvas = concat_images(canvas, dice_images)
    canvas = concat_images(canvas, text_prompt)

    return canvas

def show_text(text):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 2
    thickness = 4
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Red background
    canvas = np.zeros((500, 500))

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate text position
    x = 10
    y = text_height + 20

    # Put text on top of background
    cv2.putText(canvas, text, (x, y), font, font_scale, text_color, thickness)

    return canvas

def show_error(img, text):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 2
    thickness = 4
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Red background

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate text position
    image_height, image_width, _ = img.shape
    x = (image_width - text_width) // 2
    y = (image_height + text_height) // 2

    # Draw background rectangle
    cv2.rectangle(img, (x - 10, y - text_height - 10), (x + text_width + 10, y + baseline + 10), bg_color, -1)

    # Put text on top of background
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def show_dice(dice_images, dice_num):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    thickness = 2
    text_color = (255, 255, 255)  # White text
    dice_show = []
    dice_count = min(len(dice_images), len(dice_num))
    # print(f"Count: {dice_count}")
    for i in range(dice_count):
        img = np.zeros_like(dice_images[i])
        text = str(dice_num[i])
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Calculate text position
        image_height, image_width = img.shape
        x = (image_width - text_width) // 2
        y = (image_height + text_height) // 2
        # Put text on top of background
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
        dice_show.append(concat_images(dice_images[i], img))

        # cv2.imwrite(f"/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/webcam_frames_5/dice_{dice_num[i]}.jpg", dice_images[i])

    if dice_count == 0:
        return None
    if dice_count == 1:
        return dice_show[0]
    if dice_count == 2:
        return concat_images(dice_show[0], dice_show[1], "v")

def scale_image(ref_img, resized_img):
    # Ensure both images are 3 dimensional
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB) if len(ref_img.shape) == 2 else ref_img
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB) if len(resized_img.shape) == 2 else resized_img
    # Ensure both images have the same height
    h1, w1 = ref_img.shape[:2]
    # h2, w2 = resized_img.shape[:2]
    # # Resize edges_img to match frame's height
    # scale_factor = h1 / h2  # Compute scaling ratio
    # new_width = int(w2 * scale_factor)  # Adjust width proportionally
    # resized_img = cv2.resize(resized_img, (new_width, h1))
    resized_img = cv2.resize(resized_img, (w1, h1), interpolation=cv2.INTER_LINEAR)
    return resized_img

def concat_images(ref_img, resized_img, ctype="h"):
    # Ensure both images are 3 dimensional
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB) if len(ref_img.shape) == 2 else ref_img
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB) if len(resized_img.shape) == 2 else resized_img
    # Ensure both images have the same height
    h1, w1 = ref_img.shape[:2]
    h2, w2 = resized_img.shape[:2]

    # Concatenate images side by side
    if ctype == "h":
        scale_factor = h1 / h2  # Compute scaling ratio
        new_width = int(w2 * scale_factor)  # Adjust width proportionally
        resized_img = cv2.resize(resized_img, (new_width, h1))
        combined = cv2.hconcat([ref_img, resized_img])
    else:
        # Resize edges_img to match frame's height
        scale_factor = w1 / w2  # Compute scaling ratio
        new_height = int(h2 * scale_factor)  # Adjust width proportionally
        resized_img = cv2.resize(resized_img, (w1, new_height))
        combined = cv2.vconcat([ref_img, resized_img])
    return combined

def refine_board_corners(img, board_corners):
    """
    Refines detected board corners by identifying the most accurate points using Harris Corner Detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray32 = np.float32(gray)
    dst = cv2.cornerHarris(gray32, 10, 5, 0.04)
    # dst = cv2.dilate(dst, None)
    threshold = 0.01 * dst.max()
    img[dst > threshold] = [255, 255, 0]
    corner_candidates = np.argwhere(dst > threshold)
    corner_candidates = np.array(corner_candidates)
    exact_board_corners = np.zeros((4, 2))

    for i, corner in enumerate(board_corners):
        swapped_corner = corner[::-1]
        distances = np.linalg.norm(corner_candidates - swapped_corner, axis=1)
        closest_point = corner_candidates[np.argmin(distances)]
        exact_board_corners[i] = (closest_point[::-1])

    return exact_board_corners.astype(np.int32)

def safe_polyfit(x_values, y_values):
    """
    Computes the slope of a line using np.polyfit, but handles edge cases
    where the line is vertical or poorly conditioned.

    Parameters:
        x_values (tuple): Two x-coordinates
        y_values (tuple): Two y-coordinates

    Returns:
        float or None: Slope (m) if valid, otherwise None
    """
    if np.isnan(x_values).any() or np.isnan(y_values).any():
        return None  # Handle NaN values silently

    if abs(x_values[0] - x_values[1]) < 1e-6:  # Near-vertical line check
        return None  # Return None without printing anything

    return np.polyfit(x_values, y_values, 1)[0]  # Return slope m

def detect_board(frame, game):
    # Step 1: Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 3)  # Kernel size of 3

    # Step 2: Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 3: Morphological Closing to connect lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)  # Expand edges slightly changed from 2 to 1
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Step 4: Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board_corners = None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in sorted_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            board_corners = approx.reshape(4, 2)
            break

    # Debugging: Draw the detected polygon
    if board_corners is None:
        # print("No polygon found!")
        return None, closed

    # cv2.drawContours(frame, [board_corners], -1, (0, 0, 255), 3)  # Green line
    # board_corners = refine_board_corners(frame, board_corners)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    # Order the corners
    ordered_corners = order_points(board_corners)

    if game["legal_state_buffer"] > 2:
        game["last_corners"] = ordered_corners
        game["legal_state_buffer"] = 0
    else:
        if game["last_corners"] is None:
            game["last_corners"] = ordered_corners
        else:
            distances = np.linalg.norm(ordered_corners - game["last_corners"], axis=1)
            moved_corners = np.sum(distances > 5)
            if moved_corners >= 3:
                if game["board_buffer"] < 2:
                    game["board_buffer"] += 1
                    return None, closed
                    # ordered_corners = game["last_corners"]
                else:
                    game["board_buffer"] = 0
                    game["last_corners"] = ordered_corners
            else:
                ordered_corners = game["last_corners"]

        # print(distances)
        # distances = distances.reshape(-1, 1)
        # stable_corners = np.where(distances > 20, ordered_corners, game["last_corners"])
        # # print(np.sort(stable_corners,axis=0))
        # if not np.array_equal(stable_corners, game["last_corners"]):
        #     if game["board_buffer"] <= 10:
        #         game["board_buffer"] += 1
        #     else:
        #         game["board_buffer"] = 0
        #         game["last_corners"] = stable_corners
        #         ordered_corners = stable_corners
        # else:
        #     ordered_corners = game["last_corners"]

    # Compute the width and height of the new image
    (tl, tr, br, bl) = ordered_corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)

    try:
        w1 = safe_polyfit((tl[0], tr[0]), (tl[1], tr[1]))  # Top edge
        w2 = safe_polyfit((bl[0], br[0]), (bl[1], br[1]))  # Bottom edge
        h1 = safe_polyfit((bl[0], tl[0]), (bl[1], tl[1]))  # Left edge
        h2 = safe_polyfit((br[0], tr[0]), (br[1], tr[1]))  # Right edge

        # Convert slopes to angles
        def slope_to_angle(m):
            return np.degrees(np.arctan(m)) if m is not None else None

        angle_w1 = slope_to_angle(w1)
        angle_w2 = slope_to_angle(w2)
        angle_h1 = slope_to_angle(h1)
        angle_h2 = slope_to_angle(h2)

        # print(f"Width ratio: {width_top / width_bottom}, Height ratio: {height_left / height_right}")
        # print(f"Width Angles ratio: {abs(angle_w1 - angle_w2):.2f}, Height Angles ratio: {abs(angle_h1 - angle_h2):.2f}")

        if abs(angle_w1 - angle_w2) > 1.5 or abs(angle_h1 - angle_h2) > 1.5:
            # print(f"Stopped at {abs(angle_w1 - angle_w2):.2f}, {abs(angle_h1 - angle_h2):.2f}")
            return None, closed
    except:
        # print("Error")
        a = 1

    if max(width_top, width_bottom) < frame.shape[1] * 0.3 or max(height_left, height_right) < frame.shape[0] * 0.3:
      # cv2.drawContours(frame, [board_corners], -1, (0, 255, 0), 3)  # Green line
      return None, closed

    # Use the maximum width and height for the new image
    # if max_width == 0 and max_height == 0:
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
    warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    cv2.drawContours(frame, [board_corners], -1, (0, 255, 0), 3)  # Green line
    return warped, closed

def black_threshold(img, n_bins = 64): # Image in BGR format
    blue_hist = cv2.calcHist([img], [2], None, [n_bins], [0, 256])
    # plt.plot(blue_hist, color='b', label="Blue"), plt.show()
    blue_hist_smooth = cv2.GaussianBlur(blue_hist, (5, 5), 0)  # Adjust kernel size as needed
    # plt.plot(blue_hist_smooth, color='b', label="Blue"), plt.show()
    peaks = find_peaks(blue_hist_smooth.flatten(), prominence=0.001)[0]  # Adjust prominence as needed
    if blue_hist_smooth.flatten()[0] > blue_hist_smooth.flatten()[1]: # Peak is at the start boundary
      black_thresh = (np.argmin(blue_hist_smooth[:peaks[0]])) * (256 / n_bins)
    # elif len(peaks) > 1:
    else:
      black_thresh = (np.argmin(blue_hist_smooth[peaks[0]:peaks[1]]) + peaks[0]) * (256 / n_bins)
    # else:
    #     black_thresh = 200
    # print(f"Black Threshold: {black_thresh}")
    black_mask = cv2.threshold(img[:, :, 2], black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
    # plt.figure(figsize=(5, 5)), plt.title("Black"), plt.imshow(black_mask), plt.axis('off'), plt.tight_layout(), plt.show()
    return black_mask

def white_threshold(img, n_bins = 64): # Image in BGR format
    red_hist = cv2.calcHist([img], [0], None, [n_bins], [0, 256])
    red_hist_smooth = cv2.GaussianBlur(red_hist, (5, 5), 0)  # Adjust kernel size as needed
    peak = np.argmax(red_hist_smooth.flatten())
    valley = np.argmin(red_hist_smooth[peak:n_bins]) + peak
    indices = np.argwhere(red_hist_smooth < ((red_hist_smooth[peak] - red_hist_smooth[valley]) * 0.05) + red_hist_smooth[valley])[:,0]
    indices = indices[indices > peak]
    white_thresh = indices[0] * (256/n_bins)
    # print(f"White Threshold: {white_thresh}")
    white_mask = cv2.threshold(img[:, :, 0], white_thresh, 255, cv2.THRESH_BINARY)[1]
    # plt.figure(figsize=(5, 5)), plt.title("Black"), plt.imshow(white_mask), plt.axis('off'), plt.tight_layout(), plt.show()
    return white_mask

def template_match(input_img, output_img, threshold=0.5, min_samples=1, contour_color=(0, 0, 255), checker_radius=22, draw=True):
    # Create a Circular Template Matching Checker Size
    # checker_radius = 22  # Approximate checker radius (Tune this based on actual size)
    template_size = 2 * checker_radius
    template = np.zeros((template_size, template_size), dtype=np.uint8)
    cv2.circle(template, (checker_radius, checker_radius), checker_radius, 255, -1)

    # Perform Template Matching (Normalized Cross-Correlation)
    result = cv2.matchTemplate(input_img, template, cv2.TM_CCOEFF_NORMED)

    # Extract Locations Where Similarity is High
    locations = np.where(result >= threshold)

    # Extract detected checker locations from template matching
    raw_coordinates = np.array([(pt[0] + checker_radius, pt[1] + checker_radius) for pt in zip(*locations[::-1])])

    # Define Clustering Parameters
    eps = checker_radius  # Maximum distance for points to be considered in the same cluster

    final_checker_coordinates = []
    # Apply DBSCAN Clustering
    if len(raw_coordinates) > 0:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(raw_coordinates)
        unique_clusters = np.unique(clustering.labels_)

        # Compute Mean Position for Each Cluster (Final Detected Checkers)
        final_checker_coordinates = []
        for cluster in unique_clusters:
            if cluster == -1:
                continue  # Ignore noise points

            cluster_points = raw_coordinates[clustering.labels_ == cluster]
            mean_x = int(np.mean(cluster_points[:, 0]))
            mean_y = int(np.mean(cluster_points[:, 1]))
            final_checker_coordinates.append((mean_x, mean_y))

    # Draw Detected Checkers
    if draw:
        for (x, y) in final_checker_coordinates:
            cv2.circle(output_img, (x, y), checker_radius, contour_color, cv2.FILLED)  # Green circles

    return final_checker_coordinates

def segment_checkers(img, white_thresh, black_thresh):
    # Preprocessing
    output = img.copy()
    white_checkers = template_match(white_thresh, output, threshold=0.5, min_samples=1,
                                    contour_color=(0, 0, 255), checker_radius=int(img.shape[0] * 0.034))
    black_checkers = template_match(black_thresh, output, threshold=0.5, min_samples=1,
                                    contour_color=(0, 255, 0), checker_radius=int(img.shape[0] * 0.034))

    return output, white_checkers, black_checkers

def detect_dice(img, white_mask, black_mask, white_checkers):
    """ Detect and transform dice """
    white_checkers_img = np.zeros_like(img)
    for (x, y) in white_checkers:
        cv2.circle(white_checkers_img, (x, y), int(img.shape[0] * 0.034), (255,255,255), cv2.FILLED)  # Green circles
    white_checkers = cv2.cvtColor(white_checkers_img, cv2.COLOR_RGB2GRAY)  # 3 to 1 channel
    difference = white_mask - white_checkers
    difference = cv2.medianBlur(difference, 5)
    difference = cv2.dilate(difference, np.ones((3, 3), np.uint8), iterations=1)
    # plt.figure(figsize=(5, 5)), plt.imshow(difference, cmap='gray'), plt.show()
    contours = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    output = img.copy()
    filtered_contours = [c for c in contours if 500 < cv2.contourArea(c) < 2000]

    dice_images = []
    for contour in contours:
        # if len(dice_images) == 2:
        #   break
        if 500 < cv2.contourArea(contour) < 2000:
            rect = cv2.minAreaRect(contour)
            h, w = rect[1]
            if h == 0 or w == 0:
                continue
            aspect_ratio = h / float(w) if w > h else w / float(h)
            # print(aspect_ratio, cv2.contourArea(contour))
            if (500 < cv2.contourArea(contour) < 1200 and 0.8 < aspect_ratio < 1.2) or (
                    1000 < cv2.contourArea(contour) < 2000 and 0.3 < aspect_ratio < 0.7):
                cv2.drawContours(img, [contour], -1, (255, 255, 0), 3)
                # print(aspect_ratio, cv2.contourArea(contour))
                box = np.intp(cv2.boxPoints(rect))
                # cv2.polylines(img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                # print(cv2.contourArea(contour), aspect_ratio)
                # Compute the rotation matrix
                angle = rect[-1] if rect[-1] > -45 else rect[-1] + 90  # Get the rotation angle

                # Get the rotation matrix
                (h, w) = output.shape[:2]
                center = rect[0]
                M = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Rotate the image
                rotated = cv2.warpAffine(black_mask, M, (w, h))

                # Get bounding box of rotated dice
                x, y, w, h = cv2.boundingRect(np.intp(cv2.transform(np.array([box]), M)))
                dice_roi = rotated[y:y + h, x:x + w]  # Crop rotated dice
                dice_roi = cv2.threshold(dice_roi, 50, 255, cv2.THRESH_BINARY)[1]
                # print(f"Before cut: {dice_roi.shape},{x},{y},{w},{h}, {cv2.contourArea(contour)}")
                if dice_roi is None:
                    continue
                if cv2.contourArea(contour) < 1000:
                    size = int((0.05 * (dice_roi.shape[0] + dice_roi.shape[1]) / 2))
                    dice_images.append(dice_roi[size:-size,size:-size])
                    cv2.drawContours(output, [contour], -1, (255, 255, 0), 3)
                else:
                    cut_dice = [dice_roi[:, :w // 2], dice_roi[:, w // 2:]] if w>h else [dice_roi[:h //2,:], dice_roi[h//2:,:]]
                    for dice in cut_dice:
                        size = int((0.05 * (dice.shape[0] + dice.shape[1]) / 2))
                        dice_images.append(dice[size:-size,size:-size])

    def rotate_image(image, angle):
        """ Rotate an image by a given angle while keeping size consistent. """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated

    def get_best_match(dice_image, reference_images):
        """ Compare the real-time dice image against reference images with rotation handling. """
        best_score = 0
        best_match = None

        for value, ref_img in reference_images.items():
            # Rotate reference images only for dice values 2, 3, and 6
            rotations = [0, 90] if value in [2, 3, 6] else [0]

            for angle in rotations:
                rotated_img = rotate_image(ref_img, angle)
                # Resize to match the dice image dimensions
                resized_ref = cv2.resize(rotated_img, (dice_image.shape[1], dice_image.shape[0]))

                # Compute similarity score using matchTemplate
                dice_image_norm = dice_image / 255
                resized_ref = resized_ref / 255

                # score = np.sum(dice_image_norm * resized_ref)
                score = np.mean((dice_image_norm - resized_ref)**2)
                print(value, score)

                if score > best_score:
                    best_score = score
                    best_match = value

        return best_match

    """ Detect Number """
    dice_numbers = []
    # print(len(dice_images))
    for dice in dice_images:
        # pred = get_best_match(dice, number_templates)
        # dice = cv2.threshold(dice, 1, 255, cv2.THRESH_BINARY)[1]
        # print(dice.shape)
        try:
            if dice.shape[0] == 0 or dice.shape[1] == 0 or dice.shape[0] > 35 or dice.shape[1] > 35:
                continue
            dots = template_match(dice, dice, threshold=0.5, min_samples=1, contour_color=(255, 255, 255),
                                  checker_radius=2, draw=False)
            if len(dots) > 0:
                dice_num = int(np.clip(len(dots), 1, 6))
                dice_numbers.append(dice_num)
            # print(f"Correct: {len(dots)}, Predicted: {pred}")
        except:
            continue
        continue

    # print(len(dice_images))
    # if len(dice_images) > 1:
    #     cv2.imshow("Dice", concat_images(dice_images[0], dice_images[1]))
    return dice_images, dice_numbers

def plot_histograms(image):
    # Define color channels
    color_labels = ['R', 'G', 'B']
    colors = ['r', 'g', 'b']

    # Create figure
    plt.figure(figsize=(8, 4))

    for i in range(3):
        plt.subplot(1, 2, 1)  # Single plot for HSV
        hist = cv2.calcHist([image], [i], None, [64], [0, 256])
        plt.plot(hist, color=colors[i], label=color_labels[i])

    plt.title("RGB Color Space Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def create_grid(img):
    def detect_triangles(img):
        # Load the image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Preprocessing
        edges = cv2.Canny(gray, 20, 100)
        # edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)  # Expand edges slightly changed from 2 to 1
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3) # Change Back
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

        def calculate_compactness(contour):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                return 0  # Avoid division by zero
            return (4 * np.pi * area) / (perimeter ** 2)

        # Detect triangles
        triangles = []
        compactness_measures = []
        for contour in contours:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # If the approximated contour has 3 vertices, it's likely a triangle
            if len(approx) <= 4:
                area = cv2.contourArea(contour)
                compactness = calculate_compactness(contour)
                compactness_measures.append(compactness)
                # print(compactness)
                if area > 1800:  # Filter out small triangles (tune this threshold)
                    triangles.append(approx)

        # cv2.drawContours(img, triangles, -1, (0, 0, 255), 2)

        median_compactness = np.median(compactness_measures)
        # print(f"Median: {median_compactness}")
        def_triangles = []
        for contour in triangles:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            compactness = calculate_compactness(contour)
            if abs(compactness - median_compactness) < median_compactness * 0.2:
                def_triangles.append(approx)

        # cv2.drawContours(img, def_triangles, -1, (255, 0, 0), cv2.FILLED)

        def get_centroid(contour):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            return None

        # Filter duplicates by centroid proximity
        unique_triangles = []
        distance_threshold = 10  # Minimum distance between centroids

        for triangle in def_triangles:
            centroid = get_centroid(triangle)
            is_duplicate = False
            for existing in unique_triangles:
                existing_centroid = get_centroid(existing)
                if centroid and existing_centroid:
                    distance = np.linalg.norm(np.array(centroid) - np.array(existing_centroid))
                    if distance < distance_threshold:
                        is_duplicate = True
                        break
            if not is_duplicate:
                unique_triangles.append(triangle)
                # cv2.drawContours(img, [triangle], -1, (0, 255, 0), 2)

        return unique_triangles

    def split_triangle_points(triangle):
        # Flatten the triangle points for easy access
        points = [point[0] for point in triangle]
        right = max(points, key=lambda p: p[0])
        left = min(points, key=lambda p: p[0])

        return right, left

    def get_grid_points(img, triangles):
        # Store results for each triangle
        left_points = []
        right_points = []
        for triangle in triangles:
            right, left = split_triangle_points(triangle)
            right_points.append(right)
            left_points.append(left)

        sorted_right_points = sorted(right_points, key=lambda p: p[0])
        sorted_left_points = sorted(left_points, key=lambda p: p[0])

        mid_xs = []
        for i in range(11):
            if i == 5:
                continue
            mid_x = (sorted_right_points[i][0] + sorted_left_points[i + 1][0]) // 2
            mid_xs.append(mid_x)

        return mid_xs

    def detect_margin(triangles):
        left_points = []
        right_points = []
        for triangle in triangles:
            right, left = split_triangle_points(triangle)
            right_points.append(right)
            left_points.append(left)

        y_values = [point[1] for point in left_points + right_points]
        mean_base = int(np.mean(y_values) * 0.75)
        return mean_base

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    grid = np.zeros_like(img[:,:,1])
    val = 1

    h, w = img.shape[:2]
    upper_img = img[0:h//2, :]
    upper_triangles = detect_triangles(upper_img)
    if len(upper_triangles) != 12:
        return None
    lower_img = img[h//2:, :]
    lower_img = cv2.flip(lower_img, 0)
    lower_triangles = detect_triangles(lower_img)
    if len(lower_triangles) != 12:
        return None

    margin = detect_margin(upper_triangles + lower_triangles)
    lower_x = get_grid_points(img, lower_triangles)
    lower_x.insert(0, margin//2)
    lower_x.insert(6, img.shape[1]//2 - margin)
    lower_x.insert(7, img.shape[1]//2 + margin)
    lower_x.append(img.shape[1] - margin//2)
    for i in range(len(lower_x) - 1, 0, -1):
      # print(f"tile {val}: {lower_x[i-1]}, {lower_x[i]}  ;   {i}")
      if i == 7:
        continue
      grid[img.shape[0]//2:img.shape[0]-margin, lower_x[i-1]:lower_x[i]] = val
      val += 1

    upper_x = get_grid_points(img, upper_triangles)
    upper_x.insert(0, margin//2)
    upper_x.insert(6, img.shape[1]//2 - margin)
    upper_x.insert(7, img.shape[1]//2 + margin)
    upper_x.append(img.shape[1] - margin//2)
    for i in range(1, len(upper_x)):
      # print(f"tile {val}: {upper_x[i-1]}, {upper_x[i]}  ;   {i}")
      if i == 7:
        continue
      grid[margin:img.shape[0]//2, upper_x[i-1]:upper_x[i]] = val
      val += 1

    grid[margin:img.shape[0] - margin , img.shape[1]//2 - margin:img.shape[1]//2 + margin] = val # val = 25
    if val == 25:
        return grid * (255 // val)

    return None

"""Logic ------------------------------------------------------------------------------------------------------------"""
def get_game_state(grid, white_checkers, black_checkers):
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2GRAY) if len(grid.shape) == 3 else grid
    grid = grid // 10 # 255//24=10
    game_state = np.zeros(26, dtype = np.int8)
    for (x, y) in white_checkers:
        tile = grid[y, x]
        if tile < 25:
            game_state[tile] += 1
        else:
            game_state[0] += 1

    for (x, y) in black_checkers:
        tile = grid[y, x]
        if tile < 25:
            game_state[tile] -= 1
        else:
            game_state[25] -= 1

    return game_state

def get_start_player(game):
    dice_numbers = game["curr_dice"]

    if game["restart_throw"]:
        if len(dice_numbers) == 0:
            game["restart_throw"] = False
            # Show error to remove dice
        return game

    if len(dice_numbers) == 1 and game["white_throw"] is None:
        if game["frame_buffer"] < 3:
            game["frame_buffer"] += 1
            return game
        else:
            game["white_throw"] = dice_numbers[0]
            game["frame_buffer"] = 0
            print(f"White: {game["white_throw"]}, Black: {game["black_throw"]}")
            game["text"] = f"White: {game["white_throw"]}, Black: {game["black_throw"]}"
        return game

    elif len(dice_numbers) == 2 and game["black_throw"] is None and game["white_throw"] is not None:
        if game["frame_buffer"] < 3:
            game["frame_buffer"] += 1
            return game
        else:
            dice_numbers.remove(game["white_throw"])
            game["black_throw"] = dice_numbers[0]
            game["frame_buffer"] = 0
            print(f"White: {game["white_throw"]}, Black: {game["black_throw"]}")
            game["text"] = f"White: {game["white_throw"]}, Black: {game["black_throw"]}"
        return game

    if game["white_throw"] is not None and game["black_throw"] is not None:
        if game["black_throw"] == game["white_throw"]:
            game["black_throw"], game["white_throw"] = None, None
            game["restart_throw"] = True
            print("Throw again dice")
            game["text"] = "Throw again dice"
        else:
            game["turn"] = "black" if game["black_throw"] > game["white_throw"] else "white" # Who starts
            game["mode"] = "play game"
            print(f"{game["turn"]} starts!")
            game["text"] = f"{game["turn"]} starts!"
        return game

    return game

def is_legal_state(game):
    # True if found all checkers
    # curr_state, prev_state = game["curr_game_state"].copy(), game["prev_game_state"].copy()
    game_state = game["curr_game_state"]
    if np.sum(game_state[game_state > 0]) == game["checkers_count"][0] and abs(np.sum(game_state[game_state < 0])) == game["checkers_count"][1]:
        # if game["turn"] == "white":
        #     curr_state[curr_state > 0] = 0
        #     prev_state[prev_state > 0] = 0
        #     curr_state, prev_state = np.abs(curr_state[::-1]), np.abs(prev_state[::-1])
        #     change_state = curr_state - prev_state
        #     # print(change_state)
        #     # print(26 * [0])
        #     if np.array_equal(change_state, 26 * [0]) or (change_state[0] == -np.sum(change_state < 0)):
        #         return True
        # else:
        #     curr_state[curr_state < 0] = 0
        #     prev_state[prev_state < 0] = 0
        #     curr_state, prev_state = np.abs(curr_state[::-1]), np.abs(prev_state[::-1])
        #     change_state = curr_state - prev_state
        #     # print(change_state)
        #     # print(26 * [0])
        #     if np.array_equal(change_state, 26 * [0]) or (change_state[0] == -np.sum(change_state < 0)):
        #         return True
        return True
    return False

def is_legal_state_bear_off(game):
    # curr_state, prev_state = game["curr_game_state"].copy(), game["prev_game_state"].copy()
    # True if found all checkers
    dice = game["turn_dice"]
    game_state = game["curr_game_state"]
    # player = 0 if game["turn"] == "white" else 1
    max_bear_off = 2 if dice[0] != dice[1] else 4
    if game["turn"] == "white": # white
        if np.sum(game_state[game_state > 0]) >= game["checkers_count"][0] - max_bear_off and abs(np.sum(game_state[game_state < 0])) == game["checkers_count"][1]:
            # curr_state[curr_state > 0] = 0
            # prev_state[prev_state > 0] = 0
            # curr_state, prev_state = np.abs(curr_state[::-1]), np.abs(prev_state[::-1])
            # change_state = curr_state - prev_state
            # if np.array_equal(change_state, 26 * [0]) or (change_state[0] == -np.sum(change_state < 0)):
            #     return True
            return True
    else: # black
        if np.sum(game_state[game_state > 0]) == game["checkers_count"][0] and abs(np.sum(game_state[game_state < 0])) >= game["checkers_count"][1] - max_bear_off:
            # curr_state[curr_state < 0] = 0
            # prev_state[prev_state < 0] = 0
            # curr_state, prev_state = np.abs(curr_state[::-1]), np.abs(prev_state[::-1])
            # change_state = curr_state - prev_state
            # if np.array_equal(change_state, 26 * [0]) or (change_state[0] == -np.sum(change_state < 0)):
            #     return True
            # else:
            #     change_state
            return True

    return False

def get_change_state(game):
    curr_state, prev_state = game["curr_game_state"].copy(), game["prev_game_state"].copy()
    # print(f"curr: {curr_state}")
    # print(f"prev: {prev_state}")
    if game["turn"] == "white":
        curr_state[curr_state < 0] = 0
        prev_state[prev_state < 0] = 0
    else:
        curr_state[curr_state > 0] = 0
        prev_state[prev_state > 0] = 0
        curr_state, prev_state = np.abs(curr_state[::-1]), np.abs(prev_state[::-1])

    return curr_state - prev_state

def get_available_tiles(game):
    player_checkers, available_tiles = game["prev_game_state"].copy(), game["prev_game_state"].copy()
    if game["turn"] == "white":
        player_checkers[player_checkers < 0] = 0
        # available_tiles = np.where((available_tiles == 0) | (available_tiles == -1), 1, 0)
        available_tiles = np.where(available_tiles > - 2, 1, 0)
        available_tiles[0] = available_tiles[25] = 0
    else:
        player_checkers[player_checkers > 0] = 0
        player_checkers = np.abs(player_checkers[::-1])
        # available_tiles = np.where((available_tiles == 0) | (available_tiles == 1) | , 1, 0)[::-1]
        available_tiles = np.where(available_tiles < 2, 1, 0)[::-1]
        available_tiles[0] = available_tiles[25] = 0

    return player_checkers, available_tiles

def can_bear_off(game):
    if len(game["turn_dice"]) != 2:
        missing_dice_txt = f"Dice missing"
        if missing_dice_txt != game["missing_dice_txt"]:
            game["missing_dice_txt"] = missing_dice_txt
            print(missing_dice_txt)
        return False

    player = game["turn"]
    dice = game["turn_dice"]
    player_checkers, available_tiles = get_available_tiles(game)

    steps = dice if dice[0] != dice[1] else 4 * [dice[0]]
    steps = sorted(steps, reverse=True)[:-1]
    if 0 < np.sum(player_checkers[:19]) <= 3:
        for step in steps:
            for i in range(19):
                if player_checkers[i] > 0:
                    if available_tiles[i + step] == 1:
                        player_checkers[i + step] += 1
                        player_checkers[i] -= 1
                        break
    if np.sum(player_checkers[:19]) == 0:
        # game["can_bear_txt"][player] = True
        return True
    return False

def is_available_moves_bear_off(game):
    # Changed curr_dice to turn_dice
    if len(game["turn_dice"]) != 2:
        # print("[ERROR] Dice missing")
        return False

    player_checkers, available_tiles = get_available_tiles(game)
    # print(f"Player: {player_checkers}")
    # print(f"Available: {available_tiles}")
    dice = game["turn_dice"]
    steps_used = 0
    step_options = []
    options = []
    bar_checkers = player_checkers[0]
    # print(player_checkers)
    # print(bar_checkers)

    if dice[0] == dice[1]:  # Double dice roll
        num = dice[0]
        options.append([4 * num])
        options.append([num, 3 * num])
        options.append([2 * num, 2 * num])
        options.append([num, num, 2 * num])
        options.append(4 * [num])
    else:  # Regular dice roll
        options.append([sum(dice)])
        options.append(dice)
    # print(options)
    moves = []
    idx = -1
    # print(player_checkers)
    for s, option in enumerate(options):
        for k, perm in enumerate(set(permutations(option))):
            check_lst = player_checkers.copy()
            moves.append(0)
            idx += 1
            # print(f"Perm: {perm}")
            # if perm == (5, 5, 5, 5):
                # print(f"Perm: {perm}")
            for step in perm:
                for i in range(25):
                    if check_lst[i] > 0:
                        if i + step < 25 and available_tiles[i + step] == 1:
                            valid_move = True
                            if dice[0] == dice[1]:
                                num = dice[0]
                                for j in range(1, step // num):
                                    # print(f"start: {i}, iter: {j}")
                                    if available_tiles[i + j * num] == 0:
                                        valid_move = False
                                        # print(f"check tile: {i + j * num}, {available_tiles[i + j * num]}, is_valid: {valid_move}")
                                        break
                                if valid_move:
                                    moves[idx] += step
                                    check_lst[i] -= 1
                                    check_lst[i + step] += 1
                                    # print(f"move: {(i, i+step)}")
                                    break
                            else:
                                if available_tiles[i + dice[0]] == 0 and available_tiles[i + dice[1]] == 0:
                                    valid_move = False
                                if valid_move:
                                    moves[idx] += step
                                    check_lst[i] -= 1
                                    check_lst[i + step] += 1
                                    break
                        elif i + step > 24:
                            valid_move = True
                            if dice[0] == dice[1]:
                                num = dice[0]
                                for j in range(1, step // num):
                                    tile = i + j * num
                                    if tile < 25:
                                        if available_tiles[tile] == 0:
                                            valid_move = False
                                            break
                                    else:
                                        break
                                if valid_move:
                                    moves[idx] += step
                                    check_lst[i] -= 1
                                    break
                            else:
                                try:
                                    if available_tiles[i + dice[0]] == 0 and available_tiles[i + dice[1]] == 0:
                                        valid_move = False
                                except:
                                    valid_move = True
                                if valid_move:
                                    moves[idx] += step
                                    check_lst[i] -= 1
                                    break

    steps_used = (max(move for move in moves))
    if dice[0] == dice[1]:
        return (steps_used // dice[0]) * [dice[0]]
    else:
        if steps_used == sum(dice):
            return dice
        else:
            if steps_used == 0:
                return []
            return [steps_used]

def max_bear_off(game):
    player_checkers, available_tiles = get_available_tiles(game)
    dice = game["turn_dice"]
    options = []
    # print(player_checkers)
    steps = dice if dice[0] != dice[1] else 4 * [dice[0]]
    steps = sorted(steps, reverse=True)[:-1]
    used_steps = 0
    if 0 < np.sum(player_checkers[:19]) <= 3:
        for step in steps:
            for i in range(19):
                if player_checkers[i] > 0:
                    if available_tiles[i + step] == 1:
                        used_steps += step
                        player_checkers[i + step] += 1
                        player_checkers[i] -= 1
                        break

    if dice[0] == dice[1]:
        num = dice[0]
        remain_steps = 4 - used_steps // dice[0]
        if remain_steps == 4:
            options.append([4 * num])
            options.append([num, 3 * num])
            options.append([2 * num, 2 * num])
            options.append([num, num, 2 * num])
            options.append(4 * [num])
        if remain_steps == 3:
            options.append([3 * num])
            options.append([num, 2 * num])
            options.append(3 * [num])
        if remain_steps == 2:
            options.append([2 * num])
            options.append(2 * [num])
        if remain_steps == 1:
            options.append([num])
    else:
        remain_steps = sum(dice) - used_steps
        if remain_steps == sum(dice):
            options.append([sum(dice)])
            options.append(dice)
        else:
            options.append([remain_steps])

    moves = []
    bear_off = []
    idx = -1
    for s, option in enumerate(options):
        for k, perm in enumerate(set(permutations(option))):
            # print(f"Permutation: {perm}")
            check_lst = player_checkers.copy()
            moves.append(0)
            bear_off.append(0)
            idx += 1
            for step in perm:
                t = 25 - step
                # print(f" Checking tile {t}")
                if check_lst[t] > 0:
                    # print("  Found checker in tile")
                    valid_move = True
                    if dice[0] == dice[1]:
                        num = dice[0]
                        for j in range(1, step // num):
                            tile = t + j * num
                            if tile < 25:
                                if available_tiles[tile] == 0:
                                    valid_move = False
                                    break
                            else:
                                break
                        if valid_move:
                            moves[idx] += step
                            check_lst[t] -= 1
                            bear_off[idx] += 1
                            # print(t, "Bear off")
                    else:
                        try:
                            if available_tiles[t + dice[0]] == 0 and available_tiles[t + dice[1]] == 0:
                                valid_move = False
                        except:
                            valid_move = True
                        if valid_move:
                            moves[idx] += step
                            check_lst[t] -= 1
                            bear_off[idx] += 1
                else: # not a perfect bear off
                    for i in range(19, 25):
                        if check_lst[i] > 0 and i + step > 24 and sum(check_lst[19:i]) == 0:
                            # print("  Found checker in tile")
                            valid_move = True
                            if dice[0] == dice[1]:
                                num = dice[0]
                                for j in range(1, step // num):
                                    tile = t + j * num
                                    if tile < 25:
                                        if available_tiles[tile] == 0:
                                            valid_move = False
                                            break
                                    else:
                                        break
                                if valid_move:
                                    moves[idx] += step
                                    check_lst[t] -= 1
                                    bear_off[idx] += 1
                                    # print(t, "Bear off")
                                    break
                            else:
                                try:
                                    if available_tiles[t + dice[0]] == 0 and available_tiles[t + dice[1]] == 0:
                                        valid_move = False
                                except:
                                    valid_move = True
                                if valid_move:
                                    moves[idx] += step
                                    check_lst[t] -= 1
                                    bear_off[idx] += 1
                                    break

    player = 0 if game["turn"] == "white" else 1
    return min(max(bear_off), game["checkers_count"][player])

def is_legal_move_bear_off(game):
    # Check if dice are visible
    if len(game["turn_dice"]) != 2:
        missing_dice_txt = f"Dice missing"
        if missing_dice_txt != game["missing_dice_txt"]:
            game["missing_dice_txt"] = missing_dice_txt
            print(missing_dice_txt)
        return False

    dice = game["turn_dice"]
    change_state = get_change_state(game)
    # print(f"Prev state: {game["prev_game_state"]}")
    # print(f"Curr state: {game["curr_game_state"]}")
    # print(f"Change state: {change_state}")
    change_state_txt = f"Change state: {change_state}"
    if change_state_txt != game["change_state_txt"]:
        game["change_state_txt"] = change_state_txt
        print(change_state_txt)
    moved_checkers = abs(np.sum(change_state[change_state < 0]))

    player_checkers, _ = get_available_tiles(game)
    bar_checkers = player_checkers[0]
    if bar_checkers > 0 and change_state[0] == 0:
        error_txt = f"❌ No available moves! switching to player {game["turn"]}"
        if error_txt != game["last_error"]:
            game["last_error"] = error_txt
            print(error_txt)
        return False
    steps = is_available_moves_bear_off(game)
    # print(len(steps), moved_checkers)
    options = []
    if any(isinstance(sub, list) for sub in steps): # 1 bar, 1 dice
        options = steps
    else:
        num = dice[0]
        if len(steps) == 4:
            if moved_checkers == 1:
                options.append([4 * num])
            elif moved_checkers == 2:
                options.append([num, 3 * num])
                options.append([2 * num, 2 * num])
            elif moved_checkers == 3:
                options.append([num, num, 2 * num])
            elif moved_checkers == 4:
                options.append(4 * [num])
        elif len(steps) == 3:
            if moved_checkers == 1:
                options.append([3 * num])
            elif moved_checkers == 2:
                options.append([num, 2 * num])
            elif moved_checkers == 3:
                options.append(3 * [num])
        elif len(steps) == 2:
            if moved_checkers == 1:
                options.append([sum(steps)])
                # options.append([steps[0]])
                # options.append([steps[1]])
            elif moved_checkers == 2:
                options.append(steps)
        elif len(steps) == 1:
            if moved_checkers == 1:
                options.append(steps)

    options_txt = f"Available steps: {steps} ; Dice options: {options}"
    if options_txt != game["options_txt"]:
        game["options_txt"] = options_txt
        print(options_txt)

    moves = []
    idx = -1
    for option in options:
        for k, perm in enumerate(set(permutations(option))):
            moves.append([])
            idx += 1
            state = change_state.copy()
            for step in perm:
                for i in range(25):
                    if state[i] < 0:
                        if i + step < 25 and state[i + step] > 0:
                            moves[idx].append((i, i + step))
                            state[i] += 1
                            state[i + step] -= 1
                            break
                        elif i + step > 24:
                            moves[idx].append((i, "bear_off"))
                            state[i] += 1
                            break

    chosen_move = None
    num_bear_off = max_bear_off(game)
    for move in moves:
        if len(move) == moved_checkers:
            if sum(1 for step in move if step[1] == "bear_off") == num_bear_off:
                # print(num_bear_off, sum(1 for step in move if step[1] == "bear_off"), move)
                chosen_move = move
                break

    if chosen_move is not None:
        # Edit eaten checkers
        player = 0 if game["turn"] == "white" else 1
        game["checkers_count"][player] -= num_bear_off
        print(f"{game["turn"]} move: {chosen_move}")
        print(f"Bear off {num_bear_off} checkers, {game["checkers_count"][player]} checkers remaining")
        return True

    return False

def is_available_moves(game):
    # Changed curr_dice to turn_dice
    if len(game["turn_dice"]) != 2:
        # print("[ERROR] Dice missing")
        return False

    player_checkers, available_tiles = get_available_tiles(game)
    # print(f"Player: {player_checkers}")
    # print(f"Available: {available_tiles}")
    dice = game["turn_dice"]
    steps_used = 0
    step_options = []
    options = []
    bar_checkers = player_checkers[0]
    # print(player_checkers)
    # print(bar_checkers)
    if bar_checkers > 0:
        if dice[0] == dice[1]:
            num = dice[0]
            remain_steps = 0
            if available_tiles[num] == 1:
                steps_used = bar_checkers * num # Can't eat more than 5 - otherwise use np.clip
                remain_steps = 4 - bar_checkers
                player_checkers[num] = bar_checkers

            if remain_steps == 3:
                options.append([3 * num])
                options.append([num, 2 * num])
                options.append(3 * [num])
            elif remain_steps == 2:
                options.append([2 * num])
                options.append([2 * [num]])
            elif remain_steps == 1:
                options.append([num])

            moves = []
            idx = -1
            for option in options:
                for k, perm in enumerate(set(permutations(option))):
                    check_lst = player_checkers.copy()
                    moves.append(0)
                    idx += 1
                    for step in perm:
                        for i in range(25):
                            if check_lst[i] > 0:
                                if i + step < 25 and available_tiles[i + step] == 1:
                                    valid_move = True
                                    for j in range(1, step//num):
                                        if available_tiles[i + j * num] == 0:
                                            valid_move = False
                                            break
                                    if valid_move:
                                        moves[idx] += step
                                        check_lst[i] -= 1
                                        check_lst[i + step] += 1
                                        break
            try:
                total_steps = steps_used + max(move for move in moves)
            except:
                total_steps = 0
            return (total_steps // num) * [num]

        else:   # Different dice
            if bar_checkers == 2: # Can only eat up to 2 checkers
                if available_tiles[dice[0]] == 1 and available_tiles[dice[1]] == 1:
                    return dice
                elif available_tiles[dice[0]] == 1:
                    return [dice[0]]
                elif available_tiles[dice[1]] == 1:
                    return [dice[1]]
            else: # 1 in bar
                if available_tiles[dice[0]] == 1:
                    steps_used = [dice[0]]
                    player_checkers[dice[0]] += 1
                    step = dice[1]
                    for i in range(25):
                        if player_checkers[i] > 0:
                            if i + step < 25 and available_tiles[i + step] == 1:
                                return dice
                    step_options.append(steps_used)

                if available_tiles[dice[1]] == 1:
                    steps_used = [dice[1]]
                    player_checkers[dice[1]] += 1
                    step = dice[0]
                    for i in range(25):
                        if player_checkers[i] > 0:
                            if i + step < 25 and available_tiles[i + step] == 1:
                                return dice
                    step_options.append(steps_used)

                return step_options

    if dice[0] == dice[1]:  # Double dice roll
        num = dice[0]
        options.append([4 * num])
        options.append([num, 3 * num])
        options.append([2 * num, 2 * num])
        options.append([num, num, 2 * num])
        options.append(4 * [num])
    else:  # Regular dice roll
        options.append([sum(dice)])
        options.append(dice)
    # print(options)
    moves = []
    idx = -1
    # print(player_checkers)
    for s, option in enumerate(options):
        for k, perm in enumerate(set(permutations(option))):
            check_lst = player_checkers.copy()
            moves.append(0)
            idx += 1
            # print(f"Perm: {perm}")
            # if perm == (5, 5, 5, 5):
                # print(f"Perm: {perm}")
            for step in perm:
                for i in range(25):
                    if check_lst[i] > 0:
                        if i + step < 25 and available_tiles[i + step] == 1:
                            valid_move = True
                            if dice[0] == dice[1]:
                                num = dice[0]
                                for j in range(1, step // num):
                                    # print(f"start: {i}, iter: {j}")
                                    if available_tiles[i + j * num] == 0:
                                        valid_move = False
                                        # print(f"check tile: {i + j * num}, {available_tiles[i + j * num]}, is_valid: {valid_move}")
                                        break
                                if valid_move:
                                    moves[idx] += step
                                    check_lst[i] -= 1
                                    check_lst[i + step] += 1
                                    # print(f"move: {(i, i+step)}")
                                    break
                            else:
                                if available_tiles[i + dice[0]] == 0 and available_tiles[i + dice[1]] == 0:
                                    valid_move = False
                                if valid_move:
                                    moves[idx] += step
                                    check_lst[i] -= 1
                                    check_lst[i + step] += 1
                                    break
            # if perm == (5, 5, 5, 5):
            # print(perm, moves[idx])

    steps_used = (max(move for move in moves))
    # print(f"Chose: {steps_used}")
    # print("FINISHED CHECK")
    if dice[0] == dice[1]:
        return (steps_used // dice[0]) * [dice[0]]
    else:
        if steps_used == sum(dice):
            return dice
        else:
            if steps_used == 0:
                return []
            return [steps_used]

def is_legal_move(game):
    # Check if dice are visible
    if len(game["turn_dice"]) != 2:
        missing_dice_txt = f"Dice missing"
        if missing_dice_txt != game["missing_dice_txt"]:
            game["missing_dice_txt"] = missing_dice_txt
            print(missing_dice_txt)
        return False

    dice = game["turn_dice"]
    change_state = get_change_state(game)
    # print(f"Prev state: {game["prev_game_state"]}")
    # print(f"Curr state: {game["curr_game_state"]}")
    # print(f"Change state: {change_state}")
    change_state_txt = f"Change state: {change_state}"
    if change_state_txt != game["change_state_txt"]:
        game["change_state_txt"] = change_state_txt
        print(change_state_txt)
    moved_checkers = np.sum(change_state[change_state > 0])

    player_checkers, _ = get_available_tiles(game)
    bar_checkers = player_checkers[0]
    if bar_checkers > 0 and change_state[0] == 0:
        error_txt = f"❌ No available moves! switching to player {game["turn"]}"
        if error_txt != game["last_error"]:
            game["last_error"] = error_txt
            print(error_txt)
        return False
    steps = is_available_moves(game)
    options = []
    if any(isinstance(sub, list) for sub in steps): # 1 bear off, 1 dice
        options = steps
    else:
        num = dice[0]
        if len(steps) == 4:
            if moved_checkers == 1:
                options.append([4 * num])
            elif moved_checkers == 2:
                options.append([num, 3 * num])
                options.append([2 * num, 2 * num])
            elif moved_checkers == 3:
                options.append([num, num, 2 * num])
            elif moved_checkers == 4:
                options.append(4 * [num])
        elif len(steps) == 3:
            if moved_checkers == 1:
                options.append([3 * num])
            elif moved_checkers == 2:
                options.append([num, 2 * num])
            elif moved_checkers == 3:
                options.append(3 * [num])
        elif len(steps) == 2:
            if moved_checkers == 1:
                options.append([sum(steps)])
            elif moved_checkers == 2:
                options.append(steps)
        elif len(steps) == 1:
            if moved_checkers == 1:
                options.append(steps)

    options_txt = f"Available steps: {steps} ; Dice options: {options}"
    if options_txt != game["options_txt"]:
        game["options_txt"] = options_txt
        print(options_txt)

    moves = []
    idx = -1
    for option in options:
        for k, perm in enumerate(set(permutations(option))):
            moves.append([])
            idx += 1
            state = change_state.copy()
            for step in perm:
                for i in range(25):
                    if state[i] < 0:
                        if i + step < 25 and state[i + step] > 0:
                            moves[idx].append((i, i + step))
                            state[i] += 1
                            state[i + step] -= 1
                            break

    chosen_move = None
    for move in moves:
        if len(move) == moved_checkers:
            chosen_move = move
            break

    if chosen_move is not None:
        # Edit eaten checkers
        print(f"{game["turn"]} move: {chosen_move}")
        return True

    # print("No legal moves")
    return False

def is_new_state(game):
    if not np.array_equal(game["prev_game_state"], game["curr_game_state"]):
        return True
    return False

def update_game_state(game):
    game["prev_game_state"] = game["curr_game_state"]
    return game

def init_game():
    game = {
        "prev_game_state": np.array([0,  2,  0,  0,  0,  0, -5,  0, -3,  0,  0,  0,  5, -5,  0,  0,  0,  3,  0,  5,  0,  0,  0,  0, -2,  0]), #[np.zeros(26, dtype=np.int8),
        "curr_game_state" : np.zeros(26, dtype=np.int8),
        "prev_dice" : [],
        "curr_dice" : [],
        "checkers_count" : [15, 15],
        "mode" : "create grid", # "create grid" / "get start player" / "play game" / "bear off game"
        "turn" : "", # Change
        "restart" : False,
        "white_throw" : None,
        "black_throw" : None,
        "restart_throw" : False,
        "frame_buffer" : 0,
        "last_corners" : None,
        "board_buffer" : 0,
        "last_error" : "",
        "legal_state_buffer" : 0,
        "dice_buffer" : 0,
        "options_txt" : "",
        "change_state_txt" : "",
        "missing_dice_txt" : "",
        "turn_dice" : [],
        "can_bear_txt" : ["",""],
        "text" : "",
        "last_valid_board" : None
    }
    return game

"""Main -------------------------------------------------------------------------------------------------------------"""
def main():
    # data_dir = "/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/webcam_frames_5/"
    # blank_board = cv2.imread(os.path.join(data_dir, "blank_board.jpg"), cv2.IMREAD_GRAYSCALE)
    # grid = create_grid(blank_board) # Change to dynamic creation
    print("start")
    # Initiate Game
    game = init_game()
    # game_board = blank_board # Need to Delete
    # game["prev_game_state"] = np.array([ 0, -1, -1, 0, -1, -2, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  1,  0,  0, 0, 0,  0])
    # game["turn"] = "black"
    # game["checkers_count"] = [1, 5]
    # Start Processing
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect game board
        aligned_board, edges_img = detect_board(frame, game)

        if aligned_board is not None: # Successfully detected board
            # Create grid at the start of the game
            if game["mode"] == "create grid":
                grid = create_grid(aligned_board)
                if grid is not None:
                    game["mode"] = "get start player"
                    game_board = aligned_board
                    game["text"] = "white throw dice"
                # print("Can't Find Grid")
                continue

            # Align grid to detected board
            grid_show = scale_image(aligned_board, grid)

            # Detect checkers
            try:
                white_mask = white_threshold(aligned_board)
                black_mask = black_threshold(aligned_board)
                aligned_board, white_checkers, black_checkers = segment_checkers(aligned_board, white_mask, black_mask)
            except:
                # print("error")
                a = 1
                continue
            dice_images, game["curr_dice"] = detect_dice(aligned_board, white_mask, black_mask, white_checkers)
            if sorted(game["curr_dice"]) != sorted(game["turn_dice"]) and len(game["curr_dice"]) == 2:
                if game["dice_buffer"] < 3:
                    game["dice_buffer"] += 1
                else:
                    game["dice_buffer"] = 0
                    game["turn_dice"] = game["curr_dice"]
                    print(f"Turn dice: {game["turn_dice"]}")
            # if len(game["curr_dice"]) == 2:
            #     game["prev_dice"] = game["curr_dice"]
            # elif len(game["curr_dice"]) == 0:
            #     game["curr_dice"] = game["prev_dice"]

            # Get starting player in the start of the game
            if game["mode"] == "get start player":
                game = get_start_player(game)
                # if not is_legal_state(game):
                # aligned_board = game_board
                white_checkers, black_checkers = [], []

            # Start game!
            if game["mode"] == "play game":
                game["curr_game_state"] = get_game_state(grid_show, white_checkers, black_checkers)
                # print(f"PRINT GAME STATE: {game["curr_game_state"]}")
                player = 0 if game["turn"] == "white" else 1
                if can_bear_off(game):
                    can_bear_txt = f"{game["turn"]} can bear off with {max_bear_off(game)} checkers"
                    if can_bear_txt != game["can_bear_txt"][player]:
                        game["can_bear_txt"][player] = can_bear_txt
                        print(can_bear_txt)
                    if is_legal_state_bear_off(game):
                        # print("yes")
                        game["legal_state_buffer"] = 0
                        if len(game["turn_dice"]) == 2:  # changed from curr_dice
                            if is_available_moves_bear_off(game):
                                if is_new_state(game):
                                    if is_legal_move_bear_off(game):
                                        if game["checkers_count"][player] == 0:
                                            print(f"🎉 {game["turn"]} won!")
                                            break
                                        game["turn"] = "white" if game["turn"] == "black" else "black"
                                        game["turn_dice"] = []
                                        game = update_game_state(game)
                                        game_board = aligned_board
                                        print(f"New Game State: {game["curr_game_state"]}")
                                        game["can_bear_txt"][player] = ""

                            else:
                                game["turn"] = "white" if game["turn"] == "black" else "black"
                                error_txt = f"❌ No available moves! switching to player {game["turn"]}"
                                if error_txt != game["last_error"]:
                                    game["last_error"] = error_txt
                                    print(error_txt)
                    else:
                        game["legal_state_buffer"] += 1

                elif is_legal_state(game):
                    game["legal_state_buffer"] = 0
                    if len(game["turn_dice"]) == 2: # changed from curr_dice
                        if is_available_moves(game):
                            if is_new_state(game):
                                if is_legal_move(game):
                                    game["turn"] = "white" if game["turn"] == "black" else "black"
                                    game["turn_dice"] = []
                                    game = update_game_state(game)
                                    game_board = aligned_board
                                    print(f"New Game State: {game["curr_game_state"]}")
                        else:
                            game["turn"] = "white" if game["turn"] == "black" else "black"
                            error_txt = f"❌ No available moves! switching to player {game["turn"]}"
                            if error_txt != game["last_error"]:
                                game["last_error"] = error_txt
                                print(error_txt)
                else:
                    game["legal_state_buffer"] += 1


                    # print(f"No Legal state for {game["legal_state_buffer"]} frames")

                # if is_legal_state(game) and is_new_state(game) and is_available_moves(game) and is_legal_move(game):
                #     game["turn"] = "white" if game["turn"] == "black" else "black"
                #     game = update_game_state(game)
                #     game_board = aligned_board
                #     print(f"New Game State: {game["curr_game_state"]}")

            # Show detection
            for (x, y) in white_checkers:
                cv2.circle(grid_show, (x, y), 3, (0, 0, 255), cv2.FILLED)  # Green circles
            for (x, y) in black_checkers:
                cv2.circle(grid_show, (x, y), 3, (0, 255, 0), cv2.FILLED)  # Green circles
            dice_show = show_dice(dice_images, game["curr_dice"])
            # print(dice_show)
            # board_show = concat_images(game_board, concat_images(aligned_board, grid_show))
            text = show_text(f"{game["turn"]} {game["turn_dice"]}")
            # text = show_text(game["text"])
            if dice_show is not None:
                canvas = show_game(game_board, aligned_board, grid_show, text, dice_show)
                cv2.imshow("Aligned Backgammon Board", canvas)#concat_images(board_show, dice_show))
            else:
                canvas = show_game(game_board, aligned_board, grid_show, text, np.zeros((1, 1)))
                cv2.imshow("Aligned Backgammon Board", canvas)
            # cv2.imshow("Aligned Backgammon Board", board_show)
            # except:
            #     print("error")
                # continue
        cv2.imshow("Backgammon Board Detection", concat_images(frame, edges_img))

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()