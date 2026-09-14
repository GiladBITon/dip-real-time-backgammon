import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as ssim

def show_game(last_verified_state, real_time_state):#, grid, text_prompt, dice_images):
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
    # grid = fit_to_window((500, 500), grid)
    # dice_images = fit_to_window((500, 500), dice_images)
    # text_prompt = fit_to_window((500, 500), text_prompt)
    canvas = concat_images(last_verified_state, real_time_state)
    # canvas = concat_images(canvas, grid)
    # canvas = concat_images(canvas, dice_images)
    # canvas = concat_images(canvas, text_prompt)

    return canvas

def find_board_by_corners(frame, data):
    board_corners = data["board_corners"]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray32 = np.float32(gray)
    dst = cv2.cornerHarris(gray32, 10, 3, 0.04)
    dst = cv2.dilate(dst, None)
    threshold = 0.0001 * dst.max()
    # frame[dst > threshold] = [255, 255, 0]
    corner_candidates = np.argwhere(dst > threshold)
    corner_candidates = np.array(corner_candidates)
    exact_board_corners = np.array(board_corners)

    for i, corner in enumerate(board_corners):
        swapped_corner = corner[::-1]
        distances = np.linalg.norm(corner_candidates - swapped_corner, axis=1)
        # if np.min(distances) < 20:
        closest_point = corner_candidates[np.argmin(distances)]
        exact_board_corners[i] = (closest_point[::-1])

    # print(exact_board_corners.shape, type(exact_board_corners))
    # Convert to the correct format for OpenCV
    ordered_corners = exact_board_corners
    exact_board_corners = exact_board_corners.reshape((-1, 1, 2)).astype(np.int32)

    # Compute the width and height of the new image
    (tl, tr, br, bl) = ordered_corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)

    if max(width_top, width_bottom) < 100 or max(height_left, height_right) < 100:
        # cv2.drawContours(frame, [board_corners], -1, (0, 255, 0), 3)  # Green line
        return frame, None

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
    cv2.drawContours(frame, [exact_board_corners], -1, (0, 0, 255), 3)  # Green line
    data["board_corners"] = ordered_corners
    return frame, warped

def detect_board(frame, data):
    # Step 1: Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 3)  # Kernel size of 3

    # Step 2: Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 3: Morphological Closing to connect lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)  # Expand edges slightly changed from 2 to 1
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
    # detected_corners = refine_board_corners(frame, board_corners)
    # cv2.drawContours(frame, [detected_corners], -1, (0, 0, 255), 3)  # Green line

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

    # if game["legal_state_buffer"] > 2:
    #     game["board_corners"] = ordered_corners
    #     game["legal_state_buffer"] = 0
    # else:

    # if data["board_corners"] is None:
    #     data["board_corners"] = ordered_corners
    # else:
    #     distances = np.linalg.norm(ordered_corners - data["board_corners"], axis=1)
    #     moved_corners = np.sum(distances > 5)
    #     if moved_corners >= 3:
    #         if data["board_buffer"] < 2:
    #             data["board_buffer"] += 1
    #             ordered_corners = data["board_corners"]
    #         else:
    #             data["board_buffer"] = 0
    #             data["board_corners"] = ordered_corners
    #     else:
    #         ordered_corners = data["board_corners"]
    if data["board_corners"] is None:
        data["board_corners"] = ordered_corners
    distances = np.linalg.norm(ordered_corners - data["board_corners"], axis=1)
    moved_corners = np.sum(distances > 5)
    # print(moved_corners, distances.flatten())
    # if moved_corners >= 3:
    #     data["board_corners"] = ordered_corners
    # else:
    #     ordered_corners = data["board_corners"]

    # Compute the width and height of the new image
    (tl, tr, br, bl) = ordered_corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)

    if max(width_top, width_bottom) < 500 or max(height_left, height_right) < 500:
      # cv2.drawContours(frame, [board_corners], -1, (0, 255, 0), 3)  # Green line
      # print(data["board_corners"])
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
    data["board_corners"] = ordered_corners
    return warped, closed

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

def detect_triangles(img):
    # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Preprocessing
    edges = cv2.Canny(gray, 20, 100)
    # edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)  # Expand edges slightly changed from 2 to 1
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
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
    # cv2.imshow("Aligned Backgammon Board", concat_images(img, dilated))
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

def create_grid(img):
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

def detect_hand_and_analyze(image_empty, image_hand):
    # Resize both images to match dimensions
    width, height = image_empty.shape[:2]
    image_hand = cv2.resize(image_hand, (height, width))
    # Step 1: Reduce Image Size for Faster Processing
    resize_factor = 0.5  # Reduce image size to 50% (tune this value)
    image_empty = cv2.resize(image_empty, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    image_hand = cv2.resize(image_hand, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

    # Convert images to grayscale
    gray_empty = cv2.cvtColor(image_empty, cv2.COLOR_BGR2GRAY)
    gray_hand = cv2.cvtColor(image_hand, cv2.COLOR_BGR2GRAY)

    # Compute Optical Flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(gray_empty, gray_hand, None, 0.5, 2, 50, 1, 5, 1.2, 0)
    # lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # flow = cv2.calcOpticalFlowPyrLK(gray_empty, gray_hand, None, None, **lk_params)
    # Step 3: Detect Features to Track
    # prev_pts = cv2.goodFeaturesToTrack(gray_empty, maxCorners=500, qualityLevel=0.01, minDistance=5)
    # if prev_pts is None:
    #     print("No features detected in the reference frame.")
    #     return None  # Return if no keypoints detected
    # # Convert to np.float32 for Optical Flow
    # prev_pts = np.float32(prev_pts)
    # # Step 4: Compute Optical Flow Using Lucas-Kanade (Pyramidal LK)
    # lk_params = dict(winSize=(15, 15), maxLevel=2,
    #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # next_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_empty, gray_hand, prev_pts, None, **lk_params)
    # # Keep only valid keypoints
    # valid_prev_pts = prev_pts[status == 1]
    # valid_next_pts = next_pts[status == 1]
    #
    # # Step 5: Create a Motion Mask
    # motion_mask = np.zeros_like(gray_hand)
    # for (x1, y1), (x2, y2) in zip(valid_prev_pts, valid_next_pts):
    #     cv2.line(motion_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    #
    # # Step 6: Threshold Motion Mask
    # _, motion_thresh = cv2.threshold(motion_mask, 5, 255, cv2.THRESH_BINARY)

    # Create a motion mask
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Find contours of motion areas
    _, motion_thresh = cv2.threshold(motion_mask, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hand_data = {
        "bounding_box": None,
        "motion_mask": motion_mask,
        "lowest_convex_point": None,
        "lowest_concave_point": None,
        "processed_image": image_hand.copy()
    }

    # If motion is detected, create a Bounding Box around the hand
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(largest_contour)
        # # Scale Bounding Box Back to Original Size
        # # x, y, w, h = int(x / resize_factor), int(y / resize_factor), int(w / resize_factor), int(h / resize_factor)
        # # Store bounding box coordinates
        # hand_data["bounding_box"] = (x, y, w, h)
        # # Draw bounding box on the image
        # cv2.rectangle(hand_data["processed_image"], (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Get minimum bounding rectangle
        rect = cv2.minAreaRect(largest_contour)  # (center (x,y), (width, height), angle of rotation)
        box = cv2.boxPoints(rect)  # Get corner points of rectangle
        box = np.intp(box)
        # box = np.intp(box * (1 / resize_factor))  # Scale back to original size
        hand_data["min_area_rectangle"] = box  # Store rectangle points
        cv2.drawContours(hand_data["processed_image"], [box], 0, (0, 255, 0), 2)

    return hand_data

def detect_dice(img, white_mask, black_mask, white_checkers, dice_data):
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
    dice_xy = []
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
                    dice_xy.append(box[0])
                    cv2.drawContours(output, [contour], -1, (255, 255, 0), 3)
                else:
                    cut_dice = [dice_roi[:, :w // 2], dice_roi[:, w // 2:]] if w>h else [dice_roi[:h //2,:], dice_roi[h//2:,:]]
                    for dice in cut_dice:
                        size = int((0.05 * (dice.shape[0] + dice.shape[1]) / 2))
                        dice_images.append(dice[size:-size,size:-size])
                        dice_xy.append(box[0])

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
    for i, dice in enumerate(dice_images):
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
                dice_data["coordinates"][i] = dice_xy[i]
            # print(f"Correct: {len(dots)}, Predicted: {pred}")
        except:
            continue
        continue

    # print(len(dice_images))
    # if len(dice_images) > 1:
    #     cv2.imshow("Dice", concat_images(dice_images[0], dice_images[1]))
    dice_data["prev_frame_count"] = dice_data["curr_frame_count"]
    dice_data["curr_frame_count"] = min(len(dice_images), len(dice_numbers))
    return dice_images, dice_numbers

def black_threshold(img, n_bins = 64): # Image in BGR format
    blue_hist = cv2.calcHist([img], [2], None, [n_bins], [0, 256])
    # plt.plot(blue_hist, color='b', label="Blue"), plt.show()
    blue_hist_smooth = cv2.GaussianBlur(blue_hist, (5, 5), 0)  # Adjust kernel size as needed
    # plt.plot(blue_hist_smooth, color='b', label="Blue"), plt.show()
    peaks = find_peaks(blue_hist_smooth.flatten(), prominence=0.001)[0]  # Adjust prominence as needed
    if blue_hist_smooth.flatten()[0] > blue_hist_smooth.flatten()[1]: # Peak is at the start boundary
      black_thresh = (np.argmin(blue_hist_smooth[:peaks[0]])) * (256 / n_bins)
    else:
      black_thresh = (np.argmin(blue_hist_smooth[peaks[0]:peaks[1]]) + peaks[0]) * (256 / n_bins)
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

def segment_checkers(img, white_thresh, black_thresh):
    # Preprocessing
    output = img.copy()
    white_checkers = template_match(white_thresh, output, threshold=0.5, min_samples=1,
                                    contour_color=(0, 0, 255), checker_radius=int(img.shape[0] * 0.034))
    black_checkers = template_match(black_thresh, output, threshold=0.5, min_samples=1,
                                    contour_color=(0, 255, 0), checker_radius=int(img.shape[0] * 0.034))

    return output, white_checkers, black_checkers

def start_flow(dice_data, hand_data):
    try:
        x1, y1 = dice_data["coordinates"][0]
        x2, y2 = dice_data["coordinates"][1]
        xmin, ymin, xmax, ymax = hand_data["bbox"]
        # print(f"({x1},{y1}), ({x2},{y2}), {hand_data["bbox"]}, prev:{dice_data["prev_frame_count"]}, curr:{dice_data["curr_frame_count"]}")
        if dice_data["prev_frame_count"] > 0 and dice_data["curr_frame_count"] == 0:
            if xmin <= x1 <= xmax and xmin <= x2 <= xmax and ymin <= y1 <= ymax and ymin <= y2 <= ymax:
                hand_data["cover_dice"] = True
                print("Covered dice !!!")
                return
        elif dice_data["prev_frame_count"] == 0 and dice_data["curr_frame_count"] == 0 and hand_data["cover_dice"]:
            if not(xmin <= x1 <= xmax) and not(xmin <= x2 <= xmax) and not(ymin <= y1 <= ymax) and not(ymin <= y2 <= ymax):
                hand_data["show_flow"] = True
                player = "white" if ymin+abs((ymax-ymin)/2) < hand_data["half_board"]/2 else "black"
                _, ymin, _, ymax = hand_data["last_hand_bbox"]
                if ymin+abs((ymax-ymin)/2) < hand_data["half_board"]/2:
                    txt = f"🎉 white took the dice, {hand_data["last_hand_bbox"]}, {ymin+abs((ymax-ymin)/2), hand_data["half_board"]/2}"
                else:
                    txt = f"🎉 black took the dice, {hand_data["last_hand_bbox"]}, {ymin+abs((ymax-ymin)/2), hand_data["half_board"]/2}"
                # if hand_data["took_dice_txt"] != txt:
                print(txt)
                    # hand_data["took_dice_txt"] = txt
                return
    except:
        return
    return
    # if hand_data["show_flow"]:

def detect_hand(image_empty, image_hand, data, hand_data):
    # Resize both images to match dimensions
    height, width = image_empty.shape[:2]
    image_hand = cv2.resize(image_hand, (width, height))

    resize_factor = 0.5  # Reduce image size to 50% (tune this value)
    image_empty = cv2.resize(image_empty, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    image_hand = cv2.resize(image_hand, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

    # Convert images to grayscale
    gray_empty = cv2.cvtColor(image_empty, cv2.COLOR_BGR2GRAY)
    gray_hand = cv2.cvtColor(image_hand, cv2.COLOR_BGR2GRAY)

    # Compute Dense Optical Flow using Farneback
    flow = cv2.calcOpticalFlowFarneback(gray_empty, gray_hand, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude and angle of flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize magnitude to [0, 255]
    motion_mask = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply threshold to highlight movement
    _, motion_thresh = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)

    # Find contours of motion areas
    contours, _ = cv2.findContours(motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert angle to degrees (0-360 mapped to 0-180 for HSV)
    hsv = np.zeros_like(cv2.cvtColor(gray_empty, cv2.COLOR_GRAY2BGR))
    hsv[..., 1] = 255  # Full saturation (pure colors)

    # Map angle to hue (0-180)
    hsv[..., 0] = ang * (180 / np.pi / 2)  # Normalize to HSV range

    # Normalize magnitude to value channel (brightness)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for display
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    hand_data["motion_mask"] = flow_color
    hand_data["processed_image"] = image_hand.copy()

    # If motion is detected, find bounding shape
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get minimal bounding rectangle
        rect = cv2.minAreaRect(largest_contour)  # (center, (width, height), angle)
        box = cv2.boxPoints(rect)  # Get corner points
        box = np.intp(box)  # Convert to integer
        hand_data["min_area_rectangle"] = box

        x, y, w, h = cv2.boundingRect(largest_contour)

        # Store bounding box coordinates
        hand_data["bounding_box"] = (x, y, w, h)
        # Draw bounding box on the image

        # Draw rectangle around detected hand
        if w*h < 50000:
            # data["detect_counter"] += 1
            # if data["detect_counter"] > 3:
            hand_data["last_hand_bbox"] = hand_data["bbox"]
            hand_data["bbox"] = (x*2, y*2, (x+w)*2, (y+h)*2)
            # print(f"BBOX: {hand_data["bbox"]}")
            cv2.rectangle(hand_data["processed_image"], (x, y), (x + w, y + h), (255, 255, 0), 3)
            if hand_data["show_flow"]:
                cv2.drawContours(hand_data["processed_image"], [box], 0, (255, 0, 0), 2)
            # print(np.max(mag), w * h, np.sum(motion_mask > motion_thresh))
        else:
            hand_data["bbox"] = (0, 0, 0, 0)
            data["detect_counter"] = 0
            hand_data["show_flow"] = False

    return



def init_data():
    data = {
        "board_corners" : None,
        "board_buffer" : 0,
        "detect_counter" : 0,
        "player_hand" : ""
    }
    dice_data = {
        "coordinates" : [(), ()],
        "prev_frame_count" : None,
        "curr_frame_count" : None,

    }
    hand_data = {
        "cover_dice": False,
        "show_flow": False,
        "bbox" : (0, 0, 0, 0),
        "half_board" : 0,
        "took_dice_txt" : "",
        "last_hand_bbox" : (0, 0, 0, 0)
    }
    return data, dice_data, hand_data

def main():
    # data_dir = "/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/webcam_frames_5/"
    # blank_board = cv2.imread(os.path.join(data_dir, "blank_board.jpg"), cv2.IMREAD_GRAYSCALE)
    # grid = create_grid(blank_board) # Change to dynamic creation
    blank_board = None
    data, dice_data, hand_data = init_data()
    # bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=25, detectShadows=True)

    # Start Processing
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect game board
        aligned_board, edges_img = detect_board(frame, data)

        if aligned_board is None:
            if data["board_corners"] is not None:
                frame, aligned_board = find_board_by_corners(frame, data)
            else:
                aligned_board, edges_img = detect_board(frame, data)
        if aligned_board is not None: # Successfully detected board
            if blank_board is None:
                blank_board = aligned_board
                continue

            detect_hand(blank_board, aligned_board, data, hand_data)
            hand_data["half_board"] = aligned_board.shape[0]
            blank_board = aligned_board

            # # fg_mask = bg_subtractor.apply(aligned_board)
            # # contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # # for contour in contours:
            # #     if cv2.contourArea(contour) > 1000:  # Ignore small noise
            # #         x, y, w, h = cv2.boundingRect(contour)
            # #         cv2.rectangle(aligned_board, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # white_mask = white_threshold(aligned_board)
            # black_mask = black_threshold(aligned_board)
            # aligned_board, white_checkers, black_checkers = segment_checkers(aligned_board, white_mask, black_mask)
            # dice_images, dice = detect_dice(aligned_board, white_mask, black_mask, white_checkers, dice_data)
            # start_flow(dice_data, hand_data)
            cv2.imshow("Aligned Backgammon Board", show_game(hand_data["processed_image"], aligned_board))
            # cv2.imshow("Aligned Backgammon Board", concat_images(hand["processed_image"], hand["motion_mask"]))
            # if blank_board is None:
            #     blank_board = aligned_board
            # triangles = detect_triangles(aligned_board)
            # if len(triangles) == 24:
            #     print("Worked")
            # grid = create_grid(aligned_board)
            # Create grid at the start of the game
            # if grid is not None:
            #     cv2.imshow("New", grid)
            # cv2.imshow("Aligned Backgammon Board", aligned_board)
        cv2.imshow("Backgammon Board Detection", concat_images(frame, edges_img))

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()