import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

"""Image Processing -------------------------------------------------------------------------------------------------"""
def extract_patch(img, center, size=20):
    x, y = int(center[0]), int(center[1])
    half = size // 2
    return img[max(0, y - half) : min(y + half, img.shape[0]), max(0, x - half) : min(x + half, img.shape[1])]

def compare_patches(prev_img, curr_img, prev_corners, patch_size=20, threshold=0.5):
    match_corners = 0
    scores = []
    if prev_img is None or curr_img is None:
        return False

    for i in range(4):
        patch_prev = extract_patch(prev_img, prev_corners[i], patch_size)
        patch_curr = extract_patch(curr_img, prev_corners[i], patch_size)
        # print(f"Patch Prev Shape: {patch_prev.shape}, Patch Curr Shape: {patch_curr.shape}")

        score = ssim(patch_prev, patch_curr, channel_axis=-1)
        scores.append(score)
        if score > threshold:
            match_corners += 1

    if match_corners > 1:
        return True
    # print(scores)
    return False

def safe_polyfit(x_values, y_values):
    if np.isnan(x_values).any() or np.isnan(y_values).any():
        # print("Warning: NaN values detected in polyfit inputs!")
        return None

    if abs(x_values[0] - x_values[1]) < 1e-6:  # Detect near-vertical lines
        # print(f"Warning: Near-vertical line detected at x = {x_values[0]}")
        return None  # Undefined slope

    return np.polyfit(x_values, y_values, 1)[0]  # Return slope m

def detect_board(last_frame, frame, game):
    # Step 1: Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 3)  # Kernel size of 3
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)  # Expand edges slightly changed from 2 to 1
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board_corners = None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in sorted_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * frame.size:
            board_corners = approx.reshape(4, 2)
            break

    # Debugging: Draw the detected polygon
    if board_corners is None:
        if game["board_corners"] is not None:
            if compare_patches(last_frame, frame, game["board_corners"]):
                board_corners = game["board_corners"].astype(np.int32)
            else:
                return None, closed
        else:
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

    # Compute the width and height of the new image
    (tl, tr, br, bl) = ordered_corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)


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
    game["board_corners"] = ordered_corners
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
    dice_xy = []
    dice_images = []
    for contour in contours:
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
                    dice_images.append(dice_roi)
                    dice_xy.append(box[0])
                    cv2.drawContours(output, [contour], -1, (255, 255, 0), 3)
                else:
                    cut_dice = [dice_roi[:, :w // 2], dice_roi[:, w // 2:]] if w>h else [dice_roi[:h //2,:], dice_roi[h//2:,:]]
                    for dice in cut_dice:
                        dice_images.append(dice)
                        dice_xy.append(box[0])

    """ Detect Number """
    dice_numbers = []
    # print(len(dice_images))
    for i, dice in enumerate(dice_images):
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

    return dice_images, dice_numbers

def easy_detect_dice(img, white_mask, black_mask, white_checkers):
    """ Detect and transform dice """
    white_checkers_img = np.zeros_like(img)
    for (x, y) in white_checkers:
        cv2.circle(white_checkers_img, (x, y), int(img.shape[0] * 0.034), (255,255,255), cv2.FILLED)  # Green circles
    white_checkers = cv2.cvtColor(white_checkers_img, cv2.COLOR_RGB2GRAY)  # 3 to 1 channel
    difference = white_mask - white_checkers
    difference = cv2.medianBlur(difference, 5)
    difference = cv2.dilate(difference, np.ones((3, 3), np.uint8), iterations=1)
    contours = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # print([cv2.contourArea(c) for c in sorted_contours][:3])

    output = img.copy()
    dice_xy = []
    dice_images, dice_images_uncut = [], []
    for contour in contours:

        if 500 < cv2.contourArea(contour) < 2000:
            cv2.drawContours(img, [contour], -1, (255, 0, 0), 3)
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
                    dice_images_uncut.append(dice_roi)
                    dice_xy.append(box[0])
                    cv2.drawContours(output, [contour], -1, (255, 255, 0), 3)
                else:
                    cut_dice = [dice_roi[:, :w // 2], dice_roi[:, w // 2:]] if w>h else [dice_roi[:h //2,:], dice_roi[h//2:,:]]
                    for dice in cut_dice:
                        size = int((0.05 * (dice.shape[0] + dice.shape[1]) / 2))
                        dice_images.append(dice[size:-size,size:-size])
                        dice_images_uncut.append(dice)
                        dice_xy.append(box[0])

    """ Detect Number """
    dice_numbers, dice_numbers_uncut = [], []
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
            dots_uncut = template_match(dice_images_uncut[i], dice_images_uncut[i], threshold=0.5, min_samples=1, contour_color=(255, 255, 255),
                                  checker_radius=2, draw=False)

            if len(dots) > 0:
                dice_num = int(np.clip(len(dots), 1, 6))
                # dice_num = len(dots)
                dice_num_uncut = int(np.clip(len(dots_uncut), 1, 6))
                dice_numbers.append(dice_num)
                dice_numbers_uncut.append(dice_num_uncut)
            # print(f"Correct: {len(dots)}, Predicted: {pred}")
        except:
            continue
        continue

    return dice_images, dice_numbers, dice_images_uncut, dice_numbers_uncut

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
        "last_valid_board" : None,
        "board_corners" : None
    }
    data = {
        "board_corners": None,
        "board_buffer": 0,
        "detect_counter": 0,
        "player_hand": ""
    }
    dice_data = {
        "coordinates": [(), ()],
        "prev_frame_count": None,
        "curr_frame_count": None,

    }
    hand_data = {
        "cover_dice": False,
        "show_flow": False,
        "bbox": (0, 0, 0, 0),
        "half_board": 0,
        "took_dice_txt": "",
        "last_hand_bbox": (0, 0, 0, 0),
        "last_center": (0, 0),
        "dist_lst": []
    }
    return game, data, dice_data, hand_data

"""Main -------------------------------------------------------------------------------------------------------------"""
def main():
    os.chdir("/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/")
    data_dirs = ["filtered_frames/", "filtered_frames_3/", "filtered_frames_4/", "filtered_frames_6/", "filtered_frames_7/"]

    success = 0
    count = [0] * 6
    total_frames = 0
    prev_dice = [0, 0]

    for data_dir in data_dirs[4:]:
    # data_dir = "filtered_frames"
    # data_dir = "webcam_frames_7/"
    # output_dir = "filtered_frames_7/"
    # os.makedirs(output_dir, exist_ok=True)
        game, data, dice_data, hand_data = init_game()
        last_frame = None

        frame_names = sorted([filename[:-4] for filename in os.listdir(data_dir)])
        frame_paths = [os.path.join(data_dir, filename + ".jpg") for filename in frame_names]
        total_frames += len(frame_names)

        # Start Processing
        for i, frame_path in enumerate(tqdm(frame_paths[:], desc=f"Processing items")):
            frame = cv2.imread(frame_path)
            # if last_frame is None:
            #     last_frame = frame
            #     continue
            # # Detect game board
            # aligned_board, _ = detect_board(last_frame, frame, game)
            # if aligned_board is not None: # Successfully detected board
            #     cv2.imwrite(os.path.join(output_dir, frame_names[i] + ".jpg"), aligned_board)

            # Detect checkers
            try:
                white_mask = white_threshold(frame)
                black_mask = black_threshold(frame)
                frame, white_checkers, black_checkers = segment_checkers(frame, white_mask, black_mask)
            except:
                continue
            # Detect dice
            dice_images, dice_numbers, dice_images_uncut, dice_numbers_uncut = easy_detect_dice(frame, white_mask, black_mask, white_checkers)
            # if sorted(game["curr_dice"]) != sorted(game["turn_dice"]) and len(game["curr_dice"]) == 2:
            #     if game["dice_buffer"] < 3:
            #         game["dice_buffer"] += 1
            #     else:
            #         game["dice_buffer"] = 0
            #         game["turn_dice"] = game["curr_dice"]
            #         print(f"Turn dice: {game["turn_dice"]}")
            # print(len(dice_images))
            if len(dice_numbers) == 2 and set(dice_numbers) != set(prev_dice):
                prev_dice = dice_numbers
                success += 1
                for k in range(len(dice_numbers)):
                    plt.figure(figsize=(4,2))
                    plt.subplot(2,2,1), plt.imshow(dice_images[k]), plt.title(f"{dice_numbers[k]}"), plt.axis('off')
                    plt.subplot(2,2,2), plt.imshow(dice_images_uncut[k]), plt.title(f"{dice_numbers_uncut[k]}"), plt.axis('off')
                    plt.tight_layout(), plt.show()
                    count[dice_numbers[k] - 1] += 1


    print(f"Success: {success}/{total_frames}")
    print(count)


if __name__ == "__main__":
    plt.close('all')
    main()