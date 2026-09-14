import numpy as np
import cv2
from sklearn.cluster import DBSCAN

def concat_images(ref_img, resized_img):
    # Ensure both images are 3 dimensional
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB) if len(ref_img.shape) == 2 else ref_img
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB) if len(resized_img.shape) == 2 else resized_img
    # Ensure both images have the same height
    h1, w1 = ref_img.shape[:2]
    h2, w2 = resized_img.shape[:2]

    # Resize edges_img to match frame's height
    scale_factor = h1 / h2  # Compute scaling ratio
    new_width = int(w2 * scale_factor)  # Adjust width proportionally
    resized_img = cv2.resize(resized_img, (new_width, h1))

    # Concatenate images side by side
    combined = cv2.hconcat([ref_img, resized_img])
    return combined

def detect_board(frame, ref_img):
    """Detect board"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(edges, kernel, iterations=3)  # Expand edges slightly
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board_corners = None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in sorted_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > ref_img.size:
            board_corners = approx
            break

    cv2.drawContours(frame, [board_corners], -1, (0, 255, 0), 3)

    if board_corners is None:
        return None, closed

    """Detect exact board corners"""
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

    # Harris corner detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray32 = np.float32(gray)
    dst = cv2.cornerHarris(gray32, 10, 3, 0.04)
    dst = cv2.dilate(dst, None)
    threshold = 0.001 * dst.max()
    # frame[dst > threshold] = [0, 0, 255]  # Dice pips
    corner_candidates = np.argwhere(dst > threshold)
    corner_candidates = np.array(corner_candidates)

    exact_board_corners = np.zeros((4,2))
    for i, corner in enumerate(board_corners):
        swapped_corner = corner[::-1]
        distances = np.linalg.norm(corner_candidates - swapped_corner, axis=1)  # Compute Euclidean distance
        closest_point = corner_candidates[np.argmin(distances)]  # Get the closest detected corner
        exact_board_corners[i] = (closest_point[::-1])
        # cv2.circle(frame, closest_point[::-1], 5, (255, 255, 0), -1)  # Draw detected corner (green)

    """Warp board"""
    ordered_corners = order_points(exact_board_corners)
    (tl, tr, br, bl) = ordered_corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)

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

    if warped is None:
        print(" Didn't warp")
        return None, closed

    return warped, closed

def segment_checkers(img, white_thresh, black_thresh):
    def template_match(input_img, output_img, threshold=0.5, min_samples=1, contour_color=(0, 0, 255),
                       checker_radius=22):
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
        for (x, y) in final_checker_coordinates:
            cv2.circle(output_img, (x, y), checker_radius, contour_color, cv2.FILLED)  # Green circles

        return final_checker_coordinates

    # Preprocessing
    output = img.copy()
    white_checkers = template_match(white_thresh, output, threshold=0.5, min_samples=1,
                                    contour_color=(0, 0, 255), checker_radius=int(img.shape[0] * 0.034))
    black_checkers = template_match(black_thresh, output, threshold=0.5, min_samples=1,
                                    contour_color=(0, 255, 0), checker_radius=int(img.shape[0] * 0.034))

    return output, white_checkers, black_checkers

def checkers_thresh(image, k=5):
    # Convert image to LAB color space for better clustering
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l = cv2.split(lab_image)[0]

    # Reshape image into a list of pixels
    pixel_values = lab_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers back to 8-bit values
    centers = np.uint8(centers)

    # Separate masks for black checkers, white checkers, and background
    masks = []
    for i in range(k):
        mask = (labels.flatten() == i).astype(np.uint8) * 255
        mask = mask.reshape(image.shape[:2])
        masks.append(mask)

    checkers_masks = []
    for mask in masks:
        val_amount = np.mean((mask == 255))
        if val_amount < 0.1:
            checkers_masks.append((mask, np.mean(l[mask == 255])))

    if checkers_masks[0][1] > checkers_masks[1][1]:
        white_thresh = checkers_masks[0][0]
        black_thresh = checkers_masks[1][0]
    else:
        white_thresh = checkers_masks[1][0]
        black_thresh = checkers_masks[0][0]

    return white_thresh, black_thresh

def detect_dice(img, white_mask, black_mask, white_checkers):
    """ Detect and transform dice """
    white_checkers_img = np.zeros_like(img)
    for (x, y) in white_checkers:
        cv2.circle(white_checkers_img, (x, y), int(img.shape[0] * 0.034), (255), cv2.FILLED)  # Green circles
    white_checkers = cv2.cvtColor(white_checkers_img, cv2.COLOR_RGB2GRAY)  # 3 to 1 channel
    difference = white_mask - white_checkers
    difference = cv2.medianBlur(difference, 5)
    difference = cv2.dilate(difference, np.ones((3, 3), np.uint8), iterations=1)

    contours = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    output = img.copy()
    filtered_contours = [c for c in contours if 500 < cv2.contourArea(c) < 2000]


    dice_images = []
    for contour in contours:
        # if len(dice_images) == 2:
        #   break
        if 750 < cv2.contourArea(contour) < 3000:
            # print(1)
            rect = cv2.minAreaRect(contour)
            h, w = rect[1]
            if h == 0 or w == 0:
                continue
            aspect_ratio = h / float(w) if w > h else w / float(h)
            # print(aspect_ratio, cv2.contourArea(contour))
            if (750 < cv2.contourArea(contour) < 1500 and 0.8 < aspect_ratio < 1.2) or (
                    1500 < cv2.contourArea(contour) < 3000 and 0.3 < aspect_ratio < 0.7):
                cv2.drawContours(img, [contour], -1, (255, 255, 0), 3)
                # print(aspect_ratio, cv2.contourArea(contour))
                box = np.intp(cv2.boxPoints(rect))

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

                if cv2.contourArea(contour) < 1500:
                    dice_images.append(dice_roi)
                    cv2.drawContours(output, [contour], -1, (255, 255, 0), 3)
                else:
                    dice_images.append(dice_roi[:, :w // 2])  # Left dice
                    dice_images.append(dice_roi[:, w // 2:])  # right side
                #   print(w, h)

    # plt.figure(figsize=(15,5))
    # plt.subplot(1,3,1), plt.imshow(output)
    # plt.subplot(1,3,2), plt.imshow(white_mask, cmap='gray')
    # plt.subplot(1,3,3), plt.imshow(difference, cmap='gray')
    # plt.tight_layout(), plt.show()
    # print(len(dice_images))

    """ Detect Number """
    dice_numbers = []
    # for dice in dice_images:
    #     dice = cv2.threshold(dice, 1, 255, cv2.THRESH_BINARY)[1]
    #     try:
    #         if dice.shape[0] == 0 or dice.shape[1] == 0 or dice.shape[0] > 35 or dice.shape[1] > 35:
    #             continue
    #         dots = template_match(dice, dice, threshold=0.5, min_samples=1, contour_color=(255, 255, 255),
    #                               checker_radius=2)
    #         if len(dots) > 0:
    #             dice_numbers.append(len(dots))
    #     except:
    #         continue
    #     continue

    # print(len(dice_images))
    if len(dice_images) > 1:
        cv2.imshow("Dice", concat_images(dice_images[0], dice_images[1]))
    # return dice_numbers

# def create_grid(frame):

def main():
    blank_board = cv2.imread(
        "/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/webcam_frames_5/blank_board_ref.jpg",
        cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        aligned_board, edges_img = detect_board(frame, blank_board)
        edges_img = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2RGB)

        if aligned_board is not None:
            white_th, black_th = checkers_thresh(aligned_board, k=5)
            aligned_board, white_checkers, black_checkers = segment_checkers(aligned_board, white_th, black_th)
            detect_dice(aligned_board, white_th, black_th, white_checkers)
            cv2.imshow("Aligned Backgammon Board", aligned_board)


        cv2.imshow("Backgammon Board Detection", cv2.hconcat([frame, edges_img]))

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()