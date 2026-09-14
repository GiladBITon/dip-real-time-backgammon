import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def straighten_and_crop_board(img, max_width=0, max_height=0):
    # Step 1: Preprocessing
    # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    # return gray
    # blurred = cv2.medianBlur(gray, 3)  # Kernel size of 3
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    # return edges
    # Step 3: Morphological Closing to connect lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(edges, kernel, iterations=3)  # Expand edges slightly
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
    return closed
    # Step 4: Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), cv2.FILLED)
    # return img
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
        return None
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
    if max_width == 0 and max_height == 0:
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

main_dir = '/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/'
os.chdir(main_dir)
data_dir = "webcam_frames_5"
output_dir = "filtered_frames_4"
os.makedirs(output_dir, exist_ok=True)

# img_paths = [os.path.join(data_dir, img_name) for img_name in os.listdir(data_dir)]
img_names = [img_name for img_name in os.listdir(data_dir)]
img_paths = ["/Users/razbarak/PycharmProjects/PythonProject/DIP_Final_Project/webcam_frames_5/reference.jpg"]
for i, img_path in enumerate(img_paths):
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # aligned_board = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    aligned_board = straighten_and_crop_board(image)
    # lines = cv2.HoughLinesP(aligned_board, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    # if lines is not None:
    #     for line in lines[:, 0]:
    #         x1, y1, x2, y2 = line
    #         cv2.line(aligned_board, (x1, y1), (x2, y2), (255, 0, 0), 1)
    plt.figure()
    plt.title(img_names[i]), plt.imshow(aligned_board)
    plt.tight_layout(), plt.show()
    # if aligned_board is not None:# and 500 < aligned_board.shape[0] < 720:
        # cv2.imwrite(os.path.join(output_dir, img_names[i]), aligned_board)
        # print("saved!")