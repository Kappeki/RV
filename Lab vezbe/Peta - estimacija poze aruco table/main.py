import cv2
import cv2.aruco as aruco
import glob
import numpy as np

# Camera calibration


def camera_calibration_from_images(image_names):
    print("Camera Calibration started. Please wait...")
    aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    aruco_params = aruco.DetectorParameters()
    board = aruco.GridBoard([5, 7], 1.75, 0.5, aruco_dict)
    all_corners = []
    all_ids = []
    counter = []
    img_size = None

    for image_name in image_names:
        calib_image = cv2.imread(image_name)
        (l_corners, l_ids, rejected) = cv2.aruco.detectMarkers(calib_image, aruco_dict, parameters=aruco_params)
        all_corners.extend(l_corners)
        all_ids.extend(l_ids)
        counter.append(len(l_corners))

        img_size = calib_image.shape[0:2]

    all_ids = np.array(all_ids)
    counter = np.array(counter)
    ret_value, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraAruco(
        all_corners,
        all_ids,
        counter,
        board,
        img_size,
        None,
        None
    )
    print("Camera Calibration has finished")
    return ret_value, camera_matrix, dist_coeffs, rvecs, tvecs, aruco_dict, aruco_params, board


img_names = glob.glob("./Aruco/calib_image_*jpg")
# seting camera parametres
ret_value, camera_matrix, dist_coeffs, rvecs, tvecs, aruco_dict, aruco_params, board = camera_calibration_from_images(
    img_names)

cam = cv2.VideoCapture('./Aruco/Aruco_board.mp4')
frame_number = 0
# catching frames and detecting aruco markers
while cam.isOpened():
    ret, frame = cam.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        # bordering
        frame = aruco.drawDetectedMarkers(frame, corners, borderColor=(0, 255, 0))
        if ids is not None and len(ids) > 0:
            flatten_ids = ids.flatten()
            # labeling
            for (markerCorner, markerID) in zip(corners, flatten_ids):
                (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.putText(frame, "id=" + str(markerID), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # estimation pose of aruco board
            pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None)
            if pose:
                frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 5)
            cv2.imshow('Aruco', frame)
            # saving one frame
            if frame_number == 74:
                cv2.imwrite('output.jpg', frame)
            frame_number = frame_number+1
        else:
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
