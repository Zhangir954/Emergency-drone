import cv2

print("Searching for available cameras...")

for index in range(10):
    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

    if camera.isOpened():
        success, frame = camera.read()

        if success:
            height, width = frame.shape[:2]
            print(f"Camera index {index}: working ({width}x{height})")

            cv2.imshow(f"Camera {index}", frame)
            print("Press any key to close this camera preview.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Camera index {index}: opened, but no image received")

        camera.release()

    else:
        print(f"Camera index {index}: unavailable")

cv2.destroyAllWindows()