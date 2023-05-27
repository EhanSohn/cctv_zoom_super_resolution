import cv2 as cv
import numpy as np

# global variables

roi_selected = False
x_start, y_start, w, h = -1,-1,-1,-1
zoom_level = 1
paused = False

# load video
video = cv.VideoCapture('./pedestrians.webm')
frame_total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
frame_shift = 15

def super_resolution(img):
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    path = "./models/FSRCNN_x4.pb"
    sr.readModel(path)
    sr.setModel("fsrcnn", 4)
    img_resolution = sr.upsample(img)

    return img_resolution


# mouse callback function for selecting ROI
def select_roi(event, x, y, flags, param):
    global roi_selected, x_start, y_start, w, h, frame, roi
    if event == cv.EVENT_LBUTTONDOWN:
        roi_selected = False
        x_start, y_start = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if not roi_selected:
            frame_copy = frame.copy()
            cv.rectangle(frame_copy, (x_start, y_start), (x, y), (255, 255, 255), 1)
            cv.imshow('CCTV VIDEO', frame_copy)
    elif event == cv.EVENT_LBUTTONUP:
        roi_selected = True
        w, h = x - x_start, y - y_start
        roi = frame[y_start:y_start+h, x_start:x_start+w]
        cv.imshow('DRAGGED IMAGE', roi)


cv.namedWindow('CCTV VIDEO')
cv.setMouseCallback('CCTV VIDEO', select_roi)

while True:
    frame_video = int(video.get(cv.CAP_PROP_POS_FRAMES))
    if not paused:
        ret, frame = video.read()
        if not ret:
            break

    key = cv.waitKey(1)

    if key == ord(' '):  # Space bar to pause/unpause
        paused = not paused
        cv.putText(frame, 'Zoom in: d   Zoom out: a', (10, 75), 
                            cv.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 2)
        cv.putText(frame, 'Zoom in: d   Zoom out: a', (10, 75), 
                            cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        cv.imshow('CCTV VIDEO', frame)

    elif key == 27:  # 'q' to quit
        break

    elif key == ord('q'):
        cv.destroyWindow('DRAGGED IMAGE')
        roi_selected = False

    elif key == ord(']') or key == ord('}'):
        video.set(cv.CAP_PROP_POS_FRAMES, frame_video + frame_shift)

    elif key == ord('[') or key == ord('{'):
        video.set(cv.CAP_PROP_POS_FRAMES, max(frame_video - frame_shift, 0))

    elif roi_selected:
        if key == ord('a'):  # press 'a' to zoom out
            print("left arrow pressed")
            zoom_level -= 1
            if zoom_level < 1:
                zoom_level = 1
        elif key == ord('d'):  # press 'd' to zoom in
            print("right arrow pressed")
            zoom_level += 1
        zoomed_roi = cv.resize(roi, (roi.shape[1]*zoom_level, roi.shape[0]*zoom_level))
        upscaled_roi = super_resolution(zoomed_roi)
        cv.imshow('DRAGGED IMAGE', upscaled_roi)
    cv.putText(frame, 'Drag to select image for zoom', (10, 25),
                         cv.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 2)
    cv.putText(frame, 'Drag to select image for zoom', (10, 25),
                         cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    cv.putText(frame, '15sec before: [   15sec after: ]', (10, 50), 
                        cv.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 2)
    cv.putText(frame, '15sec before: [   15sec after: ]', (10, 50), 
                        cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    cv.imshow('CCTV VIDEO', frame)

cv.destroyAllWindows()
video.release()
