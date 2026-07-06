import cv2 as cv

def main():
    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        print('Unable to turn on camera!')
        return
    
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  #width of the frame
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)  #height of the frame
    
    while True:
        success, frame = camera.read()

        if not success:
            print('Camera is not able to Process frames')
            break

        frame = cv.flip(frame, 1)
        display = frame.copy()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imshow('Video', display)

        if cv.waitKey(1) & 0xFF == ord('q'):
            print('Camera Closed Successfully!')
            break

    camera.release()
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()