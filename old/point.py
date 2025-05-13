import cv2  
  
points = []  
  
def click_event(event, x, y, flags, params):  
    global points  
    if event == cv2.EVENT_LBUTTONDOWN:  
        points.append((x, y))  
        print(f"Point marked: ({y}, {x})")  
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  
        cv2.imshow("Image", image)  
  
if __name__ == "__main__":  
    image_path = "9.png"  
    image = cv2.imread(image_path)  
    cv2.namedWindow("Image")  
    cv2.setMouseCallback("Image", click_event)  
  
    while True:  
        cv2.imshow("Image", image)  
        key = cv2.waitKey(1) & 0xFF  
        if key == ord("q"):  
            break  
  
    cv2.destroyAllWindows()  

