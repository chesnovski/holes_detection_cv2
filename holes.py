# import required libraries
import cv2
import numpy as np

path_img =r'C:\Users\Dell\lesson_1\holes_detection\images\Kamera2_ok194_13_12_2022_8_15_0.bmp'
#preprocess image 
def preprocess_img(path_img):
    img = cv2.imread(fr'{path_img}')
    noiseless_image_colored = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21) 
# convert the image to grayscale
    gray = cv2.cvtColor(noiseless_image_colored, cv2.COLOR_BGR2GRAY)
# apply thresholding to convert the grayscale image to a binary image
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
    canny = cv2.Canny(thresh, 75, 200)
    return canny, img



# find the contours
def find_contours(canny, img):
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = img.copy()
    contour_list = []
    small_holes_list=[]
    big_holes_list=[]
    middle_holes_list=[]
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        print(area)
        if 1000 < area < 5000:
            contour_list.append(contour)
          
        if 1200 < area < 2000:
            small_holes_list.append(area)
        elif 4000 < area < 6000:
            big_holes_list.append(area) 
        elif 2000< area < 4000:
            middle_holes_list.append(area)
    return image_copy, contour_list, contours, small_holes_list, big_holes_list,  middle_holes_list
  

#print contours
def draw_contours(image_copy, contour_list, contours,small_holes_list, big_holes_list,  middle_holes_list):
    cont=cv2.drawContours(image_copy,contour_list, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")
   
    quantity_small_holes=len(small_holes_list)//2
    quantity_big_holes=len(big_holes_list)//2
    quantity_middle_holes=len(middle_holes_list)//2

    return cont, quantity_small_holes, quantity_big_holes, quantity_middle_holes

# cv2.imshow('Objects Detected', image_copy)


def main():
    canny, img=preprocess_img(path_img)
    image_copy, contour_list, contours, small_holes_list, big_holes_list, middle_holes_list=find_contours(canny, img)
    cont, quantity_small_holes, quantity_big_holes, quantity_middle_holes =draw_contours(image_copy, contour_list, contours, small_holes_list, big_holes_list, middle_holes_list)
    print_val=f"Quantity big holes:{quantity_big_holes}\nQuantity small holes:{quantity_small_holes}\nQuantity middle holes:{quantity_middle_holes}"
    print(print_val)
    # cv2.imshow('Objects Detected', canny)
    # cv2.imshow('Objects Detected', image_copy)
    cv2.imwrite('holes.png', image_copy)
    print("After saving image:") 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    



