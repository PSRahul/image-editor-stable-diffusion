import numpy as  np
import matplotlib.pyplot as plt
import cv2

def main():

    bounding_box=list([94, 118, 213, 285])
    X = bounding_box[0]
    Y= bounding_box[1]
    w=bounding_box[2]-bounding_box[0]
    h=bounding_box[3]-bounding_box[1]

    image_A = cv2.imread('task_2_inter_output_images/chair.jpg')
    image_A_size = (320,320)

    image_A = cv2.resize(image_A, image_A_size, interpolation= cv2.INTER_LINEAR)

    image_B = cv2.imread('cropped_object/chair_novel_view.png')  
    image_B_size = (w,h)

    image_B = cv2.resize(image_B, image_B_size, interpolation= cv2.INTER_LINEAR)

    # Get a mask for the background white regions
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(image_B, lower_white, upper_white)

    # Invert the white background mask and remove the background
    mask_inv = cv2.bitwise_not(mask_white)
    roi = cv2.bitwise_and(image_B, image_B, mask=mask_inv)

    # Overlap the two images in the bounding box region
    roi_A = cv2.bitwise_and(image_A[Y:Y+h, X:X+w], image_A[Y:Y+h, X:X+w], mask=mask_white)
    result = cv2.add(roi_A, roi)

    # Assign the novel bounding box region back to A
    image_A[Y:Y+h, X:X+w] = result

    cv2.imwrite("final_output_chair.png",image_A)



if __name__ == "__main__":

    main()