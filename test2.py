import cv2
img = cv2.imread("images/test-image.jpg")
img2 = img.copy()
crop_img = img[0:1080, 0:960] #ymin:ymax and xmin:xmax
cv2.imshow("cropped", crop_img)
cv2.imshow("Orginal image", img2)
cv2.imwrite('images/cropped-test-image.jpg', crop_img)
print("Image saved")
cv2.waitKey(0)
cv2.destroyAllWindows()