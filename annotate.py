import cv2
import os

def annotate_lines(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to binarize the image
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h < 10:  # Filter for horizontal lines
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color for horizontal lines

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    contours, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 and h > 50:  # Filter for vertical lines
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for vertical lines

    # Save the annotated image
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    input_image_path = r"C:\Users\AKASH M\Desktop\check5.png"
    output_image_path = r"C:\Users\AKASH M\Desktop\check115.png"

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    annotate_lines(input_image_path, output_image_path)
