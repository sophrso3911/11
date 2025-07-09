import cv2
import numpy as np
import time
import random
from PIL import Image, ImageDraw, ImageFont

# Initialize webcam
cap = cv2.VideoCapture(0)

# ASCII characters from darkest to lightest
ASCII_CHARS = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.']

# Hand gesture detection parameters
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
min_hand_size = (50, 50)

def resize_image(image, new_width=100):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    ratio = height / width
    new_height = int(new_width * ratio)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def grayify(image):
    """Convert image to grayscale"""
    grayscale_image = image.convert("L")
    return grayscale_image

def pixels_to_ascii(image):
    """Convert pixels to ASCII characters"""
    pixels = image.getdata()
    characters = "".join([ASCII_CHARS[pixel // 25] for pixel in pixels])
    return characters

def detect_hands(frame):
    """Detect hands in the frame and return their positions"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_hand_size
    )
    return hands

def create_ascii_art(frame, hands):
    """Generate ASCII art based on frame and hand positions"""
    # Convert OpenCV BGR image to RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Resize and convert to grayscale
    resized_img = resize_image(pil_img)
    gray_img = grayify(resized_img)
    
    # Generate ASCII art
    ascii_str = pixels_to_ascii(gray_img)
    
    # Modify ASCII art based on hand positions
    if len(hands) > 0:
        for (x, y, w, h) in hands:
            # Get the region of interest in ASCII coordinates
            ascii_width, ascii_height = resized_img.size
            roi_x = int(x * ascii_width / frame.shape[1])
            roi_y = int(y * ascii_height / frame.shape[0])
            roi_w = int(w * ascii_width / frame.shape[1])
            roi_h = int(h * ascii_height / frame.shape[0])
            
            # Add dynamic elements based on hand position
            for i in range(roi_y, min(roi_y + roi_h, ascii_height)):
                line_start = i * ascii_width
                line_end = line_start + ascii_width
                line = ascii_str[line_start:line_end]
                
                # Add sparkles around hands
                sparkle_chars = ['*', 'o', '+', '.']
                sparkle_line = list(line)
                for j in range(max(roi_x, 0), min(roi_x + roi_w, ascii_width)):
                    if random.random() < 0.3:
                        sparkle_line[j] = random.choice(sparkle_chars)
                ascii_str = ascii_str[:line_start] + "".join(sparkle_line) + ascii_str[line_end:]
    
    # Format ASCII art into lines
    ascii_lines = [ascii_str[index:index + resized_img.size[0]] for index in range(0, len(ascii_str), resized_img.size[0])]
    return "\n".join(ascii_lines)

def display_ascii_on_frame(frame, ascii_art):
    """Overlay ASCII art on the video frame"""
    # Create a blank image to draw ASCII text
    ascii_img = Image.new('RGB', (frame.shape[1], frame.shape[0]), color='black')
    d = ImageDraw.Draw(ascii_img)
    
    # Try to load a monospace font, fallback to default
    try:
        font = ImageFont.truetype("cour.ttf", 10)  # Courier New
    except:
        font = ImageFont.load_default()
    
    # Draw ASCII text on the image
    y_position = 0
    for line in ascii_art.split('\n'):
        d.text((0, y_position), line, font=font, fill=(255, 255, 255))
        y_position += 12  # Line height
    
    # Convert PIL image back to OpenCV format
    ascii_cv = cv2.cvtColor(np.array(ascii_img), cv2.COLOR_RGB2BGR)
    
    # Blend ASCII image with the original frame
    alpha = 0.6  # Transparency factor
    blended = cv2.addWeighted(frame, 1 - alpha, ascii_cv, alpha, 0)
    return blended

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect hands
    hands = detect_hands(frame)
    
    # Draw rectangles around detected hands
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Create ASCII art
    ascii_art = create_ascii_art(frame, hands)
    
    # Overlay ASCII art on the frame
    output_frame = display_ascii_on_frame(frame, ascii_art)
    
    # Display the result
    cv2.imshow('Hand-Gesture ASCII Art', output_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
