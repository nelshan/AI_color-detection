import cv2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Reading the image with opencv
img = cv2.imread('colorpic.jpg')

# Reading csv file with pandas and giving names to each column
index=["color","color_name","hex","R","G","B"]
colors_df = pd.read_csv('colors.csv', names=index, header=None)

# Define feature and label arrays
X = colors_df[['R', 'G', 'B']]
y = colors_df['color_name']

# Create and fit a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

def on_mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        b, g, r = img[y, x]
        b, g, r = int(b), int(g), int(r)
        color_name = knn.predict([[r, g, b]])
        text = color_name[0] + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)

        # cv2.rectangle(image, startpoint, endpoint, color, thickness) -1 fills entire rectangle 
        cv2.rectangle(img, (20, 20), (710, 60), (b, g, r), -1)
        
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

# Create a window
cv2.namedWindow('image')
cv2.setMouseCallback('image', on_mouse_event)

while True:
    cv2.imshow("image", img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
