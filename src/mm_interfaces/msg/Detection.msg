string label
float32 score

# Pixel coordinates in the request image: x_min, y_min, x_max, y_max
int32[4] bbox

# mono8 PNG, same size as the request image (255 = object)
sensor_msgs/CompressedImage mask
