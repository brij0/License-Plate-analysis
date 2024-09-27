import os
import xml.etree.ElementTree as ET

#---------------------------------------------------------
# Setup file paths for XML annotations and YOLO labels
#---------------------------------------------------------
xml_folder = "D:\\University\\All Projects\\CCTV_analysis_system\\Dataset\\annotations"
yolo_folder = "D:\\University\\All Projects\\CCTV_analysis_system\\Dataset\\Yolo_folder"
images_folder = "D:\\University\\All Projects\\CCTV_analysis_system\\Dataset\\images"

# Create the YOLO folder if it doesn't exist
if not os.path.exists(yolo_folder):
    os.makedirs(yolo_folder)

#---------------------------------------------------------
# Function to convert XML (VOC) bounding boxes to YOLO format
#---------------------------------------------------------
def convert_to_yolo(size, box):
    dw = 1.0 / size[0]  # Calculate width scaling factor
    dh = 1.0 / size[1]  # Calculate height scaling factor
    x_center = (box[0] + box[1]) / 2.0  # Get x center of bounding box
    y_center = (box[2] + box[3]) / 2.0  # Get y center of bounding box
    width = box[1] - box[0]  # Width of bounding box
    height = box[3] - box[2]  # Height of bounding box
    x_center *= dw  # Scale x center
    y_center *= dh  # Scale y center
    width *= dw  # Scale width
    height *= dh  # Scale height
    return (x_center, y_center, width, height)

#---------------------------------------------------------
# Process each XML file, convert, and save as YOLO format
#---------------------------------------------------------
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        tree = ET.parse(os.path.join(xml_folder, xml_file))  # Parse the XML file
        root = tree.getroot()

        # Get image dimensions (width, height)
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # Generate YOLO label file path
        yolo_file_name = os.path.splitext(xml_file)[0] + ".txt"
        yolo_file_path = os.path.join(yolo_folder, yolo_file_name)

        #---------------------------------------------------------
        # Write object annotations to YOLO format
        #---------------------------------------------------------
        with open(yolo_file_path, "w") as yolo_file:
            for obj in root.iter("object"):
                class_name = obj.find("name").text  # Get class label
                if class_name == "licence":
                    class_id = 0  # Assign class ID for "licence"
                else:
                    continue  # Skip if it's a different class

                # Extract bounding box coordinates
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # Convert bounding box to YOLO format
                bbox = convert_to_yolo((width, height), (xmin, xmax, ymin, ymax))

                # Write the class ID and bounding box to the YOLO label file
                yolo_file.write(f"{class_id} {' '.join(map(str, bbox))}\n")

print("Conversion completed!")
