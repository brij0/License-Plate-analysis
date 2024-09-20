import os
import xml.etree.ElementTree as ET

# Define the path to your XML annotation folder and the path to save the YOLO labels
xml_folder = "D:\\University\\All Projects\\CCTV_analysis_system\\Dataset\\annotations"  # Replace with the path to your XML annotations
yolo_folder = "D:\\University\\All Projects\\CCTV_analysis_system\\Dataset\\Yolo_folder"  # Replace with the path where YOLO txt files will be saved
images_folder = "D:\\University\\All Projects\\CCTV_analysis_system\\Dataset\\images"  # Replace with the path to your images folder

# Make sure the YOLO folder exists, if not, create it
if not os.path.exists(yolo_folder):
    os.makedirs(yolo_folder)

# Function to convert VOC bounding box format (XML) to YOLO format
def convert_to_yolo(size, box):
    dw = 1.0 / size[0]  # width of image
    dh = 1.0 / size[1]  # height of image
    x_center = (box[0] + box[1]) / 2.0  # (xmin + xmax) / 2
    y_center = (box[2] + box[3]) / 2.0  # (ymin + ymax) / 2
    width = box[1] - box[0]  # xmax - xmin
    height = box[3] - box[2]  # ymax - ymin
    x_center *= dw
    y_center *= dh
    width *= dw
    height *= dh
    return (x_center, y_center, width, height)

# Loop over all XML files in the folder
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        # Parse the XML file
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        # Get the size of the image (width and height)
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # Create a corresponding YOLO txt file
        yolo_file_name = os.path.splitext(xml_file)[0] + ".txt"
        yolo_file_path = os.path.join(yolo_folder, yolo_file_name)

        with open(yolo_file_path, "w") as yolo_file:
            # Loop over each object in the XML
            for obj in root.iter("object"):
                # Get the class name
                class_name = obj.find("name").text
                # Map your class names to numbers (assuming "licence" is class 0)
                if class_name == "licence":
                    class_id = 0  # You can modify this to handle multiple classes
                else:
                    continue  # Skip other classes if needed

                # Get the bounding box coordinates
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # Convert the bounding box to YOLO format
                bbox = convert_to_yolo((width, height), (xmin, xmax, ymin, ymax))

                # Write the converted annotation to the YOLO file
                yolo_file.write(f"{class_id} {' '.join(map(str, bbox))}\n")

print("Conversion completed!")
