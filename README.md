# License Plate Recognition System for Parking Zone Monitoring

[![License](https://img.shields.io/static/v1?label=License&message=MIT&color=blue&style=plastic&logo=appveyor)](https://opensource.org/licenses/MIT)

## Table Of Contents

- [Description](#description)
- [Deployed Website Link](#deployed-website-link)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Tests](#tests)
- [GitHub](#github)
- [Contact](#contact)
- [License](#license)

![GitHub repo size](https://img.shields.io/github/repo-size/brij0/License-Plate-analysis?style=plastic)
![GitHub top language](https://img.shields.io/github/languages/top/brij0/License-Plate-Recognition-System-for-parking-zone-monitoring?style=plastic)

## Description

This project is a real-time license plate recognition system designed for parking zone monitoring and management. It uses the YOLOv8 object detection models to detect vehicles and recognize license plates from CCTV footage or live camera feeds. Once detected, the license plate is checked against an authorized vehicle list from a database. If the vehicle is not authorized, an email notification is automatically sent to the admin with the license plate details. The data is also saved in a CSV file for reporting and further analysis. This system helps enforce tiered parking allotments and ensures that vehicles park in their allocated zones.

Currently, the system processes video files from a local folder, but in theory, it could be integrated with any system that provides a video output, such as live surveillance feeds, traffic cameras, or even dashboard cameras.


### Key Features:
- **Real-time license plate recognition** using YOLOv8.
- **Automatic detection of unauthorized vehicles** with email notifications.
- **Parking zone monitoring** for vehicles across multiple tiers.
- **CSV logging** of all detected vehicles for future reference.

## Deployed Website Link

Currently, the project is not deployed online. For any further details, please contact the project maintainer.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/brij0/License-Plate-Recognition-System-for-parking-zone-monitoring.git
    ```
    
2. **Navigate to the project directory**:
    ```bash
    cd License-Plate-Recognition-System-for-parking-zone-monitoring
    ```
    
3. **Set up a virtual environment** (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows: env\Scripts\activate
    ```

4. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Download the YOLO pre-trained models** and save them in the `/YOLO-PretrainedModels` folder.

6. **Run the system**:
    ```bash
    python plate_recognition.py
    ```

## Usage

1. Once the system is running, it will process live footage or CCTV videos and detect vehicles and license plates.
2. The license plates are checked against the `allowed_cars.csv` file.
3. If a vehicle is detected that is not in the allowed list, an email will be sent to the admin with an image of the license plate.
4. The system will log all detected vehicles and their details (including timestamp) into an output CSV file.
   
### Example Workflow:

- **Admin uploads the list of allowed vehicles** (e.g., license plates) into `allowed_cars.csv`.
- **CCTV feed is processed**: each vehicle's license plate is recognized and compared to the list.
- **Email notification**: If an unauthorized vehicle is detected, an email is automatically sent to the admin.

## Contributing

Contributions to this project are welcome! Follow these steps:

1. Fork the repository.
2. Create a new feature branch: `git checkout -b feature-branch`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-branch`.
5. Submit a pull request.

Please ensure that your code follows best practices and is well documented.

## Tests

To run the tests, you can use the following command:

```bash
pytest
