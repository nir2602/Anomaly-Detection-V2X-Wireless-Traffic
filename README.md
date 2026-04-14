# Machne Learning Based Anomaly Detection for V2X Wireless Traffic

This is a study of the performance of various machine learning models on detecting anomalous behaviour in a Internet of Vehicles context. Training was done on the CICIov2024 dataset. More information can be found below.

This study was conducted to stasify the requirements of CSCI4179/CSCI6711 - Intelligent Wireless Networks final project.  

# Installation and Usage

### Installation
```bash
# Clone Repository 
git clone ---

# Create a virtual environment
python3 -m venv project

# Activate the virtual environment
source project/bin/activate

# install requirements
pip install -r requirements.txt
```

### Usage
```bash
python3 main.py
```
### Project Layout

The project directory is structured as follows:

```
CSCI4179-Project/
├── dataset/
├── utils/                  # utility scripts
│   ├── process_dataset.py  # load and process dataset
├── models/                 # stored models after training
├── algorithms/
│   ├── random_forest.py    # Random Forest implementation
│   ├── decision_tree.py    # Decision Tree implementation
│   ├── svm.py              # SVM implementation
├── main.py                 # Entry point for the project
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```


## Implenetation of Algorithms
All implementations of algorithms can be found in `algorithms/` directory.

Note: SVM training may take between 4-5 hours, depending on user hardware.


## Dataset
Our dataset is the CIC Internet of Vehicles 2024 dataset, provided by the Canadian Institute fo Cybersecurity. 
The dataset recorded CAN Bus traffic of a 2019 Ford Car vehicle. The authors played traffic containing benign payloads, Denial of Service (DoS) attacks on the CAN BUS, as well as Spoofing Attacks intended to change behavior of the vehicle, including spoofed RPM, Speed, Steering Wheel, and Gas Pedel. 

The paper describing the original dataset, and a link to retrieve the dataset can be found below.

E. C. P. Neto, H. Taslimasa, S. Dadkhah, S. Iqbal, P. Xiong, T. Rahmanb, and A. A. Ghorbani, ["CICIoV2024: Advancing Realistic IDS Approaches against DoS and Spoofing Attack in IoV CAN bus,"](https://www.sciencedirect.com/science/article/pii/S2542660524001501) Internet of Things, 101209, 2024.

https://www.unb.ca/cic/datasets/iov-dataset-2024.html




<!-- ---
## References
1. https://scikit-learn.org/stable/modules/ensemble.html#forest
2. 
3. -->

