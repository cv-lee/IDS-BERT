# AI Based Intrusion Detection System (IDS)

- Implementation of an **NLP-based Intrusion Detection System (IDS)** for binary classification of detected attack packets. 
- This task won **1st place (과기정통부 장관상)** in the **Cybersecurity AI Big Data Challenge (November 2022)**, hosted by the **Ministry of Science and ICT**.
<br/>


## 📋 Task
The primary task is to classify intrusion detection system (IDS) results into **attack packet** or **non-attack packet**, using a binary classification approach.
<br/>

## 🤖 Model
- **Base Model**: RoBERTa
  - Fine-tuned on IDS-related binary classification data.
  - Leverages pre-trained language model capabilities for analyzing attack packet data.
<br/>

## 📊 Dataset
- **Intrusion Detection System Dataset**
  - Contains labeled samples for binary classification.
  - Size: **N million samples**.
  - Includes features extracted from network traffic packets:
  ```bash
  'PAYLOAD', 'APP_PROTO', 'SRC_PORT', 'DST_PORT', 'IMPACT', 'RISK', 'JUDGEMENT', 'Method', 'Method-URL', 'HTTP', 'Host', 'User-Agent', 'Accept', 'Accept-Encoding', 'Accept-Language', 'Accept-Charset', 'Content-Type', 'Content-Length', 'Connection', 'Cookie', 'Upgrade-Insecure-Requests', 'Pragma', 'Cache-Control', 'Body'
  ```
<br/>

## 📂 Repository Structure

```bash
IDS-BERT/
├── ckpt/                 
│   ├── pretrained/              
│   └── trained/                 
├── dataset/                    
├── data_preprocess.py        
├── train.py    
├── inference.py          
├── utils.py                 
├── main.ipynb                       
└── README.md                   
```
<br/>

## 📚 Requirements
- **Python**: 3.9+
- **CUDA**: 11.7+ (for GPU-based training and inference)
- For a complete list of dependencies, see `requirements.txt`.
<br/>

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/cv-lee/IDS-BERT.git
```
### 2. Prepare the Dataset & Pretrained Model
- Place your dataset files in the `dataset/` folder.
- Place your pretrained files in the `ckpt/pretrained` folder.
- Open the `main.ipynb` file.
- Execute the data preprocessing step:
  ```bash
  python3 data_preprocess.py
  ```

### 3. Train the Model
- Train the RoBERTa model using the preprocessed dataset and pretrained model:
  ```bash
  python3 train.py
  ```

### 4. Run Inference
- Use the trained model for binary classification:
  ```bash
  python3 inference.py
  ```
<br/>

## 📄 Configuration
The `config.json` file contains adjustable parameters for preprocessing, training, and inference. Key fields include:

```json
{
  "model_name": "roberta-base",
  "max_seq_length": 512,
  "batch_size": 16,
  "learning_rate": 1e-5,
  "weight_decay": 0.01,
  "num_epochs": 10,
  "device": "cuda"
}
```
<br/>

## 📬 Contact
For any questions or issues, please contact:
- **[Joohyun Lee](mailto:dlee110600@gmail.com)**
