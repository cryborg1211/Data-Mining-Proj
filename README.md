Markdown

# IT160IU - Data Mining Project: World Happiness Prediction 

## 1. Project Overview
This project builds a Data Mining framework to predict the "Happiness Score" (Life Ladder) of countries based on socio-economic indicators using the **World Happiness Report 2024** dataset.

The framework is implemented in **Java** using the **Weka** library.

## 2. Project Structure
```text
IT160IU_Project/
│
├── lib/
│   └── weka.jar              # Core Weka Library (Required)
│
├── data/
│   ├── World Happiness Report 2024.csv  # Raw Input
│   └── world-happiness.arff             # Processed Data (Generated automatically)
│
├── src/
│   ├── Preprocessing.java    # STEP 1: Cleaning, Missing Values, Outliers, Normalization
│   └── Main.java             # STEP 2: Main workflow & Random Forest Implementation
│
└── README.md
```
## 3. Current Progress 

### Step 1: Pre-processing (Completed)
**File:** `src/Preprocessing.java`

The data pipeline has been fully implemented with the following steps:
* **Conversion:** Converted raw CSV to ARFF format.
* **Cleaning:** Successfully removed duplicates.
* **Missing Values:** Handled using Mean/Mode Imputation (`ReplaceMissingValues`).
* **Outliers:** Removed extreme values using `InterquartileRange` (IQR).
* **Feature Selection:** Removed ID attributes (`Country name`, `Year`) to prevent overfitting.
* **Transformation:** Applied **Min-Max Normalization [0, 1]** to scale data for better model performance.

### Step 2: Classification/Prediction Algorithm (Completed)
**File:** `src/Main.java`

The classification module has been fully implemented with the following steps:
* **Random Forest:** Applied the Random Forest algorithm to predict the "Happiness Score" (Life Ladder) of countries based on socio-economic indicators.
* **Feature Selection:** Selected the most important features using the Random Forest algorithm.
* **Model Evaluation:** Evaluated the model using the Random Forest algorithm.

### Step 3: Cluster Feature Generation (Completed)
**File:** : ` src/ClusterFeatureGenerator.java`

The clustering module has been fully implemented with the following steps:
* **K-Means Clustering:** Applied the SimpleKMeans algorithm to group data points based on feature similarity, using a configurable number of clusters.
* **Class-Safe Clustering:** The class attribute was excluded during K-Means training to prevent label leakage and ensure unbiased clustering.
* **Feature Engineering:** After clustering, a new attribute named cluster_id was added to the dataset to represent the assigned cluster for each instance.
* **Nominal Conversion:** The numeric cluster labels were converted to a nominal attribute, ensuring compatibility with classification algorithms that require categorical inputs.
* **Dataset Augmentation:** The original dataset was preserved while the new cluster feature was appended, allowing downstream models to evaluate the performance impact of clustering-based enrichment.


## How to Run the Code

Requirements: Java Development Kit (JDK) 8 or higher.

### Compile (Windows PowerShell)
```powershell
javac -d bin -cp "lib/weka.jar" src/*.java
```

### Run
Use this command to avoid `InaccessibleObjectException` (required for Java 16+):

```powershell
java --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED -cp "bin;lib/weka.jar" Main
```

### Alternative Run (For older Java versions)
If you are using an older Java version (e.g., Java 8), you can try this simpler command:

```powershell
java -cp "bin;lib/weka.jar" Main
```
