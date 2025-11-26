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
## 3. Current Progress (Done by Khương)

### Step 1: Pre-processing (Completed)
**File:** `src/Preprocessing.java`

The data pipeline has been fully implemented with the following steps:
* **Conversion:** Converted raw CSV to ARFF format.
* **Cleaning:** Successfully removed duplicates.
* **Missing Values:** Handled using Mean/Mode Imputation (`ReplaceMissingValues`).
* **Outliers:** Removed extreme values using `InterquartileRange` (IQR).
* **Feature Selection:** Removed ID attributes (`Country name`, `Year`) to prevent overfitting.
* **Transformation:** Applied **Min-Max Normalization [0, 1]** to scale data for better model performance.


## How to Run the Code

Requirements: Java Development Kit (JDK) 8 or higher.

Compile (Windows PowerShell)
PowerShell
```text
javac -d bin -cp "lib/weka.jar" src/*.java
```
Run (Fixing Java 21+ "Access Denied" issues)
Use this exact command to avoid InaccessibleObjectException:

PowerShell
```text
java --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED -
```
