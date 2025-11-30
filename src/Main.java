import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class Main {
    public static void main(String[] args) {
        try {
             // ---------------- STEP 1: PRE-PROCESSING ----------------
            System.out.println(">>> STEP 1: PRE-PROCESSING...");
            String csvPath = "data/World Happiness Report 2024.csv";
            String arffPath = "data/world-happiness.arff";
            Instances data = Preprocessing.loadAndCleanData(csvPath, arffPath);

            if (data == null) {
                System.err.println("Error: cannot load data!"); //Debug
                return;
            }
            
            // SETUP TARGET 
            if (data.attribute("Life Ladder") != null) {
                data.setClass(data.attribute("Life Ladder"));
            } else {
                data.setClassIndex(data.numAttributes() - 1);   
            }
            
            System.out.println("Target Attribute: " + data.classAttribute().name());

            // ---------------- STEP 2: RANDOM FOREST ----------------
            System.out.println("\n>>> STEP 2: APPLYING ALGORITHM (Random Forest)...");
            System.out.println("(Running 10-fold Cross-validation...)");
            
            RandomForest model = new RandomForest();
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));

            // REPORT RESULTS 
            System.out.println(eval.toSummaryString("\n=== FINAL PREDICTION RESULTS ===\n", false));
            System.out.println("Accuracy: " + String.format("%.2f%%", (1 - eval.errorRate()) * 100));
            
            // ---------------- STEP 3: ADD CLUSTER FEATURE ----------------
            System.out.println("\n>>> STEP 3: ADDING CLUSTER FEATURE...");
            int numClusters = 5; // Example: set number of clusters
            Instances clusteredData = ClusterFeatureGenerator.addClusterFeature(arffPath, numClusters);

            // SET TARGET 
            if (clusteredData.attribute("Life Ladder") != null) {
                clusteredData.setClass(clusteredData.attribute("Life Ladder"));
            } else {
                clusteredData.setClassIndex(clusteredData.numAttributes() - 1);
            }

            System.out.println("Data with cluster feature has " + clusteredData.numAttributes() + " attributes.");
            

            // ---------------- STEP 3b: RANDOM FOREST ON CLUSTERED DATA ----------------
            System.out.println("\n>>> STEP 3b: RANDOM FOREST WITH CLUSTER FEATURE...");
            RandomForest modelClustered = new RandomForest();
            Evaluation evalClustered = new Evaluation(clusteredData);
            evalClustered.crossValidateModel(modelClustered, clusteredData, 10, new Random(1));

            System.out.println(evalClustered.toSummaryString("\n=== STEP 3 RESULTS ===\n", false));
            System.out.println("Accuracy with cluster feature: " + String.format("%.2f%%", (1 - evalClustered.errorRate()) * 100));
            
            // ---------------- STEP 3c: NAIVE BAYES ----------------
            System.out.println("\n>>> STEP 3c: NAIVE BAYES WITH CLUSTER FEATURE...");
            NaiveBayesClassifier.runNaiveBayes(arffPath, 3); // 3 bins: Low/Medium/High



        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}