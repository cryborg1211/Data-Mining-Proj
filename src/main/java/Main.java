import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        try {
            // PRE-PROCESSING 
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

            // RUN ALGORITHM (RANDOM FOREST) 
            System.out.println("\n>>> STEP 2: APPLYING ALGORITHM (Random Forest)...");
            System.out.println("(Running 10-fold Cross-validation...)");
            
            RandomForest model = new RandomForest();
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));

            // REPORT RESULTS 
            System.out.println(eval.toSummaryString("\n=== FINAL PREDICTION RESULTS ===\n", false));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
