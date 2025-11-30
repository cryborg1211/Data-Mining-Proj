import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class NaiveBayesClassifier {

    /**
     * Runs NaiveBayes on the given dataset with the class attribute discretized.
     * @param inputFile Path to the ARFF dataset
     * @param bins Number of bins to discretize the class into
     * @throws Exception
     */
    public static void runNaiveBayes(String inputFile, int bins) throws Exception {
        // Load dataset
        Instances data = DataSource.read(inputFile);
        if (data.classIndex() == -1) {
            // Optionally set a default class index if none is set
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Discretize the class attribute
        Discretize discretize = new Discretize();
        discretize.setBins(bins);
        discretize.setUseEqualFrequency(true);
        // Make sure we target the correct attribute by index
        discretize.setAttributeIndices("" + (data.attribute("Life Ladder").index() + 1));
        discretize.setInputFormat(data);

        Instances discretizedData = Filter.useFilter(data, discretize);

        // Set the class attribute AFTER discretization
        discretizedData.setClass(discretizedData.attribute("Life Ladder"));

        // Verify the class type
        System.out.println("Class type after discretization: " + discretizedData.classAttribute().type());

        // Build and evaluate NaiveBayes
        NaiveBayes nb = new NaiveBayes();
        Evaluation evalNB = new Evaluation(discretizedData);
        evalNB.crossValidateModel(nb, discretizedData, 10, new Random(1));

        // Print results
        System.out.println(evalNB.toSummaryString("\n=== NAIVE BAYES RESULTS ===\n", false));
        System.out.println("NaiveBayes Accuracy: " 
                           + String.format("%.2f%%", (1 - evalNB.errorRate()) * 100));
    }

}
