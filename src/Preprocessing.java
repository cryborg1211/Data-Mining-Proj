import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.InterquartileRange;
import java.io.File;
import java.util.HashSet;
import weka.filters.unsupervised.attribute.Normalize;

public class Preprocessing {

    public static Instances loadAndCleanData(String csvPath, String arffPath) {
        try {
            System.out.println("--- START PREPROCESSING ---");
            
            // 1. CONVERT CSV -> ARFF
            File csvFile = new File(csvPath);
            if (!csvFile.exists()) {
                System.err.println("Error: File CSV not found: " + csvPath);
                return null;
            }
            
            System.out.println("1. Loading & Converting CSV...");
            CSVLoader loader = new CSVLoader();
            loader.setSource(csvFile);
            Instances rawData = loader.getDataSet();
            System.out.println("   -> Raw data size: " + rawData.numInstances() + " rows.");

            // 2. REMOVING DUPLICATES 
            System.out.println("2. Removing Duplicates...");
            Instances uniqueData = removeDuplicates(rawData);
            System.out.println("   -> After removing duplicates: " + uniqueData.numInstances() + " rows.");

            // 3. HANDLING MISSING VALUES
            System.out.println("3. Handling Missing Values...");
            ReplaceMissingValues fixMissing = new ReplaceMissingValues(); // Fill missing values with mean
            fixMissing.setInputFormat(uniqueData);
            Instances noMissingData = Filter.useFilter(uniqueData, fixMissing);

            // 4. ADDRESSING OUTLIERS
            System.out.println("4. Handling Outliers (InterquartileRange)...");
            InterquartileRange iqr = new InterquartileRange();
            iqr.setInputFormat(noMissingData);
            iqr.setAttributeIndices("3-last"); // Apply IQR to all attributes except ID
            iqr.setOutlierFactor(3.0);
            iqr.setExtremeValuesFactor(6.0);
            Instances noOutlierData = Filter.useFilter(noMissingData, iqr);
            System.out.println("   -> After removing outliers: " + noOutlierData.numInstances() + " rows.");

            // 5. REMOVING ID COLUMNS
            System.out.println("5. Removing ID attributes (Country name, Year)...");
            Remove remove = new Remove();
            remove.setAttributeIndices("1,2"); // Remove ID columns
            remove.setInputFormat(noOutlierData);
            Instances dataNoID = Filter.useFilter(noOutlierData, remove);

            // 6. DATA TRANSFORMATION: NORMALIZATION
            System.out.println("6. Normalizing Data (Min-Max [0,1])...");
            Normalize norm = new Normalize(); // Normalize data to [0,1]
            norm.setInputFormat(dataNoID);
            Instances finalData = Filter.useFilter(dataNoID, norm);

            // 7. SAVE FINAL ARFF
            ArffSaver saver = new ArffSaver();
            saver.setInstances(finalData);
            saver.setFile(new File(arffPath)); 
            saver.writeBatch();
            System.out.println("Saved clean ARFF at: " + arffPath);
            System.out.println("--- PREPROCESSING DONE ---\n");
            return finalData;

        } catch (Exception e) {
            System.err.println("Preprocessing Failed!");
            e.printStackTrace();
            return null;
        }
    }

    // Remove duplicate rows class
    private static Instances removeDuplicates(Instances data) {
        Instances textUnique = new Instances(data, 0);
        HashSet<String> set = new HashSet<>();

        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);                 
            String key = inst.toString();                   
            
            if (!set.contains(key)) {
                set.add(key);
                textUnique.add(inst);
            }
        }
        return textUnique;
    }
}