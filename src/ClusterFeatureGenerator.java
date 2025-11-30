import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

public class ClusterFeatureGenerator {

    /**
     * Load dataset, apply K-Means clustering, and add cluster ID as new attribute.
     * @param inputFile Path to ARFF or CSV file
     * @param numClusters Number of clusters for K-Means
     * @return Instances with added cluster feature
     * @throws Exception
     */
    public static Instances addClusterFeature(String inputFile, int numClusters) throws Exception {
        // 1. Load dataset
        Instances data = DataSource.read(inputFile);
        if (data.classIndex() == -1) {
            // Set last attribute as class by default (optional)
            data.setClassIndex(data.numAttributes() - 1);
        }

        // 2. Prepare data for clustering (remove class)
        Instances clusterData = new Instances(data);
        clusterData.setClassIndex(-1); // ignore target for clustering

        // 3. Build K-Means
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(numClusters);
        kmeans.setSeed(42);
        kmeans.buildClusterer(clusterData);

        // 4. Add new numeric attribute for cluster ID
        Add addFilter = new Add();
        addFilter.setAttributeIndex("last");
        addFilter.setAttributeName("cluster_id");
        addFilter.setInputFormat(data);
        Instances newData = Filter.useFilter(data, addFilter);

        // 5. Assign cluster IDs
        for (int i = 0; i < clusterData.numInstances(); i++) {
            int clusterId = kmeans.clusterInstance(clusterData.instance(i));
            newData.instance(i).setValue(newData.numAttributes() - 1, clusterId);
        }

        System.out.println("K-Means clustering done! Added 'cluster_id' feature.");
        return newData;
    }
}
