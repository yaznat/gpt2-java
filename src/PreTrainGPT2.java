import static jflow.model.Builder.Adam;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import gpt2.GPT2Builder;
import jflow.data.JMatrix;
import jflow.model.Optimizer;
import jflow.model.Sequential;
import jflow.utils.AnsiCodes;
import jflow.utils.Callbacks;
import jflow.utils.Metric;
import util.GPT2Tokenizer;
import util.LossUtils;
import util.ModelConfig;
import util.SequenceLoader;
import util.TextGenUtils;

/**
 * Smart training setup for GPT-2. Training can be stopped at any point and
 * no progress will be lost. <p>
 * The loader remembers progress in the training file, and model weights are
 * saved upon exit - simply restart to resume training.
 * This setup assumes the dataset is one large text file.
 * 
 * <p><b>REQUIRED: </b> see README.md for installation instructions: https://github.com/yaznat/gpt2-java
  * System memory requirement:
 * - 9gb for SMALL 124M (If you have 16gb available, allocate exactly 16gb for
 * peak performance)
 * 
 * Heap limit may need to be configured. Add these flags to the java command for 9gb: "-Xmx9g -Xss256k"
 */
public class PreTrainGPT2 {
    // ================ DEFINE CONSTANTS ================
    private static final String PATH_TO_DATASET = "Moby Dick.txt";

    // Enable debug if desired -- prints live output in the terminal for every layer of GPT-2
    final static boolean DEBUG_ENABLED = false;

    /*
     * SMALL Model configuration - 124M param model, good for simple tasks.
     * Select ModelConfig.MEDIUM for the 355M param model, slower to train and requires more RAM.
     * Select ModelConfig.LARGE or ModelConfig.XL to see the model summary, but training isn't feasible.
     */
    private final static ModelConfig modelConfig = ModelConfig.SMALL;

    private static final int VOCAB_SIZE = modelConfig.getVocabSize();
    private static final int MAX_SEQ_LEN = modelConfig.getMaxSeqLen();

    // Configure model save dirs
    private static final String DATASET_NAME = Paths.get(PATH_TO_DATASET).getFileName().toString();
    private static final String CHECKPOINT_PATH = "saved_models/" + modelConfig.getName() + "/" + DATASET_NAME + "_pretrained_checkpoint";
    private static final String FINAL_SAVE_PATH = "saved_models/" + modelConfig.getName() + "/" + DATASET_NAME + "_pretrained_final";

    // For training
    private static final int BATCH_SIZE = 3;
    private static final int SEQ_LEN = 512; // Decrease if necessary
    private static final int TOKENS_PER_UPDATE = 30720;
    private static final int VIRTUAL_BATCH_SIZE = TOKENS_PER_UPDATE / SEQ_LEN;
    private static final int BATCHES_PER_UPDATE = VIRTUAL_BATCH_SIZE / BATCH_SIZE; // Apply gradient updates every N batches
    private static final double STRIDE_PERCENTAGE = 0.5; // Data overlap between batches
    private static final String[] SAMPLE_TEXT_STARTERS = new String[] { 
       /*
        * Varied context: law, movies, food, recipes.
        * Obvious conclusions: peanut butter and jelly sandwich, grass is green.
        */
        "According to state law, it's illegal to",
        "I love ice cream! My favorite flavor is",
        "I love movies! My favorite movie is",
        "For lunch, I had yogurt, apple slices, and a peanut butter and",
        "There are many species of grass. The color of grass is usually",
        "My grandmother's famous recipe for chocolate chip cookies:\n"
    };

    // Configuration for sample model outputs
    private static final float TEMPERATURE = 0.5f;
    private static final float TOP_P = 0.9f;
    private static final float REPEAT_PENALTY = 1.2f;
    private static final int REPEAT_PENALTY_RANGE = 128;


    // ================ DEFINE FUNCTIONS FOR TRAINING ================

    /**
     * Performs one forward and backward pass on a batch without updating weights.
     * Accumulates gradients for later optimizer steps.
     */
    private static double trainStep(Sequential model, JMatrix batch) {
        int batchSize = batch.shape(0);
        int seqLen = batch.shape(1) - 1; // batch has one extra token for input -> target shifting

        JMatrix input = JMatrix.zeros(
            batchSize,
            seqLen,
            1,
            1
        );
        JMatrix target = input.zerosLike();

        // Offset by 1 by per batch
        for (int i = 0; i < batchSize; i++) {
            JMatrix singleSequence = batch.getItem(i);

            JMatrix singleInput = singleSequence.slice(0, singleSequence.size() - 1);
            JMatrix singleTarget = singleSequence.slice(1, singleSequence.size());

            input.arrayCopyBatch(i, singleInput);
            target.arrayCopyBatch(i, singleTarget);
        }
    
        // One-hot encoded target
        JMatrix targetOneHot = JMatrix.oneHot(target, VOCAB_SIZE, 0.0f);

        // Forward pass
        JMatrix output = model.forward(input, true);

        // Track loss
        double loss = LossUtils.crossEntropyLoss(output, targetOneHot);

        // Calculate gradient for backprop
        JMatrix gradient = LossUtils.oneHotGradient(output, targetOneHot);

        // Backward pass
        model.backward(gradient);

        return loss;
    }

    private static void applyOptimizer(Sequential model, Optimizer optimizer, int batchesSeen) {
        HashMap<String, JMatrix[]> weightUpdates = model.parameterGradients();
        // Average accumulated gradients
        for (JMatrix[] layerUpdates : weightUpdates.values()) {
            for (JMatrix update : layerUpdates) {
                update.divideInPlace(batchesSeen);
            }
        }
        // Apply weight updates
        optimizer.apply(weightUpdates);
    }

    public static void main(String[] args) throws IOException {
        // ================ DEFINE DATASET AND THE MODEL ================

        // Set the seed for reproducability
        final long SEED = 42;
        JMatrix.setSeed(SEED);

        // Create tokenizer and data loader
        GPT2Tokenizer tokenizer = new GPT2Tokenizer();
        SequenceLoader loader = new SequenceLoader(
            PATH_TO_DATASET,
            BATCH_SIZE,
            SEQ_LEN + 1, // Add 1 to seqLen to account for offset between input and target
            (int)(BATCH_SIZE * SEQ_LEN * STRIDE_PERCENTAGE)
        );

        // Automatically remember number of completed batches
        int totalVirtualBatches = loader.totalNumBatches() / BATCHES_PER_UPDATE;
        int batchesCompleted = loader.numBatchesCompleted();

        // Revert to last virtual batch
        batchesCompleted -= batchesCompleted % BATCHES_PER_UPDATE;

        int virtualBatchesCompleted = batchesCompleted / BATCHES_PER_UPDATE;

        // Create the model
        Sequential model = GPT2Builder.create(modelConfig)
            .setDebug(DEBUG_ENABLED)
            .disableGradientStorage() // Save memory
            .summary(JMatrix.zeros(BATCH_SIZE, SEQ_LEN, 1, 1)); // Print a summary

        /*
         * Generate a short untrained snippet to verify reproducability. 
         * Should print: " Maybe disse maternal reins Force rookie LINElbsadjusthea" with SEED = 42
         */ 
        int snippetLength = 10;
        String starter = "avocado";
        TextGenUtils.generate(
            model, 
            tokenizer, 
            starter, 
            snippetLength, 
            MAX_SEQ_LEN, 
            1.0, 
            0.9,
            1.2,
            32,
            SEED, 
            false, 
            true
        );

        // Add shutdown hook for checkpointing
        Thread shutDownHook = new Thread(() -> {
            System.out.println("\nInterrupted! Saving current model...");
            model.saveWeights(CHECKPOINT_PATH + "/fine_tune_interrupted");
        });
        Runtime.getRuntime().addShutdownHook(shutDownHook);


        // ================ LOAD WEIGHTS AND COMPILE THE MODEL ================

        /**
         * According to the OpenAI training configuration
         * (Except uses Adam instead of AdamW)
         */ 
        Optimizer optimizer = Adam(
            0.00025, // Learning rate
            0.9, // beta1
            0.95 // beta2
        )
        .clipNorm(1.0); // Global clip norm

        model.compile(optimizer);

        // Load from checkpoint if it exists
        File checkpointWeights = new File(CHECKPOINT_PATH);
        if (checkpointWeights.exists()) {
            model.loadWeights(CHECKPOINT_PATH);
        }


        // ================ BEGIN TRAINING ================

        Callbacks.printTrainingHeader(model);

        final int NUM_SAMPLE_TOKENS = 40;

        // Track during training
        double virtualBatchLoss = 0;
        long virtualBatchStartTime = System.nanoTime();
        double prevLoss = Double.POSITIVE_INFINITY;
        double prevPerplexity = Double.POSITIVE_INFINITY;

        // Main training loop
        while (loader.hasNext()) {
            // Get data
            JMatrix batch = loader.next();

            // Pass data through the model
            virtualBatchLoss += trainStep(model, batch);

            // Track progress and print a callback
            batchesCompleted++;
            int modUpdate = batchesCompleted % BATCHES_PER_UPDATE;
            int currentBatch = (modUpdate == 0) ? BATCHES_PER_UPDATE : modUpdate;
            long timeSinceStart = System.nanoTime() - virtualBatchStartTime;
            Callbacks.printProgressCallback(
                "Batch", virtualBatchesCompleted + 1, totalVirtualBatches,
                "Step", currentBatch, BATCHES_PER_UPDATE,
                timeSinceStart
            );

            if (modUpdate == 0) {
                virtualBatchesCompleted++;

                // Update model weights
                applyOptimizer(model, optimizer, BATCHES_PER_UPDATE);

                double loss = virtualBatchLoss / BATCHES_PER_UPDATE;
                double perplexity = Math.exp(loss);

                // Prepare metrics for display
                List<Metric> metricReport = new ArrayList<>();
                metricReport.add(new Metric("Cross Entropy Loss", loss, false, loss < prevLoss));
                metricReport.add(new Metric("Perplexity", perplexity, false, perplexity < prevPerplexity));

                // Print a metric callback
                Callbacks.printMetricCallback(metricReport);

                // Generate sample text
                generateSample(model, tokenizer, NUM_SAMPLE_TOKENS);

                // Reset loss
                virtualBatchLoss = 0;

                // Store values
                prevLoss = loss;
                prevPerplexity = perplexity;

                // Renew start time
                virtualBatchStartTime = System.nanoTime();
            }
        }
        System.out.println("Training complete!");
        Runtime.getRuntime().removeShutdownHook(shutDownHook);
        model.saveWeights(FINAL_SAVE_PATH);
    }

    // Sequentially prints sample text as it is generated
    private static void generateSample(Sequential model, GPT2Tokenizer tokenizer, int numSampleTokens) throws IOException {
        String prompt = SAMPLE_TEXT_STARTERS[(int) (Math.random() * SAMPLE_TEXT_STARTERS.length)];

        System.out.print(AnsiCodes.WHITE + "Prompt: " + prompt + "\nResponse: ");
        TextGenUtils.generate(
            model, 
            tokenizer, 
            prompt + "<|endoftext|>", 
            numSampleTokens, 
            MAX_SEQ_LEN, 
            TEMPERATURE, 
            TOP_P,
            REPEAT_PENALTY,
            REPEAT_PENALTY_RANGE,
            System.nanoTime(), 
            true, 
            true
        );
        System.out.print(AnsiCodes.RESET);
    }
}
