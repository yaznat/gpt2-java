import static jflow.model.Builder.Adam;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

import jflow.data.JMatrix;
import jflow.model.Optimizer;
import jflow.model.Sequential;
import jflow.utils.AnsiCodes;
import jflow.utils.Callbacks;
import jflow.utils.Metric;
import util.FineTuneLoader;
import util.GPT2Tokenizer;
import util.LossUtils;
import util.ModelConfig;
import util.TextGenUtils;
import gpt2.GPT2Builder;

/**
 * Smart fine-tuning setup for GPT-2. Training can be interrupted at any point and no progress will be lost. <p>
 * Model weights are saved upon exit -- simply restart to resume training. 
 * You can decrease NUM_EPOCHS to match progress before exit.
 * This setup assumes the dataset is a directory of text files, one per example.
 * 
 * <p><b>REQUIRED: </b> see README.md for installation instructions: https://github.com/yaznat/gpt2-java
 * 
 * You may need to configure your memory allocation:
 *
 * System memory requirement:
 * - 9gb for SMALL 124M (If you have 16gb available, allocate exactly 16gb for
 * peak performance)
 * 
 * Heap limit may need to be configured. Add these flags to the java command for 9gb: "-Xmx9g -Xss256k"
 */
public class FineTuneGPT2 {
    // ================ DEFINE CONSTANTS ================
    private static final String PATH_TO_DATASET = "data/fine-tune-set";
    private static final String PATH_TO_PRETRAINED = "saved_models/gpt2_124M/openwebtext_pretrained_final";

    // Enable debug if desired -- prints live output in the terminal for every layer of GPT-2
    final static boolean DEBUG_ENABLED = false;

    /*
     * Model configuration - 124M param version (No pretrained weights currently for MEDIUM, LARGE, or XL). 
     */
    private final static ModelConfig modelConfig = ModelConfig.SMALL;
    private static final int VOCAB_SIZE = modelConfig.getVocabSize();
    private static final int MAX_SEQ_LEN = modelConfig.getMaxSeqLen();

    // Training constants
    private static final int NUM_EPOCHS = 3;
    private static final int EXAMPLES_PER_UPDATE = 100; // Gradient accumulation before weight update

    // Weights will be saved here upon program exit
    private static final String INTERRUPTED_PATH = "saved_models/" + modelConfig.getName() + "/fine_tuned_interrupted";
    // Weights will be saved here when training is complete
    private static final String FINAL_SAVE_PATH = "saved_models/" + modelConfig.getName() + "/fine_tuned_final";


    // Configuration for sample outputs
    private static final float TEMPERATURE = 0.5f;
    private static final float TOP_P = 0.9f;
    private static final float REPEAT_PENALTY = 1.2f;
    private static final int REPEAT_PENALTY_RANGE = 128;

    private static final int NUM_SAMPLE_TOKENS = 40;
    private static final int EXAMPLES_PER_SAMPLE_OUTPUT = 300;

    private static final String[] SAMPLE_TEXT_STARTERS = new String[] { 
        // Prompts vary based on dataset
        "Give me a recipe for a sandwich.",
        "Please give me a recipe for sushi.",
        "What's a good lamb recipe?",
        "Fix the grammar: She gone to the mall.",
        "Improve grammaticality: The fish is died.",
        "fix: I been sick for many day."
    };


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

    /**
      * Applies weight updates to the model using a given optimizer. Scales accumulated gradients by {@code 1 / examplesSeen}`.
      */
    private static void applyOptimizer(Sequential model, Optimizer optimizer, int examplesSeen) {
        HashMap<String, JMatrix[]> weightUpdates = model.parameterGradients();
        // Average accumulated gradients
        for (JMatrix[] layerUpdates : weightUpdates.values()) {
            for (JMatrix update : layerUpdates) {
                update.divideInPlace(examplesSeen);
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

        // Create tokenizer and data sequencer
        GPT2Tokenizer tokenizer = new GPT2Tokenizer();
        FineTuneLoader loader = new FineTuneLoader(
            PATH_TO_DATASET,
            tokenizer
        );

        // Create the model
        Sequential model = GPT2Builder.create(modelConfig)
            .setDebug(DEBUG_ENABLED)
            .disableGradientStorage() // Save memory (not applied when debug is enabled)
            .summary(JMatrix.zeros(2, 32, 1, 1)); // Print a summary

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

        // Add shutdown hook for interruption
        Thread shutDownHook = new Thread(() -> {
            System.out.println("\nInterrupted! Saving current model...");
            model.saveWeights(INTERRUPTED_PATH);
        });
        Runtime.getRuntime().addShutdownHook(shutDownHook);


// ================ LOAD WEIGHTS AND COMPILE THE MODEL ================

        model.loadWeights(PATH_TO_PRETRAINED);

        Optimizer optimizer = Adam(
            0.00002, // Learning rate
            0.9, // beta1
            0.95 // beta2
        )
        .clipNorm(1.0); // Global clip norm for weight updates

        model.compile(optimizer);


// ================ BEGIN TRAINING ================

        Callbacks.printTrainingHeader(model);

        // Track during training
        double prevLoss = Double.POSITIVE_INFINITY;
        double prevPerplexity = Double.POSITIVE_INFINITY;

        // Main training loop
        int numExamples = loader.numExamples();
        for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
            // Unique shuffle each epoch
            loader.shuffle(SEED + epoch);

            double epochLoss = 0;
            long epochStartTime = System.nanoTime();
            for (int e = 1; e <= numExamples; e++) {
                // Get data
                JMatrix example = loader.get(e - 1);

                // Pass data through the model
                epochLoss += trainStep(model, example);

                int modUpdate = e % EXAMPLES_PER_UPDATE;
                long timeSinceStart = System.nanoTime() - epochStartTime;

                // Print a progress callback
                LinkedHashMap<String, Double> lossReport = new LinkedHashMap<>();
                lossReport.put("Loss", epochLoss / e);
                Callbacks.printProgressCallback(
                    "Epoch", epoch, NUM_EPOCHS,
                    "Step", e, numExamples,
                    timeSinceStart,
                    lossReport
                );

                // Update model weights
                if (modUpdate == 0) {
                    applyOptimizer(model, optimizer, EXAMPLES_PER_UPDATE);
                }

                // Generate sample text
                if (e % EXAMPLES_PER_SAMPLE_OUTPUT == 0) {
                    System.out.println();
                    generateSample(model, tokenizer, NUM_SAMPLE_TOKENS);
                }
            }

            // Avoid wasting remainder examples
            int remainder_examples = numExamples % EXAMPLES_PER_UPDATE;
            if (remainder_examples != 0) {
                applyOptimizer(model, optimizer, remainder_examples);
            }

            // Prepare metrics for display
            double averageLoss = epochLoss / numExamples;
            double perplexity = Math.exp(averageLoss);
            List<Metric> metricReport = new ArrayList<>();
            metricReport.add(new Metric("Cross Entropy Loss", averageLoss, false, averageLoss < prevLoss));
            metricReport.add(new Metric("Perplexity", perplexity, false, perplexity < prevPerplexity));

            // Print a metric callback
            Callbacks.printMetricCallback(metricReport);

            // Generate a sample output
            generateSample(model, tokenizer, NUM_SAMPLE_TOKENS);
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