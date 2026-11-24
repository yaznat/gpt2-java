package gpt2;

import static jflow.model.Builder.Reshape;
import static jflow.model.Builder.Dense;
import static jflow.model.Builder.LayerNorm;

import jflow.data.JMatrix;
import jflow.model.Sequential;
import util.ModelConfig;

public class GPT2Builder {
    public static Sequential create(
        ModelConfig config
    ) {
        final String NAME = config.getName();
        final int VOCAB_SIZE = config.getVocabSize();
        final int EMBED_DIM = config.getEmbedDim();
        final int MAX_SEQ_LEN = config.getMaxSeqLen();
        final int NUM_HEADS = config.getNumHeads();
        final double DROPOUT_PROB = config.getDropoutProb();
        final int NUM_LAYERS = config.getNumLayers();
        /*
         * Create a weight matrix to be shared between
         * the token Embedding and the Dense output projection.
         */ 
        JMatrix sharedWeights = JMatrix.normal(VOCAB_SIZE, EMBED_DIM, 1, 1, 0, 0.02);
        
        Sequential model = new Sequential(NAME);

        // Embedding
        model.add(
            new GPT2Embedding(VOCAB_SIZE, EMBED_DIM, MAX_SEQ_LEN)
                .weightTie(sharedWeights.unwrap(), false)
        );
        
        // Transformer stack
        for (int i = 0; i < NUM_LAYERS; i++) {
            model.add(new TransformerBlock(NUM_HEADS, EMBED_DIM, DROPOUT_PROB));
        }

        model.add(LayerNorm().gammaValue(0.5)); // Custom gamma initialization for stability

        // Flatten for Dense
        model.add(Reshape("merge_batch_seq"));

        // Output projection
        model.add(
            Dense(VOCAB_SIZE, false)
                .scaleDuringMatmul(false)
                .weightTie(sharedWeights.unwrap(), true) // Remove weight matrix from param count
        );

        // Unflatten
        model.add(Reshape("split_batch_seq"));

        return model;
    }
}
