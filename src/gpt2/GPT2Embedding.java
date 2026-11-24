package gpt2;

import jflow.layers.Embedding;
import jflow.model.FunctionalLayer;
import jflow.model.Layer;

import static jflow.model.Builder.*;

import jflow.data.JMatrix;

public class GPT2Embedding extends FunctionalLayer {
    // Internal Layers
    private Embedding tokenEmbed;
    private Embedding posEmbed;


    public GPT2Embedding(int vocabSize, int embedDim, int maxSeqLength) {
        super("gpt2_embedding");
        this.tokenEmbed = Embedding(vocabSize, embedDim);
        this.posEmbed = Embedding(maxSeqLength, embedDim);
    }

    public GPT2Embedding weightTie(float[] matrix, boolean removeFromParamCount) {
        tokenEmbed.weightTie(matrix, removeFromParamCount);
        return this;
    }

    @Override
    public Layer[] defineLayers() {
        return new Layer[] {tokenEmbed, posEmbed};
    }

    @Override
    public JMatrix forward(JMatrix tokenIDs, boolean training) {
        JMatrix tok = tokenEmbed.forward(tokenIDs, training);
        JMatrix pos = posEmbed.forward(JMatrix.positionIDs(tokenIDs), training);
        JMatrix added = tok.add(pos);

        return trackOutput(added, training);
    }

    @Override
    public JMatrix backward(JMatrix dOutput) {
        tokenEmbed.backward(dOutput);
        posEmbed.backward(dOutput);
        return null;  // No gradient with respect to discrete token inputs
    }

    @Override
    public JMatrix[] forwardDebugData() {
        return new JMatrix[]{getOutput().label("tok + pos output")};
    }
}
