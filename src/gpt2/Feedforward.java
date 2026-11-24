package gpt2;

import jflow.data.JMatrix;
import jflow.layers.Dense;
import jflow.layers.Dropout;
import jflow.layers.GELU;
import jflow.model.FunctionalLayer;
import jflow.model.Layer;

import static jflow.model.Builder.*;

public class Feedforward extends FunctionalLayer {
    private final int embedDim;
    private final double dropoutRate;

    private int batchSize;
    private int seqLen;

    // Internal Layers
    private Dense dense1;
    private Dense dense2;
    private GELU gelu;
    private Dropout dropout;

    public Feedforward(int embedDim, double dropoutRate) {
        super("feed_forward");
        this.embedDim = embedDim;
        this.dropoutRate = dropoutRate;
    }

    @Override
    public Layer[] defineLayers() {
        int hiddenDim = embedDim * 4;
        dense1 = Dense(hiddenDim, InputShape(embedDim))
            .initNormal(0, 0.02)       
            .scaleDuringMatmul(false);

        dense2 = Dense(embedDim)
            .initNormal(0, 0.02)
            .scaleDuringMatmul(false);

        gelu = GELU();
        dropout = Dropout(dropoutRate);

        return new Layer[] {dense1, gelu, dense2, dropout};
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        batchSize = input.shape(0);
        seqLen = input.shape(1);

        // Flatten for Dense
        JMatrix x = input.reshape(batchSize * seqLen, embedDim, 1, 1);

        x = dense1.forward(x, training);
        x = gelu.forward(x, training);

        x = dense2.forward(x, training);
        x = dropout.forward(x, training);

        // Reshape to original dimensions
        x = x.reshape(batchSize, seqLen, embedDim, 1);

        return trackOutput(x, training);
    }

    @Override
    public JMatrix backward(JMatrix dOutput) {
        int batchSize = dOutput.shape(0);
        int seqLen = dOutput.shape(1); 
        
        // Flatten for Dense  
        JMatrix dX = dOutput.reshape(batchSize * seqLen, embedDim, 1, 1);
        
        dX = dropout.backward(dX);
        dX = dense2.backward(dX);

        dX = gelu.backward(dX);
        dX = dense1.backward(dX);
        
        // Reshape to original dimensions
        dX = dX.reshape(batchSize, seqLen, embedDim, 1);

        return trackGradient(dX);
    }
}
