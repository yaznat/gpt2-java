package gpt2;

import jflow.data.JMatrix;
import jflow.layers.LayerNorm;
import jflow.model.FunctionalLayer;
import jflow.model.Layer;
import static jflow.model.Builder.*;

public class TransformerBlock extends FunctionalLayer{
    private int numHeads;
    private int embedDim;
    private double dropoutRate;

    // Internal Layers
    private MultiHeadAttention attention;
    private LayerNorm norm1;
    private LayerNorm norm2;
    private Feedforward feedforward;

    public TransformerBlock(int numHeads, int embedDim, double dropoutRate) {
        super("transformer_block");
        this.numHeads = numHeads;
        this.embedDim = embedDim;
        this.dropoutRate = dropoutRate;
    }

    @Override
    public Layer[] defineLayers() {
        norm1 = LayerNorm().gammaValue(0.5);
        attention = new MultiHeadAttention(embedDim, dropoutRate, numHeads);
        norm2 = LayerNorm().gammaValue(0.5);
        feedforward = new Feedforward(embedDim, dropoutRate);

        return new Layer[]{norm1, attention, norm2, feedforward};
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        JMatrix x = input;
        
        // First sub-block: LayerNorm → Attention → Residual
        JMatrix norm1Out = norm1.forward(x, training);
        JMatrix attnOut = attention.forward(norm1Out, training);

        JMatrix x1 = x.add(attnOut); // Residual connection
        
        // Second sub-block: LayerNorm → Feedforward → Residual
        JMatrix norm2Out = norm2.forward(x1, training);
        JMatrix ffOut = feedforward.forward(norm2Out, training);

        JMatrix x2 = x1.add(ffOut); // Residual connection
        
        return trackOutput(x2, training);
    }

    @Override
    public JMatrix backward(JMatrix dOutput) {
        // Second residual: x2 = x1 + ffOut
        JMatrix dX1_fromResidual2 = dOutput; // Gradient flowing to x1
        JMatrix dFFOut = dOutput; // Gradient flowing to ffOut
        
        // Backprop through feedforward
        JMatrix dNorm2Out = feedforward.backward(dFFOut);
        
        // Backprop through norm2
        JMatrix dX1_fromNorm2 = norm2.backward(dNorm2Out);
        
        // Combine gradients at x1
        JMatrix dX1 = dX1_fromResidual2.add(dX1_fromNorm2);
        
        // First residual: x1 = x + attnOut
        JMatrix dX_fromResidual1 = dX1; // Gradient flowing to x (input)
        JMatrix dAttnOut = dX1; // Gradient flowing to attnOut
        
        // Backprop through attention
        JMatrix dNorm1Out = attention.backward(dAttnOut);
        
        // Backprop through norm1
        JMatrix dX_fromNorm1 = norm1.backward(dNorm1Out);
        
        // Combine gradients at input x
        JMatrix dInput = dX_fromResidual1.add(dX_fromNorm1);

        return trackGradient(dInput);
    }

    @Override
    public int[] outputShape() {
        return feedforward.outputShape();
    }
}