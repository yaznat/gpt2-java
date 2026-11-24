package gpt2;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.Dense;
import jflow.layers.Dropout;
import jflow.model.FunctionalLayer;
import jflow.model.Layer;

import static jflow.model.Builder.*;

public class MultiHeadAttention extends FunctionalLayer {
    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final double dropoutRate;

    // Layers for Q, K, V, and output
    private Dense qProj;
    private Dense kProj;
    private Dense vProj;
    private Dense outProj;
    private Dropout dropout;
    
    // Cache for backprop and debug
    private JMatrix attentionWeights;
    private JMatrix headOutputs;
    private JMatrix Q;
    private JMatrix K;
    private JMatrix V;
    private JMatrix gradAttentionWeights;
    private JMatrix gradHeadOutputs;
    
    public MultiHeadAttention(int embedDim, double dropoutRate, int numHeads) {
        super("multi_head_attention");
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.dropoutRate = dropoutRate;
        
        // Ensure embedDim is divisible by numHeads
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException(
                "embedDim must be divisible by numHeads."
            );
        }
        
        this.headDim = embedDim / numHeads;
    }
    
    @Override
    public Layer[] defineLayers() {
        qProj = Dense(embedDim, InputShape(embedDim), false) // No bias
            .initNormal(0, 0.02)            
            .scaleDuringMatmul(false);

        kProj = Dense(embedDim, InputShape(embedDim), false) // No bias
            .initNormal(0, 0.02)           
            .scaleDuringMatmul(false);

        vProj = Dense(embedDim, InputShape(embedDim), false) // No bias
            .initNormal(0, 0.02)        
            .scaleDuringMatmul(false);

        outProj = Dense(embedDim, InputShape(embedDim)) // Has bias
            .initNormal(0, 0.02)         
            .scaleDuringMatmul(false);

        dropout = Dropout(dropoutRate);

        return new Layer[] {qProj, kProj, vProj, dropout, outProj};
    }

    
    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        // Input Shape: (batchSize, seqLen, embedDim, 1)
        int batchSize = input.shape(0);
        int seqLen = input.shape(1);
        
        // Flatten for Dense layers
        JMatrix flattenedInput = input.reshape(batchSize * seqLen, embedDim, 1, 1);
        
        // Project input to Q, K, V
        JMatrix qFlat = qProj.forward(flattenedInput, training);
        JMatrix kFlat = kProj.forward(flattenedInput, training);
        JMatrix vFlat = vProj.forward(flattenedInput, training);
     
        // Reshape to split heads
        Q = qFlat.reshape(batchSize, seqLen, numHeads, headDim);
        K = kFlat.reshape(batchSize, seqLen, numHeads, headDim);
        V = vFlat.reshape(batchSize, seqLen, numHeads, headDim);
        
        // Transpose to (batchSize, numHeads, seqLen, headDim) for matrix multiplication
        Q = Q.permute(0, 2, 1, 3);
        K = K.permute(0, 2, 1, 3);
        V = V.permute(0, 2, 1, 3);

        // Merge batch and head dimensions for batchMatmul()
        Q = Q.reshape(batchSize * numHeads, seqLen, headDim, 1);
        K = K.reshape(batchSize * numHeads, seqLen, headDim, 1);
        V = V.reshape(batchSize * numHeads, seqLen, headDim, 1);


        // Compute attention scores
        JMatrix attentionScores = Q
            .batchMatmul(K.permute(0, 2, 1, 3), false)
            // Standard scale factor: prevents dot products from growing too large
            .multiply(1.0 / Math.sqrt(headDim)) 
            // Reshape to separate batch and head
            .reshape(batchSize, numHeads, seqLen, seqLen);
    

        // Set outputs in the upper right triangle to -Infinity
        applyCausalMask(attentionScores);

        // Apply softmax to each attention window row
        attentionWeights = attentionScores.softmax(3);

        // Apply dropout
        attentionWeights = dropout.forward(attentionWeights, training);

        // Merge batch and head dimensions for batchMatmul()
        attentionWeights = attentionWeights.reshape(batchSize * numHeads, seqLen, seqLen, 1);
        
        // Apply attention weights to values
        headOutputs = attentionWeights
            .batchMatmul(V, false)
            // Reshape to separate batch and head
            .reshape(batchSize, numHeads, seqLen, headDim);

        // Transpose back to (batchSize, seqLen, numHeads, headDim)
        headOutputs = headOutputs.permute(0, 2, 1, 3);

        // Combine heads
        JMatrix combinedHeads = headOutputs.reshape(batchSize, seqLen, embedDim, 1);
        
        // Final projection
        JMatrix flattenedHeads = combinedHeads.reshape(batchSize * seqLen, embedDim, 1, 1);
        JMatrix output = outProj.forward(flattenedHeads, training);

        // Reshape to original dimensions
        output = output.reshape(batchSize, seqLen, embedDim, 1);

        if (!training && !debugEnabled()) {
            clearCaches();
        }

        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix gradOutput) {
        int batchSize = gradOutput.length();
        int seqLen = gradOutput.channels();
        
        // Backpropagate through output projection
        JMatrix flattenedGradOutput = gradOutput.reshape(batchSize * seqLen, embedDim, 1, 1);
        JMatrix gradCombinedHeads = outProj.backward(flattenedGradOutput);
        
        // Reshape and transpose back to head format
        gradHeadOutputs = gradCombinedHeads.reshape(batchSize, seqLen, numHeads, headDim);
        gradHeadOutputs = gradHeadOutputs.permute(0, 2, 1, 3);
        
        // Merge batch and head for batchMatmul()
        gradHeadOutputs = gradHeadOutputs.reshape(batchSize * numHeads, seqLen, headDim, 1);
        
        // Calculate gradient of V
        JMatrix gradV = attentionWeights.permute(0,2,1,3).batchMatmul(gradHeadOutputs, false);
        
        // Calculate gradient of attention weights
        gradAttentionWeights = gradHeadOutputs.batchMatmul(V.permute(0,2,1,3), false);

        // Apply dropout gradient
        gradAttentionWeights = dropout.backward(gradAttentionWeights);

        // Reshape back to separate dimensions for softmax gradient
        gradAttentionWeights = gradAttentionWeights.reshape(batchSize, numHeads, seqLen, seqLen);
        
        // Apply softmax gradient
        JMatrix attentionWeightsReshaped = attentionWeights.reshape(batchSize, numHeads, seqLen, seqLen);
        JMatrix gradAttentionScores = applySoftmaxGradient(gradAttentionWeights, attentionWeightsReshaped);
        
        // Zero out gradients in the upper left triangle
        applyCausalMaskGradient(gradAttentionScores);

        // Merge batch and head for batchMatmul()
        gradAttentionScores = gradAttentionScores.reshape(batchSize * numHeads, seqLen, seqLen, 1);
        
        // Calculate gradients of Q and K
        JMatrix gradQ = gradAttentionScores
            .batchMatmul(K, false)
            .multiply(1.0 / Math.sqrt(headDim));
        JMatrix gradK = gradAttentionScores
            .permute(0,2,1,3)
            .batchMatmul(Q, false)
            .multiply(1.0 / Math.sqrt(headDim));
        
        // Reshape to separate batch and head
        gradQ = gradQ.reshape(batchSize, numHeads, seqLen, headDim);
        gradK = gradK.reshape(batchSize, numHeads, seqLen, headDim);
        gradV = gradV.reshape(batchSize, numHeads, seqLen, headDim);
                
        // Transpose back to (batchSize, seqLen, numHeads, headDim)
        gradQ = gradQ.permute(0, 2, 1, 3);
        gradK = gradK.permute(0, 2, 1, 3);
        gradV = gradV.permute(0, 2, 1, 3);
        
        // Reshape and prepare for dense layer backprop
        JMatrix gradQReshaped = gradQ.reshape(batchSize, seqLen, embedDim, 1);
        JMatrix gradKReshaped = gradK.reshape(batchSize, seqLen, embedDim, 1);
        JMatrix gradVReshaped = gradV.reshape(batchSize, seqLen, embedDim, 1);
        
        // Flatten for backprop through dense layers
        JMatrix gradQFlat = gradQReshaped.reshape(batchSize * seqLen, embedDim, 1, 1);
        JMatrix gradKFlat = gradKReshaped.reshape(batchSize * seqLen, embedDim, 1, 1);
        JMatrix gradVFlat = gradVReshaped.reshape(batchSize * seqLen, embedDim, 1, 1);

        // Backpropagate through Q, K, V projections
        JMatrix gradInputQ = qProj.backward(gradQFlat);
        JMatrix gradInputK = kProj.backward(gradKFlat);
        JMatrix gradInputV = vProj.backward(gradVFlat);
        
        // Sum the gradients from Q, K, V paths
        JMatrix dInput = gradInputQ.add(gradInputK).add(gradInputV);

        // Reshape to original input dimensions
        dInput = dInput.reshape(batchSize, seqLen, embedDim, 1);

        if (!debugEnabled()) {
            clearCaches();
        }

        return trackGradient(dInput);
    }

    private void clearCaches() {
        attentionWeights = null;
        headOutputs = null;
        Q = null;
        K = null;
        V = null;
        gradHeadOutputs = null;
        gradAttentionWeights = null;
    }

    // Fill the top right triangle of each (seqLen, seqLen) matrix in a tensor with shape (batchSize, numHeads, seqLen, seqLen)
    private void upperRightTriangleMask(JMatrix scores, float fillValue) {
        final int batchSize = scores.shape(0);
        final int numHeads = scores.shape(1); 
        final int seqLen = scores.shape(2);
        
        IntStream.range(0, batchSize * numHeads).parallel().forEach(batchHead -> {
            int b = batchHead / numHeads;
            int h = batchHead % numHeads;
            
            for (int i = 0; i < seqLen; i++) {
                for (int j = i + 1; j < seqLen; j++) {
                    scores.set(b, h, i, j, fillValue);
                }
            }
        });
    }
    
    // Apply causal mask to attention scores
    public void applyCausalMask(JMatrix attentionScores) {
        upperRightTriangleMask(attentionScores, Float.NEGATIVE_INFINITY);
    }

    // Apply causal mask to gradients
    private void applyCausalMaskGradient(JMatrix gradAttentionScores) {
        upperRightTriangleMask(gradAttentionScores, 0.0f);
    }

    // Calculate softmax gradient
    private JMatrix applySoftmaxGradient(JMatrix dOut, JMatrix softmax) {
        int B = dOut.shape(0);    
        int H = dOut.shape(1);   
        int S = dOut.shape(2);  
    
        JMatrix result = dOut.zerosLike();
    
        IntStream.range(0, B * H).parallel().forEach(bh -> {
            int b = bh / H;
            int h = bh % H;
    
            for (int i = 0; i < S; i++) {
                float dot = 0f;
                for (int j = 0; j < S; j++) {
                    dot += dOut.get(b, h, i, j) * softmax.get(b, h, i, j);
                }
                for (int j = 0; j < S; j++) {
                    float s = softmax.get(b, h, i, j);
                    float d = dOut.get(b, h, i, j);
                    result.set(b, h, i, j, s * (d - dot));
                }
            }
        });
    
        return result;
    }

    @Override
    public JMatrix[] forwardDebugData() {
        return new JMatrix[] {
            attentionWeights.label("attentionWeights"),
            headOutputs.label("headOutputs"),
            getOutput().label("outputProjection")
        };
    }

    @Override
    public JMatrix[] backwardDebugData() {
        return new JMatrix[] {
            gradHeadOutputs.label("dHeadOutputs"),
            gradAttentionWeights.label("dAttentionWeights"),
            getGradient().label("dInput")
        };
    }
}