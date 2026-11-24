package util;

import jflow.data.JMatrix;

public class LossUtils {
    /**
     * Perform a standard softmax one-hot gradient calculation.
     * @param logits Raw output logits from the model.
     * @param targetOneHot One-hot encoded labels.
     * @return The gradient w.r.t model output.
     */ 
    public static JMatrix oneHotGradient(JMatrix logits, JMatrix targetOneHot) {
        JMatrix softmaxOutput = logits.softmax(2); // Vocab dimension
        JMatrix gradient = softmaxOutput.subtract(targetOneHot);
        return gradient;
    }

    /**
     * Calculate the cross entropy loss of model output.
     * @param logits Raw output logits from the model.
     * @param targetOneHot One-hot encoded labels.
     */
    public static double crossEntropyLoss(JMatrix logits, JMatrix targetOneHot) {
        JMatrix logSoftmax = logits.logSoftmax(2); // Vocab dimension
        JMatrix perTokenLoss = targetOneHot
            .multiply(logSoftmax)
            .sum(1) // Sums over vocab dimension
            .multiply(-1.0);

        // Return the average per-token loss
        return perTokenLoss.mean();
    }
}
