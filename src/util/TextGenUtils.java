package util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import jflow.data.JMatrix;
import jflow.model.Sequential;

public class TextGenUtils {
    /**
     * Generate text using a Sequential model.
     * @param model The model to use for generation.
     * @param tokenizer The tokenizer used with the model.
     * @param inputText The prompt for the model.
     * @param numTokens The number of tokens to generate.
     * @param maxSeqLen The maximum allowed sequence length.
     * @param temperature The temperature to use in sampling. Higher = more random. 0.7 is a good default.
     * @param topP The top-p threshold for nucleus sampling. Values between 0.0 and 1.0. Set to 1.0 to disable.
     * @param repeatPenalty The repeat penalty factor. Values > 1.0 discourage repetition. 1.0 = no penalty, 1.1 is typical.
     * @param repeatPenaltyRange The number of recent tokens to consider for repeat penalty. 0 = use all context.
     * @param seed The seed to use during sampling for reproducability.
     * @param followEOS Terminate generation early if the <|endoftext|> token is generated.
     * @param sequentialPrint Print tokens sequentially to the terminal as they are generated.
     * @return The prompt + generated text in a String.
     * @throws IOException If the tokenizer encounters an error.
     */
    public static String generate(
        Sequential model, 
        GPT2Tokenizer tokenizer,
        String inputText, 
        int numTokens, 
        int maxSeqLen,
        double temperature,
        double topP,
        double repeatPenalty,
        int repeatPenaltyRange,
        long seed,
        boolean followEOS,
        boolean sequentialPrint
    ) throws IOException {
        // Encode input
        JMatrix context = tokenizer.encode(inputText); // (1, seqLen, 1, 1)
        List<Integer> generated = new ArrayList<>();
        List<Integer> allTokens = new ArrayList<>(); // Track all tokens for repeat penalty

        // Use Random for sampling
        Random random = new Random(seed);
        
        // For smart sequential printing
        List<Integer> pendingTokens = new ArrayList<>();

        for (int i = 0; i < numTokens; i++) {
            JMatrix window = context;
            // Trim to maxSeqLen if needed
            if (context.size() > maxSeqLen) {
                window = trimContext(context, maxSeqLen);
            }

            // Forward pass to get logits
            JMatrix output = model.forward(window, false);

            // Get logits for the last token
            float[] logits = extractLogits(output);
            
            // Apply repeat penalty if enabled
            if (repeatPenalty != 1.0) {
                logits = applyRepeatPenalty(logits, allTokens, repeatPenalty, repeatPenaltyRange);
            }

            // Apply temperature
            int nextToken;
            if (temperature <= 0.0001f) {
                // If temperature is close to 0, do argmax
                int maxIndex = 0;
                float maxValue = logits[0];
                for (int j = 1; j < logits.length; j++) {
                    if (logits[j] > maxValue) {
                        maxValue = logits[j];
                        maxIndex = j;
                    }
                }
                nextToken = maxIndex;
            } else {
                // Apply temperature and convert to probabilities
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (float logit : logits) {
                    if (logit > maxLogit) {
                        maxLogit = logit;
                    }
                }

                // Apply temperature and compute softmax
                float sum = 0.0f;
                float[] probs = new float[logits.length];
                for (int j = 0; j < logits.length; j++) {
                    // Divide by temperature
                    probs[j] = (float) Math.exp((logits[j] - maxLogit) / temperature);
                    sum += probs[j];
                }

                // Normalize to get probabilities
                for (int j = 0; j < probs.length; j++) {
                    probs[j] /= sum;
                }

                // Apply top-p sampling if enabled
                if (topP < 1.0) {
                    nextToken = sampleTopP(probs, topP, random);
                } else {
                    // Standard sampling from the full distribution
                    float sample = random.nextFloat();
                    float cumulativeProb = 0.0f;
                    nextToken = probs.length - 1; // default to last token

                    for (int j = 0; j < probs.length; j++) {
                        cumulativeProb += probs[j];
                        if (sample <= cumulativeProb) {
                            nextToken = j;
                            break;
                        }
                    }
                }
            }
            
            generated.add(nextToken);
            allTokens.add(nextToken); // Track for repeat penalty

            if (followEOS && nextToken == 50256) {
                if (sequentialPrint) {
                    // Print any pending tokens before ending
                    if (!pendingTokens.isEmpty()) {
                        int[] pendingArray = pendingTokens.stream().mapToInt(x -> x).toArray();
                        String pendingText = tokenizer.decode(pendingArray);
                        if (!containsReplacementChars(pendingText)) {
                            System.out.print(pendingText.replace("\n", "\r\n"));
                        }
                    }
                    System.out.println();
                }

                // Decode generated token IDs to text
                int[] outputIds = generated.stream().mapToInt(x -> x).toArray();
                return tokenizer.decode(outputIds);
            }

            if (sequentialPrint) {
                // Add token to pending list
                pendingTokens.add(nextToken);
                
                // Try to decode pending tokens
                int[] pendingArray = pendingTokens.stream().mapToInt(x -> x).toArray();
                String decodedText = tokenizer.decode(pendingArray);
                
                // If the decoded text doesn't contain replacement characters, print it
                if (!containsReplacementChars(decodedText)) {
                    System.out.print(decodedText.replace("\n", "\r\n"));
                    pendingTokens.clear(); // Clear the buffer after printing
                }
                // If it still contains replacement chars, keep accumulating tokens
            }
            
            // Append nextToken to context
            context = appendToken(context, nextToken);
        }
        
        if (sequentialPrint) {
            // Print any remaining pending tokens at the end
            if (!pendingTokens.isEmpty()) {
                int[] pendingArray = pendingTokens.stream().mapToInt(x -> x).toArray();
                String pendingText = tokenizer.decode(pendingArray);
                if (!containsReplacementChars(pendingText)) {
                    System.out.print(pendingText.replace("\n", "\r\n"));
                }
            }
            System.out.print("\r\n");
        }

        // Decode generated token IDs to text
        int[] outputIds = generated.stream().mapToInt(x -> x).toArray();
        return tokenizer.decode(outputIds);
    }

    /**
     * Perform top-p (nucleus) sampling on a probability distribution.
     * @param probs The probability distribution (must be normalized).
     * @param topP The top-p threshold (between 0.0 and 1.0).
     * @param random The random number generator.
     * @return The sampled token index.
     */
    private static int sampleTopP(float[] probs, double topP, Random random) {
        // Create array of (probability, index) pairs
        ProbabilityIndex[] probIndexes = new ProbabilityIndex[probs.length];
        for (int i = 0; i < probs.length; i++) {
            probIndexes[i] = new ProbabilityIndex(probs[i], i);
        }
        
        // Sort by probability in descending order
        Arrays.sort(probIndexes, (a, b) -> Float.compare(b.probability, a.probability));
        
        // Find the cutoff point for top-p
        float cumulativeProb = 0.0f;
        int cutoff = probIndexes.length;
        
        for (int i = 0; i < probIndexes.length; i++) {
            cumulativeProb += probIndexes[i].probability;
            if (cumulativeProb >= topP) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Renormalize the selected tokens
        float selectedSum = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            selectedSum += probIndexes[i].probability;
        }
        
        // Sample from the renormalized distribution
        float sample = random.nextFloat() * selectedSum;
        cumulativeProb = 0.0f;
        
        for (int i = 0; i < cutoff; i++) {
            cumulativeProb += probIndexes[i].probability;
            if (sample <= cumulativeProb) {
                return probIndexes[i].index;
            }
        }
        
        // Fallback (should rarely happen)
        return probIndexes[cutoff - 1].index;
    }
    
    /**
     * Apply repeat penalty to logits to discourage repetition.
     * @param logits The original logits array.
     * @param allTokens All tokens seen so far (input + generated).
     * @param repeatPenalty The penalty factor (> 1.0 discourages repetition).
     * @param repeatPenaltyRange Number of recent tokens to consider (0 = all tokens).
     * @return Modified logits with repeat penalty applied.
     */
    private static float[] applyRepeatPenalty(float[] logits, List<Integer> allTokens, 
                                            double repeatPenalty, int repeatPenaltyRange) {
        if (allTokens.isEmpty() || repeatPenalty == 1.0) {
            return logits;
        }
        
        float[] penalizedLogits = Arrays.copyOf(logits, logits.length);
        
        // Determine which tokens to consider for penalty
        int startIndex = 0;
        if (repeatPenaltyRange > 0 && allTokens.size() > repeatPenaltyRange) {
            startIndex = allTokens.size() - repeatPenaltyRange;
        }
        
        // Apply penalty to tokens that have appeared recently
        Set<Integer> seenTokens = new HashSet<>();
        for (int i = startIndex; i < allTokens.size(); i++) {
            int tokenId = allTokens.get(i);
            if (tokenId >= 0 && tokenId < logits.length) {
                seenTokens.add(tokenId);
            }
        }
        
        // Apply the penalty
        for (int tokenId : seenTokens) {
            if (penalizedLogits[tokenId] > 0) {
                // If logit is positive, divide by penalty (reduces probability)
                penalizedLogits[tokenId] = (float) (penalizedLogits[tokenId] / repeatPenalty);
            } else {
                // If logit is negative, multiply by penalty (makes it more negative)
                penalizedLogits[tokenId] = (float) (penalizedLogits[tokenId] * repeatPenalty);
            }
        }
        
        return penalizedLogits;
    }
    
    /**
     * Helper class for top-p sampling.
     */
    private static class ProbabilityIndex {
        final float probability;
        final int index;
        
        ProbabilityIndex(float probability, int index) {
            this.probability = probability;
            this.index = index;
        }
    }

    /**
     * Check if a string contains Unicode replacement characters that indicate
     * incomplete or invalid UTF-8 sequences.
     */
    private static boolean containsReplacementChars(String text) {
        return text.contains("�");
    }

    private static JMatrix trimContext(JMatrix context, int maxSeqLen) {
        // Copy last maxSeqLen tokens from context to window
        return context.slice(0, maxSeqLen).reshape(1, maxSeqLen, 1, 1);
    }

    private static float[] extractLogits(JMatrix output) {
        float[] logits = new float[output.shape(2)];
        for (int j = 0; j < logits.length; j++) {
            logits[j] = output.get(0, output.shape(1) - 1, j, 0);
        }
        return logits;
    }

    private static JMatrix appendToken(JMatrix context, int token) {
        JMatrix extended = JMatrix.zeros(1, context.size() + 1, 1, 1);
        for (int j = 0; j < context.size(); j++) {
            extended.set(0, j, 0, 0, context.get(0, j, 0, 0));
        }
        extended.set(0, extended.size() - 1, 0, 0, token);

        return extended;
    }
}