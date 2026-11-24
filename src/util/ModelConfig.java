package util;

public enum ModelConfig {
    // 124M configuration
    SMALL(50257, 768, 1024, 12, 0.0, 12, "gpt2_124M"),
    // 355M configuration
    MEDIUM(50257, 1024, 1024, 16, 0.1, 24, "gpt2_355M"),
    // 774M configuration
    LARGE(50257, 1280, 1024, 20, 0.1, 36, "gpt2_774M"),
    // 1.5B configuration
    XL(50257, 1600, 1024, 25, 0.1, 48, "gpt2_1.5B");

    private final int vocabSize;
    private final int embedDim;
    private final int maxSeqLen;
    private final int numHeads;
    private final double dropoutProb;
    private final int numLayers;
    private final String name;

    ModelConfig(int vocabSize, int embedDim, int maxSeqLen, int numHeads, double dropoutProb, int numLayers, String name) {
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;
        this.numHeads = numHeads;
        this.dropoutProb = dropoutProb;
        this.numLayers = numLayers;
        this.name = name;
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public int getEmbedDim() {
        return embedDim;
    }

    public int getMaxSeqLen() {
        return maxSeqLen;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public double getDropoutProb() {
        return dropoutProb;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public String getName() {
        return name;
    }
}

