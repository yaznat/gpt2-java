package util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import jflow.data.JMatrix;

/**
 * Smart data loader to fine-tune models easily from a directory of .txt files. <p> 
 */
public class FineTuneLoader {
    private GPT2Tokenizer tokenizer;

    private List<Path> filePaths;

    public FineTuneLoader(String pathToDirectory, GPT2Tokenizer tokenizer) {
        Path dir = Paths.get(pathToDirectory);

        try {
            filePaths = Files.list(dir)
                                  .collect(Collectors.toList());
        } catch (IOException e) {
            System.err.println("Error listing directory: " + e.getMessage());
            filePaths = List.of();
        }

        this.tokenizer = tokenizer;
    }

    public void shuffle(long seed) {
        Collections.shuffle(filePaths, new Random(seed));
    }

    /**
     * Retrieve a training example in the shape (1, length, 1, 1).
     * @param index Example index 
     */
    public JMatrix get(int index) {
        Path currentPath = filePaths.get(index);

        String text;
        try {
            text = Files.readString(currentPath);
        } catch (IOException e) {
            System.err.println(
                "Error reading " + currentPath.toString()
                 +  ": " + e.getMessage()
            );
            text = "";
        }

        try {
            return tokenizer.encode(text);
        } catch (IOException e) {
            System.err.println(
                "Error during tokenization: " + e.getMessage()
            );
            return null;
        }
    }

    /**
     * The total number of examples in the directory.
     */
    public int numExamples() {
        return filePaths.size();
    }
}
