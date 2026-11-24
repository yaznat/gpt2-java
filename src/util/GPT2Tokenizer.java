package util;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

import jflow.data.JMatrix;

/**
 * A tokenizer for GPT-2, using a python process to run Tiktoken.
 * To run this class, you must access a python virtual environment with Tiktoken installed.
 */
public class GPT2Tokenizer {
    private Process pythonProcess;
    private BufferedWriter toPython;
    private BufferedReader fromPython;

    public GPT2Tokenizer() {
        pythonProcess = PythonProcessBuilder.buildProcess("src/scripts/tokenizer_script.py");

        toPython = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream()));
        fromPython = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
    }

    /**
     * Encode text into a JMatrix with specified shape.
     * @param text Text to encode.
     * @param batchSize The batch size of the encoded matrix.
     * @param seqLen The sequence length of the encoded matrix.
     * @return A JMatrix with shape: (batchSize, seqLen, 1, 1) filled with token values.
     * @throws IllegalArgumentException If not enough text is provided for the specified shape.
     * @throws IOException If an error occurs in communication with python.
     */
    public JMatrix encode(String text, int batchSize, int seqLen) throws IOException {
        // Send text to python
        String encoded = Base64.getEncoder().encodeToString(text.getBytes(StandardCharsets.UTF_8));
        toPython.write(encoded);
        toPython.newLine();
        toPython.flush();
    
        // Receive tokens in a String, split by " "
        String line = fromPython.readLine();
        if (line == null) {
            throw new IOException("Python process terminated unexpectedly");
        }
    
        // Create a JMatrix with the specified shape
        JMatrix tokens = JMatrix.zeros(batchSize, seqLen, 1, 1);
        int totalTokens = batchSize * seqLen;
        int index = 0;
    
        // Load tokens into the JMatrix
        for (String part : line.trim().split(" ")) {
            if (!part.isEmpty()) {
                int token = Integer.parseInt(part);
                
                // if (contains(tokensToSkip, token)) continue;
                tokens.set(index++, token);
                if (index >= totalTokens) break;
            }
        }
    
        // Throw an error if not enough text is provided for the specified shape
        if (index < totalTokens) {
            throw new IllegalArgumentException("String length too small for provided dimensions.");
        }
    
        return tokens;
    }

    /**
     * Encode text into a JMatrix.
     * @param text Text to encode.
     * @return A JMatrix with shape: (1, X, 1, 1), where X is the resulting number of tokens.
     * @throws IOException If an error occurs in communication with python.
     */
    public JMatrix encode(String text) throws IOException {
        // Send text to python
        String encoded = Base64.getEncoder().encodeToString(text.getBytes(StandardCharsets.UTF_8));
        toPython.write(encoded);
        toPython.newLine();
        toPython.flush();
    
        // Receive tokens in a String, split by " "
        String line = fromPython.readLine();
        if (line == null) {
            throw new IOException("Python process terminated unexpectedly");
        }
    
        // Count tokens, excluding those to skip
        int count = 0;
        for (String part : line.trim().split(" ")) {
            if (!part.isEmpty()) {
                count++;
            }
        }
    
        // Load tokens into a JMatrix
        JMatrix sequence = JMatrix.zeros(1, count, 1, 1);
        int index = 0;
        for (String part : line.trim().split(" ")) {
            if (!part.isEmpty()) {
                int token = Integer.parseInt(part);
                // if (contains(tokensToSkip, token)) continue;
                sequence.set(index++, token);
            }
        }
    
        return sequence;
    }
    
    /**
     * Decode text from a JMatrix.
     * @param tokens Token values stored in a JMatrix
     * @return The decoded text as a String.
     * @throws IOException If an error occurs in communication with python.
     */
    public String decode(JMatrix tokens) throws IOException {
        StringBuilder cmd = new StringBuilder("DETOKENIZE");
        int size = tokens.size();
        for (int i = 0; i < size; i++) {
            cmd.append(" ").append((int) tokens.get(i));
        }
    
        toPython.write(cmd.toString());
        toPython.newLine();
        toPython.flush();
    
        return readMultilineResponse();
    }
    
    /**
     * Decode text from an array.
     * @param tokens Token values stored in an array.
     * @return The decoded text as a String.
     * @throws IOException If an error occurs in communication with python.
     */
    public String decode(int[] tokens) throws IOException {
        StringBuilder cmd = new StringBuilder("DETOKENIZE");
        for (int token : tokens) {
            cmd.append(" ").append(token);
        }
    
        toPython.write(cmd.toString());
        toPython.newLine();
        toPython.flush();
    
        return readMultilineResponse();
    }

    /**
     * Terminate the python process and close the BufferedReader and BufferedWriter.
     * @throws IOException If the components fail to close.
     */
    public void close() throws IOException {
        toPython.close();
        fromPython.close();
        pythonProcess.destroy();
    }
    
    private String readMultilineResponse() throws IOException {
        StringBuilder sb = new StringBuilder();
        String line;
        boolean firstLine = true;
        while ((line = fromPython.readLine()) != null) {
            if (line.equals("__END__")) break;
            if (!firstLine) sb.append("\n");  // Only insert \n between lines
            sb.append(line);
            firstLine = false;
        }
        return sb.toString(); 
    }
}