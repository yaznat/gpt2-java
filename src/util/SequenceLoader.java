package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import jflow.data.JMatrix;
import jflow.utils.AnsiCodes;

/**
 * Smart data loader to train models easily from a large text file. <p> 
 * Converts file to preprocessed binary tokens for faster training <p>
 * Tracks datasets and configurations to disk -- allowing you to switch between them without issue. <p>
 * Tracks position in dataset and batches completed: call {@code numBatchesCompleted()} <p>
 * Call {@code next()} for the next batch. <p>
 */
public class SequenceLoader {
    private final int batchSize;
    private final int seqLen;
    private int batchesCompleted = 0;

    private final static String ASSETS_DIR = "src/assets/";
    private final static String PROGRESS_TRACKER_DIR = ASSETS_DIR +  "batch-trackers/";
    private final static String LOG_PATH = ASSETS_DIR + "dataset-log";
    private final static String PREPROCESSED_DIR = ASSETS_DIR + "preprocessed-datasets/";
   
    private final static String SPLIT_STR = "#####"; 

    private final static String PYTHON_PREPROCESSOR_PATH = "src/scripts/preprocessor_script.py";

    private final String PROGRESS_TRACKER_PATH;
    private final String PREPROCESSED_PATH;

    private List<String> batchFileNames;

    /**
     * Create a new SequenceLoader for a specific dataset and configuration. <p>
     * If the configuration is new, loads and preprocesses the dataset. 
     * Otherwise, resumes training from the last checkpoint
     */
    public SequenceLoader(String pathToFile, int batchSize, int seqLen, int stride) {
        this.batchSize = batchSize;
        this.seqLen = seqLen;

        File assetDir = new File(ASSETS_DIR);
        if (!assetDir.exists()) {
            assetDir.mkdir();
        }

        File trackerDir = new File(PROGRESS_TRACKER_DIR);
        if (!trackerDir.exists()) {
            trackerDir.mkdir();
        }

        Path path = Paths.get(pathToFile);

        String fileName = path.getFileName().toString();

        String configuration = fileName + SPLIT_STR + batchSize + 
            SPLIT_STR + seqLen + SPLIT_STR + stride;

        // Check if this dataset configuration has already been initialized and trained on
        if (isLogged(configuration)) {
            // Load values
            int configurationNumber = getConfigurationNumber(configuration);

            PROGRESS_TRACKER_PATH = PROGRESS_TRACKER_DIR + configurationNumber + ".txt";
            PREPROCESSED_PATH = PREPROCESSED_DIR + configurationNumber;
            
            batchesCompleted = readBatchesCompleted();

        } else {
            // Log this configuration
            int configurationNumber = log(configuration);

            // Create progress tracker
            PROGRESS_TRACKER_PATH = PROGRESS_TRACKER_DIR + configurationNumber + ".txt";
            createFile(PROGRESS_TRACKER_PATH);
            trackBatch(); // Populate the tracker with "0"

            // Preprocess dataset
            PREPROCESSED_PATH = PREPROCESSED_DIR + configurationNumber;
            System.out.println( "\n" + AnsiCodes.WHITE + 
                "Preprocessing dataset to binary for maximum performance during training. " + 
                "This process will only occur once per dataset + batch/seq/stride configuration. " +
                AnsiCodes.RED + AnsiCodes.BOLD + "Do not exit while this process is running."
                + AnsiCodes.RESET
            );

            // Run preprocessing directly in python for faster performance
            try {
                runPreprocessing(
                    PYTHON_PREPROCESSOR_PATH,
                    pathToFile,
                    PREPROCESSED_PATH,
                    batchSize,
                    seqLen,
                    stride
                );
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        }
        // load preprocessed batch files into this.batchFileNames
        loadDatasetPaths();
    }

    /**
     * Retrieve a batch of tokens in the shape (batchSize, seqLen, 1, 1)
     */
    public JMatrix next() {
        JMatrix batch = JMatrix.zeros(batchSize, seqLen, 1, 1);

        String path = PREPROCESSED_PATH + "/" + batchFileNames.get(batchesCompleted) + ".bin";

        File file = new File(path);
        int fileSize = (int) file.length();
        int numInts = fileSize / 4;

        try (FileInputStream fis = new FileInputStream(file);
             FileChannel channel = fis.getChannel()) {

            ByteBuffer buffer = ByteBuffer.allocate(fileSize);
            buffer.order(ByteOrder.LITTLE_ENDIAN); // Match Python struct '<I'
            channel.read(buffer);
            buffer.flip();

            for (int i = 0; i < numInts; i++) {
                batch.set(i, buffer.getInt());
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        batchesCompleted++;
        trackBatch();

        return batch;
    }

    /**
     * @return True while enough data remains for another batch.
     */
    public boolean hasNext() {
        return batchesCompleted < batchFileNames.size();
    }
    
    /**
     * @return The number of batches completed (based on calls to {@code next()})
     */
    public int numBatchesCompleted() {
        return batchesCompleted;
    }

    public int totalNumBatches() {
        return batchFileNames.size();
    }

    // Write batchesCompleted to the progress tracker
    private void trackBatch() {
        writeToFile(PROGRESS_TRACKER_PATH, String.valueOf(batchesCompleted), false);
    }

    private int readBatchesCompleted() {
        try (BufferedReader br = new BufferedReader(
            new FileReader(PROGRESS_TRACKER_PATH))) {
            return Integer.parseInt(br.readLine());
        } catch (IOException e) {
            e.printStackTrace();
            return -1;
        }
    }
    

    private boolean isLogged(String configuration) {
        // Ensure log file exists
        File logFile = new File(LOG_PATH);
        if (!logFile.exists() && !createFile(LOG_PATH)) {
            throw new UnknownError(
                "Unable to create file: " + LOG_PATH + 
                " Please troubleshoot for your device."
            );
        }

        // Check the log file for the provided configuration
        try (BufferedReader br = new BufferedReader(
            new FileReader(LOG_PATH))) {
            String line;
            while((line = br.readLine()) != null){
                if (line.equals(configuration)) {
                    return true;
                }
            }          
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }

    // Returns this configuration's index in the log
    private int log(String configuration) {
        writeToFile(LOG_PATH, configuration + "\n", true);
        
        try (Stream<String> lines = Files.lines(Paths.get(LOG_PATH))) {
            return (int)lines.count() - 1;
        } catch (IOException e) {
            e.printStackTrace();
            return -1;
        }
    }

    private int getConfigurationNumber(String configuration) {
        int lineIndex = 0;

        // Search the log file for the provided file
        try (BufferedReader br = new BufferedReader(
            new FileReader(LOG_PATH))) {
            String line;
            while((line = br.readLine()) != null){
                if (line.equals(configuration)) {
                    return lineIndex;
                }
                lineIndex++;
            }          
        } catch (IOException e) {
            e.printStackTrace();
        }
        return -1;
    }

    private void loadDatasetPaths() {
        try {
            // Sort file paths numerically to maintain sequential order
            Path dir = Paths.get(PREPROCESSED_PATH);
            List<Path> paths = Files.list(dir)
                .filter(Files::isRegularFile)
                .sorted(Comparator.comparingInt(p -> {
                    String name = p.getFileName().toString();
                    return Integer.parseInt(name.replace(".bin", ""));
                }))
                .collect(Collectors.toList());

            batchFileNames = new ArrayList<String>();

            for (Path path : paths) {
                String fileName = path.getFileName().toString();
                batchFileNames.add(fileName.substring(0, fileName.length() - 4)); // strip ".bin"
            }

        } catch (IOException e) {
            System.err.println("Error listing directory: " + e.getMessage());
        }
    }

    public static void runPreprocessing(
        String scriptPath, 
        String inputPath, 
        String outputDir,
        int batchSize,
        int seqLen, 
        int stride
    ) throws IOException, InterruptedException {
        String pythonVar = System.getenv("PYTHON_PATH");
        String pythonPath = (pythonVar == null) ? "python" : pythonVar;

        List<String> command = new ArrayList<>();
        command.add(pythonPath);  
        command.add(scriptPath);     
        command.add("--input_path");
        command.add(inputPath);
        command.add("--output_dir");
        command.add(outputDir);
        command.add("--batch_size");
        command.add(String.valueOf(batchSize));
        command.add("--seq_len");
        command.add(String.valueOf(seqLen));
        command.add("--stride");
        command.add(String.valueOf(stride));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);

        Process process = pb.start();
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Preprocessing failed with exit code " + exitCode);
        }
    }


    private boolean createFile(String filePath) {
        Path path = Paths.get(filePath);
        try {
            Files.createFile(path);
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    private boolean writeToFile(String filePath, String line, boolean append) {
        BufferedWriter writer;
        try {
            writer = new BufferedWriter(
                new FileWriter(filePath, append)
            );
            writer.write(line);
            writer.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

}