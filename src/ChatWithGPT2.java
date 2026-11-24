import java.io.IOException;
import java.util.Scanner;

import jflow.model.Sequential;
import jflow.utils.AnsiCodes;
import util.GPT2Tokenizer;
import util.ModelConfig;
import util.TextGenUtils;
import gpt2.GPT2Builder;

/**
 * Chat with a GPT-2 124M model. Available by default as a demo of instruction tuned GPT-2 124M.
 * Provide your own trained weights and disable {@code useDemoFormat} to chat with a custom model.
 * 
 * <p><b>REQUIRED: </b> see README.md for installation instructions: https://github.com/yaznat/gpt2-java
 */
public class ChatWithGPT2 {
    // ================ CUSTOMIZABLE CONFIGURATION ================
    private static final ModelConfig GPT2_VERSION = ModelConfig.SMALL; // 124M param version

    private static final String PATH_TO_WEIGHTS = "saved_models/gpt2_124M/fine_tuned_final";

    // Disable this to chat with your own version of GPT-2 in a default format
    private static final boolean USE_DEMO_FORMAT = true; 

    // Increase this for more random responses
    private static final float TEMPERATURE = 0.5f;

    private static final float TOP_P = 0.9f;
    private static final float REPEAT_PENALTY = 1.2f;
    private static final int REPEAT_PENALTY_RANGE = 128;

    // Change this based on task and model capability
    private static final int MAX_SEQ_LEN = 512;

    private static void setTerminalRawMode() {
        try {
            // Raw mode (disables echo and canonical input)
            Runtime.getRuntime().exec(new String[]{"/bin/sh", "-c", "stty raw -echo < /dev/tty"}).waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Restore on exit
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                Runtime.getRuntime().exec(new String[]{"/bin/sh", "-c", "stty sane < /dev/tty"}).waitFor();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }));
    }

    private static void restoreTerminal() {
        try {
            Runtime.getRuntime().exec(new String[]{"/bin/sh", "-c", "stty sane < /dev/tty"}).waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void printUserTag() {
        System.out.print(
            AnsiCodes.YELLOW + AnsiCodes.BOLD 
            + "[user]: " + AnsiCodes.RESET + 
            AnsiCodes.WHITE
        );
    }

    private static void printModelTag() {
        System.out.print(
            AnsiCodes.BLUE + AnsiCodes.BOLD 
            + "[GPT-2]: " + AnsiCodes.RESET + 
            AnsiCodes.WHITE
        );
    }
    
    

    public static void main(String[] args) {
        // ================ LOAD COMPONENTS ================    

        @SuppressWarnings("resource")
        Scanner scanner = new Scanner(System.in);
        GPT2Tokenizer tokenizer = new GPT2Tokenizer();
        try {
            System.out.println(tokenizer.decode(new int[]{0}));
        } catch (IOException e) {
            e.printStackTrace();
        }

        Sequential model = GPT2Builder.create(GPT2_VERSION);

        model.loadWeights(PATH_TO_WEIGHTS);


        // ================ PRINT HEADER AND INSTRUCTIONS ================    

        // Print chat header
        System.out.print("\r\n" +
            AnsiCodes.BLUE + "=".repeat(20) +
            AnsiCodes.BOLD + " Chat with GPT-2 " + 
            AnsiCodes.RESET + AnsiCodes.BLUE + 
            "=".repeat(20) + AnsiCodes.RESET
        );

        if (USE_DEMO_FORMAT) {
            setTerminalRawMode();
            // Explain demo
            String instructions = 
                "\r\n" + AnsiCodes.YELLOW + AnsiCodes.BOLD + 
                "Welcome to the GPT-2 124M instruction-tuned demo!"
                + AnsiCodes.RESET + AnsiCodes.WHITE
                + "\r\nThe model is trained to follow simple instructions."
                + AnsiCodes.BLUE + AnsiCodes.BOLD + "\r\nCurrently available: " 
                + AnsiCodes.RESET + AnsiCodes.GREEN + 
                "\r\n - Fix Grammar" + 
                "\r\n - Generate Recipes" + 
                AnsiCodes.WHITE + "\r\nTry it out with an instruction!" + 
                "\r\ne.g.: " + AnsiCodes.BLUE + 
                "\r\n - \"Fix the grammar in this sentence: I gone to the mall.\"" +
                "\r\n - \"Give me a recipe for muffins.\" " + AnsiCodes.WHITE +
                "\r\nNote: Recipe generation demonstrates a complex understanding of related topics, " + 
                "but responses are often imperfect."
                + AnsiCodes.RESET;

            // Print one word at a time
            for (String word : instructions.split(" ")) {
                System.out.print(word + " ");
                try {
                    Thread.sleep(100);
                } catch (Exception e) {}
            }
            restoreTerminal();
            System.out.println();
        }
        System.out.println();


        // ================ BEGIN CHAT ================    

        // Main loop
        while (true) {
            // Take user input
            printUserTag();
            String prompt = scanner.nextLine() + "<|endoftext|>";

            // Prepare model response and lock terminal
            printModelTag();
            setTerminalRawMode();

            // Generate response
            long seed = System.nanoTime();
            try {
                TextGenUtils.generate(
                    model, 
                    tokenizer, 
                    prompt, 
                    MAX_SEQ_LEN, 
                    MAX_SEQ_LEN, 
                    TEMPERATURE, 
                    TOP_P,
                    REPEAT_PENALTY,
                    REPEAT_PENALTY_RANGE,
                    seed, 
                    true, 
                    true
                );
            } catch (Exception e) {
                System.out.println(
                    AnsiCodes.RED + 
                    "Error in model response: "
                    + e.getClass().getSimpleName() 
                    + AnsiCodes.RESET
                );
            }

            // Unlock terminal
            restoreTerminal();
            System.out.println();
        }
    }   
}