package util;

import java.io.IOException;

import jflow.utils.AnsiCodes;

class PythonProcessBuilder {

    protected static Process buildProcess(String pathToScript) {
        Process pythonProcess = null;
        // Try a simple build first
        ProcessBuilder pb = new ProcessBuilder("python", pathToScript);
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        try {
            pythonProcess = pb.start();
        } catch (IOException e) {
            // Needs user interaction to fix
            String pythonPath = System.getenv("PYTHON_PATH");

            if (pythonPath == null) {
                throw new IllegalStateException(
                    AnsiCodes.RED + AnsiCodes.BOLD + 
                    "Automatic Python build failed. " + 
                    "If you haven't activated a Python virtual environment, see README.md for setup: " + 
                    "https://github.com/yaznat/gpt2-java. " + 
                    "If you've already done setup, run this command in the terminal for a manual fix:\n" + 
                    "export PYTHON_PATH=$(python -c \"import sys; print(sys.executable)\")" 
                    + AnsiCodes.RESET
                );
            }

            // Try again now that the user has run the command
            ProcessBuilder secondTry = new ProcessBuilder(pythonPath, pathToScript);
            secondTry.redirectError(ProcessBuilder.Redirect.INHERIT);

            try {
                pythonProcess = secondTry.start();
            } catch (IOException e2) {
                throw new IllegalStateException(
                    AnsiCodes.RED + AnsiCodes.BOLD + 
                    "Python process failed again. Ensure this is the correct path to your python environment: "
                    + pythonPath +"\nYou may need to troubleshoot your Python installation or environment."
                    + AnsiCodes.RESET
                );
            }
        }

        return pythonProcess;
    }
}
