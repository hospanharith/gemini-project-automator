# Gemini-Powered Project Generator & Modifier

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/hospanharith/gemini-project-automator)

## The Concept: AI-Driven Development Exploration

This project explores the use of large language models, specifically **Google's Gemini family (tested with 2.5 Pro)**, for autonomous software development tasks. It provides a Python script (`project_generator.py`) capable of:

1.  **Generating New Projects:** Takes a natural language description and iteratively generates a complete project structure, including source code, documentation (README, Design, Requirements, etc.), configuration files, tests (stubs/examples), and development environment setup (like `.devcontainer`).
2.  **Modifying Existing Projects:** Leverages **Retrieval-Augmented Generation (RAG)**. It indexes an existing project's codebase into a local vector store (ChromaDB) and uses retrieved code snippets as context for Gemini to apply requested changes (modifications, additions, deletions) across the relevant files, including tests and documentation.

**Expectation Management: The Nature of AI-Generated Concepts**

> **This is a concept demonstration, largely generated *with* the assistance of Google's Gemini 2.5 Pro and refined for usability.** It showcases the potential of current AI models but is **not intended as perfect, production-ready code.**
>
> The field of AI is evolving at an unprecedented pace. Tools and capabilities that seem advanced today might be commonplace or even superseded tomorrow. This project serves as a snapshot, an idea of how development workflows *might* change.
>
> It's crucial to understand that everything – not just software development, but potentially *all* knowledge work and creative processes – is on the cusp of significant transformation driven by AI. This rapid evolution necessitates continuous learning, adaptation, and a readiness to embrace new paradigms. This tool, while functional for its purpose, is a stepping stone, illustrating that we must prepare for a future where AI collaboration is integral to how we work and create. Expect imperfections, but appreciate the direction.

## Features

*   **Iterative Project Generation:** Create boilerplate and initial structure for various project types based on a description.
*   **RAG-Based Modification:** Apply targeted changes to existing codebases using relevant context.
*   **Comprehensive Output:** Aims to generate code, tests, documentation (`README.md`, `docs/DESIGN.md`, `docs/REQUIREMENTS.md`, `CHANGELOG.md`), config (`requirements.txt`, `.gitignore`, etc.), and dev environment files (`.devcontainer/devcontainer.json`).
*   **Vector Store Management:** Creates/loads a ChromaDB vector store (`.project_vector_store`) within the target project for RAG.
*   **Interactive CLI:** Simple menu for choosing generation, modification, or indexing.

## Technology Stack

*   **Python 3.9+**
*   **Google Generative AI API:** Access to Gemini models (e.g., `gemini-1.5-pro-latest`).
*   **LangChain:** Framework for building LLM applications, used for the RAG pipeline.
    *   `langchain-google-genai`: Google integrations.
    *   `langchain-text-splitters`: Document chunking.
    *   `langchain-core`: Core abstractions.
*   **ChromaDB:** Local vector store for RAG context.
*   **pysqlite3-binary:** Required dependency for ChromaDB in certain environments (like Codespaces).
*   **python-dotenv:** For managing API keys via `.env` files.

## Setup

### Prerequisites

*   Python 3.9 or later
*   Git
*   A Google API Key with access to the Gemini models (e.g., Gemini 1.5 Pro). You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 1. Clone the Repository

```bash
git clone https://github.com/hospanharith/gemini-project-automator.git
cd gemini-project-automator
```

### 2. Set Up Google API Key

The script needs your Google API Key to interact with Gemini. It looks for the key in the following order:

1.  **Environment Variable (Recommended for Codespaces/CI/CD):**
    Set the `GOOGLE_API_KEY` environment variable.
    ```bash
    export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
    ```
    *(On Windows, use `set GOOGLE_API_KEY=YOUR_API_KEY_HERE` or use system environment variable settings)*

2.  **.env File (Recommended for Local Development):**
    *   Create a file named `.env` in the root of the cloned repository.
    *   Add the following line to the `.env` file:
        ```
        GOOGLE_API_KEY=YOUR_API_KEY_HERE
        ```
    *   *(The `.gitignore` file included in this repository prevents accidentally committing your `.env` file)*

3.  **Interactive Prompt:** If the key is not found via the methods above, the script will securely prompt you to enter it when run.

### 3. Install Dependencies

It's highly recommended to use a virtual environment:

```bash
# Create a virtual environment (replace '.venv' if you prefer a different name)
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (Git Bash):
source .venv/Scripts/activate
# On Windows (Command Prompt):
.venv\\Scripts\\activate.bat
# On Windows (PowerShell):
.venv\\Scripts\\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

## Running the Script

Once set up, run the script from your activated virtual environment:

```bash
python project_generator.py
```

You will be presented with an interactive menu:

```
Choose an action:
  1. Generate a new project
  2. Modify an existing project
  3. Re-index an existing project's Vector Store
  Q. Quit
>
```

*   **Generate:** Prompts for a project description and a *target directory path* for the new project. It then interacts with Gemini iteratively to generate files.
*   **Modify:** Prompts for the *path to the existing project directory* and a description of the changes. It indexes the project (or loads the index), retrieves relevant code snippets, and asks Gemini to generate all necessary changes in one go.
*   **Index:** Prompts for the *path to an existing project directory* and forces a re-indexing of its vector store. Useful if you've made external changes and want to update the RAG context.
*   **Quit:** Exits the script.

## Testing the Code (Easy Methods)

### 1. GitHub Codespaces (Recommended)

This is the easiest way to get started without installing anything locally.

1.  **Fork this Repository:** (Optional, but good practice).
2.  **Create Codespace:** Go to the main page of your forked repository (or this one) on GitHub. Click the `<> Code` button, go to the \"Codespaces\" tab, and click \"Create codespace on main\".
3.  **Wait for Build:** Codespaces will automatically set up the environment based on the `.devcontainer/devcontainer.json` file, including installing all Python dependencies from `requirements.txt`. This might take a few minutes the first time.
4.  **Configure API Key Secret:**
    *   While the Codespace is building or after it's ready, go back to your repository settings on GitHub.
    *   Navigate to `Settings` -> `Secrets and variables` -> `Codespaces`.
    *   Click `New repository secret`.
    *   Name the secret exactly `GOOGLE_API_KEY`.
    *   Paste your actual Google API key into the `Value` field.
    *   Click `Add secret`. The Codespace will automatically pick up this environment variable. You might need to restart the Codespace terminal if you add the secret after it's fully started.
5.  **Run the Script:** Once the Codespace is ready, a VS Code interface will open in your browser. Open a terminal (Ctrl+` or `Terminal` > `New Terminal`) and run:
    ```bash
    python project_generator.py
    ```
6.  **Follow Prompts:** Interact with the script's menu.
    *   When generating or modifying, you'll be asked for a path. You can use relative paths like `./my_new_project` or `./existing_project_folder`.
    *   Generated/modified files will appear in the file explorer panel within the Codespace.

### 2. Local Machine

1.  **Follow Setup:** Ensure you have completed all steps in the [Setup](#setup) section (cloning, API key, installing dependencies in a virtual environment).
2.  **Activate Environment:** Make sure your virtual environment is activated.
3.  **Run Script:** Open your terminal/command prompt, navigate to the cloned repository directory, and run:
    ```bash
    python project_generator.py
    ```
4.  **Follow Prompts:** Interact with the menu. Provide paths relative to your current location or absolute paths.

## How It Works Briefly

*   **Generation (Iterative):**
    *   Sends a detailed system prompt (`SETUP_PROMPT_GENERATE`) to Gemini instructing it to act like a full development team and generate files iteratively.
    *   Sends the user's project description.
    *   Receives chunks of file content formatted with `FILENAME:` headers.
    *   Parses the response using regex (`ACTION_BLOCK_REGEX`) and writes files to the specified project path.
    *   Sends `next` to get the next chunk until Gemini sends the completion signal (`=== ALL FILES GENERATED ===`).
*   **Modification (RAG):**
    *   **Indexing:** Scans the target project directory for source/doc/config files, excluding ignores (`.git`, `node_modules`, etc.). Uses `TextLoader` and `RecursiveCharacterTextSplitter` to load and chunk documents. `GoogleGenerativeAIEmbeddings` converts chunks to vectors. `Chroma.from_documents` stores vectors and text in a local directory (`.project_vector_store` inside the target project).
    *   **Retrieval:** When a modification request is made, the `Chroma` vector store is used as a retriever to find the `k` most relevant text chunks based on the request's semantic similarity.
    *   **Generation:** A specific prompt (`SETUP_PROMPT_MODIFY_RAG`) is constructed, including the user's request and the formatted content of the retrieved chunks (`format_docs_for_prompt`).
    *   **LangChain Chain:** An LCEL chain orchestrates the process: `RunnableParallel` fetches context and passes the question -> `PromptTemplate` formats the input -> `ChatGoogleGenerativeAI` (Gemini) generates the response -> `StrOutputParser` extracts the text.
    *   **Parsing & Application:** The *entire* LLM response is parsed using the same regex (`ACTION_BLOCK_REGEX`) to find `FILENAME:` (write/update) and `ACTION: DELETE` blocks. `parse_and_apply_changes` performs the file operations securely within the target project directory. The process expects a final completion signal (`=== ALL CHANGES GENERATED ===`).

## The Vector Store (`.project_vector_store`)

*   When you run the \"Modify\" or \"Index\" option, a subdirectory named `.project_vector_store` is created *inside* the target project directory you specify.
*   This folder contains the ChromaDB database files (like SQLite and Parquet files) storing the text chunks and their vector representations.
*   This store is loaded automatically on subsequent \"Modify\" runs unless you choose to re-index.
*   You can safely delete this directory; it will be regenerated the next time you index or modify that project (though indexing can take time).
*   The `.gitignore` file in *this* repository ignores this directory name, so you typically wouldn't commit it for *other* projects unless you have a specific reason.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/hospanharith/gemini-project-automator/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
