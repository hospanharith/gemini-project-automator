# -*- coding: utf-8 -*-
"""
Project Generator and Modifier using Google Gemini and LangChain RAG.

This script allows users to either:
1. Generate a new software project from a description using an iterative approach
   with a Google Gemini model.
2. Modify an existing software project using a Retrieval-Augmented Generation (RAG)
   approach, leveraging a vector store (ChromaDB) of the project's code.

Designed conceptually with Gemini 2.5 Pro.
"""

# --- Standard Library Imports ---
import os
import sys
import re
import argparse
import shutil
import warnings
from pathlib import Path
from getpass import getpass
from typing import Optional, Tuple, List, Dict, Any

# --- Environment & API Key ---
# Attempt to load environment variables from .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("Loaded environment variables from .env file (if present).")
except ImportError:
    print("Optional: 'python-dotenv' not found. Skipping .env file loading.")
    print(
        "Consider installing it ('pip install python-dotenv') for easier API key management."
    )
    pass  # python-dotenv is optional

# --- Necessary Hack for ChromaDB on certain systems (like Codespaces) ---
# ChromaDB uses sqlite3, which might need a specific binary version.
# This hack replaces the standard sqlite3 module with pysqlite3 if installed.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    print("Successfully replaced sqlite3 with pysqlite3.")
except ImportError:
    warnings.warn(
        "pysqlite3 not found. ChromaDB might face issues on some systems "
        "(like standard Codespaces) without it. If you encounter errors, "
        "try 'pip install pysqlite3-binary'."
    )
    pass  # Continue even if pysqlite3 isn't available

# --- Third-Party Imports ---
try:
    import google.generativeai as genai
    from langchain_google_genai import (
        GoogleGenerativeAIEmbeddings,
        ChatGoogleGenerativeAI,
    )
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
    )  # Language enum is less critical here
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
    from langchain.schema.output_parser import StrOutputParser
    from langchain.docstore.document import Document
except ImportError:
    print("\n--- Error: Required Libraries Not Found ---")
    print("This tool requires several Python packages to function.")
    print("Please install them by running the following command:")
    print("\n  pip install -r requirements.txt\n")
    sys.exit(1)


# --- Constants ---

# --- API & Model Configuration ---
API_KEY_ENV_VAR: str = "GOOGLE_API_KEY"  # Standard environment variable name
LLM_NAME: str = (
    "gemini-2.5-pro-exp-03-25"  # Specify desired Gemini model (e.g., 1.5 Pro)
)
EMBEDDING_MODEL_NAME: str = "models/text-embedding-004"  # Google's embedding model

# --- RAG Configuration ---
VECTOR_STORE_SUBDIR: str = (
    ".project_vector_store"  # Subdirectory within the target project for ChromaDB
)
CHUNK_SIZE: int = 1500  # Size of text chunks for vectorization
CHUNK_OVERLAP: int = 200  # Overlap between chunks
RETRIEVER_K: int = 10  # Number of relevant chunks to retrieve for context

# --- File Parsing & Generation ---
# Placeholder used in prompts and expected in LLM output paths
PROJECT_DIR_PLACEHOLDER: str = "<PROJECT_ROOT>"
# Signal for completion in iterative generation mode (Non-RAG)
COMPLETION_SIGNAL_GENERATE: str = "=== ALL FILES GENERATED ==="
# Signal for completion in modification mode (RAG)
COMPLETION_SIGNAL_MODIFY: str = "=== ALL CHANGES GENERATED ==="
# Acknowledgment signal expected from LLM after receiving setup prompt
READY_SIGNAL: str = "READY"


# --- Prompts ---

# --- Prompt for Project Generation (Iterative, Non-RAG) ---
# This prompt guides the LLM to act as a full software team, generating
# all necessary files iteratively, including comprehensive documentation.
SETUP_PROMPT_GENERATE: str = f"""
**Your Role:** You are an integrated Autonomous Software Development System. You embody the expertise of a Senior Product Manager, Business Analyst, Lead Architect, Senior Software Developer, QA Engineer, and DevOps Engineer. Your primary function is to translate user requirements into a complete set of files for a runnable software project, simulating a comprehensive Software Development Lifecycle (SDLC) internally. You generate *only* the file contents, strictly adhering to the specified format for programmatic parsing.

**Core Objective:** Based on the user's project request (which will be provided in the next message), autonomously perform an internal SDLC simulation (requirements specification, system design, implementation planning, dev environment setup, testing strategy/cases, documentation strategy) and generate the content for *all* necessary project files, including **comprehensive documentation artifacts** typically produced in a standard software project.

**Internal Simulation Mandate (For Your Guidance Only - Do NOT include this thinking process in your output):**

1.  **Requirements Specification:** Deconstruct the user's request. **Formalize this into a Requirements Specification document structure,** detailing:
    *   **Functional Requirements:** What the system must do.
    *   **Non-Functional Requirements (NFRs):** Quality attributes like performance, security, usability, scalability, maintainability.
    *   **User Roles/Personas (if applicable).**
    *   **Assumptions and Constraints.**
    *   **Define clear, testable Acceptance Criteria (AC) for each core functional requirement.**
    Elaborate reasonably based on best practices if the initial request is minimal. Avoid asking clarifying questions unless ambiguity blocks core generation.
2.  **System Design:** Brainstorm and select architectures, technology stacks. **Document the chosen architecture, tech stack rationale, key components, data models, and API design overview in a High-Level Design (HLD) document structure.** Make justified decisions internally.
3.  **Implementation Planning:** Define a logical project structure. Plan modules/components, configuration management, build processes, deployment considerations.
4.  **Dev Environment Setup:** Plan the `.devcontainer/devcontainer.json` configuration and related files for a reproducible development environment.
5.  **Testing Strategy & Case Definition:** Define unit, integration, E2E testing approaches. **Document the overall Test Strategy.** Outline specific test cases (unit/integration) based on ACs. Plan test files, fixtures, mocks.
6.  **Documentation Strategy:** Plan for all required documentation files, ensuring consistency and cross-referencing where appropriate (e.g., README links to Design doc).


**Interaction Model & Constraints (Gemini API - `start_chat`/`send_message`):**

1.  **Initial Setup:** You are receiving this prompt via `start_chat`.
2.  **Project Request:** The user will send the specific project requirements in the *next* `send_message` call.
3.  **Iterative Generation:**
    *   You will generate a logical chunk of *complete files* in each response.
    *   If more files remain after the current chunk, stop outputting `FILENAME:` blocks.
    *   The user will *only* send `next` to signal continuation.
4.  **Output Format (Strict & Critical for Parsing):**
    *   Your *entire* response MUST consist *only* of one or more file content blocks:
        ```
        FILENAME: {PROJECT_DIR_PLACEHOLDER}/path/to/your/file.ext
        ```lang
        # Content of the file goes here (e.g., code, config, docs, test cases)
        ```
        --- END OF FILE ---

        ```
    *   **`FILENAME: {PROJECT_DIR_PLACEHOLDER}/...`**: Mandatory, literal `{PROJECT_DIR_PLACEHOLDER}` required. Correct relative path/filename.
    *   **Code Fence ```lang ... ```**: Mandatory. Ensure appropriate `lang` identifiers (e.g., `markdown` for docs, `yaml` for OpenAPI, `python` for code, `gherkin` for BDD, `json`, `dockerfile`, `text`, `gitignore`, etc.).
    *   **`--- END OF FILE ---`**: Mandatory separator after ```.
    *   **NO EXTRA TEXT:** Absolutely NO conversational text outside the defined file blocks.
5.  **Chunking & Continuation:**
    *   Generate a reasonable number of complete files per response.
    *   End response after the final `--- END OF FILE ---` for the chunk if more files remain. Wait for `next`.
    *   *(Optional Internal Aid):* `--- CONTINUATION HINT --- [Note]` allowed *only* internally after the last `--- END OF FILE ---` if the project isn't finished.
6.  **Completion Signal:** After the *absolute final* file block, output on a new line:
    `{COMPLETION_SIGNAL_GENERATE}`
7.  **Post-Completion:** If user sends `next` after completion, respond *only* with: `No more files to generate.`

**Generation Requirements:**

*   **Completeness:** Ensure *all* necessary files based on your internal SDLC simulation, encompassing code, tests, configuration, and **comprehensive documentation**:
    *   **Requirements Documentation:**
        *   `docs/REQUIREMENTS.md` or `SPECIFICATION.md`: Detailing Functional Requirements, Non-Functional Requirements (NFRs), User Roles, Assumptions/Constraints derived from the user request.
        *   `docs/ACCEPTANCE_CRITERIA.md` or integrated within `REQUIREMENTS.md` or as `.feature` files: Listing the specific, testable ACs for functionalities.
    *   **Design Documentation:**
        *   `docs/DESIGN.md` or `ARCHITECTURE.md`: Outlining the High-Level Design (HLD), chosen architecture, technology stack rationale, key components/modules, data model overview, and API design principles.
        *   API Specification (if applicable, e.g., `openapi.yaml`, `api_spec.json`).
    *   **Source Code:** Backend, Frontend logic, well-commented.
    *   **Testing Artifacts:**
        *   `docs/TESTING_STRATEGY.md` or section in README: Summarizing the overall testing approach (unit, integration, E2E, tools).
        *   Unit Test Code: Functional tests implementing specific cases based on ACs.
        *   Integration/E2E Test Stubs/Examples: Placeholder files or basic examples (code or descriptive like Gherkin `.feature` files).
    *   **Configuration Files:** `requirements.txt`, `package.json`, `.env.example`, framework configs, linters, formatters.
    *   **Build & Deployment Scripts:** `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` placeholder.
    *   **Development Environment Setup:** `.devcontainer/devcontainer.json`, supporting files.
    *   **User & Project Documentation:**
        *   `README.md`: Comprehensive overview, project purpose, setup instructions, usage guide, how to run tests, environment variable explanation, **links to key documents** (Requirements, Design, Testing Strategy).
        *   `CHANGELOG.md`: Standard changelog format (can start with an initial version entry).
        *   `CONTRIBUTING.md`: Guidelines for contributors (can be a standard template).
        *   `CODE_OF_CONDUCT.md`: Project's code of conduct (can be a standard template).
        *   `LICENSE`: Appropriate license file (e.g., MIT, Apache 2.0 - default to MIT if unspecified).
    *   **Database Files:** Schema definitions, migrations (if applicable).
    *   **Dependency Management Files:** (e.g. `Pipfile`, `go.mod`, `pom.xml`).
    *   `.gitignore`: Standard ignore file for the chosen tech stack.
    *   **(If applicable) Basic UX/UI Files:** HTML, CSS, JS starters.
*   **Placeholders:** Use clear placeholders, document them in `README.md` and `.env.example`. Use `{PROJECT_DIR_PLACEHOLDER}` only in the `FILENAME:` line.
*   **Best Practices:** Clean, commented, idiomatic code. Error handling/logging. Testability. Modern libraries. Logical structure. Consistent documentation.

**Error Correction (For Future Interactions):**

*   If user provides error feedback (`--- ERROR REPORT --- ... --- END ERROR REPORT ---`), regenerate *only* the necessary files in the strict format to fix the issue. No conversational text.

**Ready:** Acknowledge you understand these comprehensive instructions, including the requirement for detailed requirements/design specifications and other standard project documents, and are ready to receive the project requirements by responding *only* with: `{READY_SIGNAL}`. Do not add any other text.
"""

# --- Prompt for Project Modification (RAG, Single-Shot) ---
# This prompt guides the LLM to act as a change management expert, using
# provided code context (via RAG) to generate all necessary file modifications,
# additions, or deletions in a single response.
SETUP_PROMPT_MODIFY_RAG: str = f"""
**Your Role:** You are an expert Autonomous Software System specializing in change management. You analyze change requests for existing software projects, leverage provided context, simulate impact analysis and planning, and generate all required file modifications, additions, deletions, including associated tests and documentation updates.

**Core Objective:** Based on the user's change request and provided **relevant code context snippets** (via RAG), simulate a standard change management process internally (impact analysis, implementation, test planning, doc update planning) and generate the *complete set* of file modifications, additions, and deletions required to implement the change comprehensively across the codebase, tests, and documentation in a **single response**.

**Input Provided by User (via RAG system):**

1.  **Change Request:** The user's description of the required modifications.
2.  **Relevant Code Context:** Snippets of existing code/documentation from the project files, automatically retrieved based on relevance. These snippets are the *primary source* for understanding the current state *and style* of relevant sections.

    ```
    --- Relevant Code Context Snippet ---
    File Path: {PROJECT_DIR_PLACEHOLDER}/path/to/relevant/file.ext

    [Code or documentation snippet content...]
    --- End Snippet ---

    [... possibly more snippets ...]
    ```

**Internal Simulation Mandate (For Your Guidance Only - Do NOT include this in output):**

1.  **Impact Analysis:** Analyze the Change Request against the Context Snippets. Identify affected components, side effects, changes to logic/interfaces.
2.  **Implementation Planning:** Determine specific code modifications, new files/functions, or deletions. Plan integration respecting existing patterns.
3.  **Testing Planning:** Identify existing tests needing updates. Plan *new* tests for changed/added functionality.
4.  **Documentation Planning:** Identify docs needing updates (e.g., `CHANGELOG.md` is almost always required, `README.md` if usage changes, `docs/REQUIREMENTS.md`/`docs/DESIGN.md` for significant spec/architecture changes). Plan content.

**Your Task:**

1.  **Analyze:** Perform the internal simulation steps.
2.  **Plan Changes:** Determine the final set of actions: modify existing files, add new files (including tests/docs), delete obsolete files.
3.  **Generate Output:** Generate the *resulting files/actions* in a single response, strictly adhering to the specified output format below. Ensure generated code/tests/docs are consistent with the style/patterns in the Context Snippets.

**Output Format (Strict & Critical):**

*   Your *entire* response MUST consist *only* of action blocks. No conversational text, explanations, summaries.
*   **For File Creation or Full Modification:** Output the *complete, final content*.
    ```
    FILENAME: {PROJECT_DIR_PLACEHOLDER}/path/to/your/file.ext
    ```lang
    # Complete content of the file (new or modified)
    # Ensure this includes all necessary code, tests, or documentation updates.
    ```
    --- END OF FILE ---
    ```
*   **For File Deletion:**
    ```
    ACTION: DELETE {PROJECT_DIR_PLACEHOLDER}/path/to/delete/file.ext
    ```
*   **Mandatory:** Include the literal `{PROJECT_DIR_PLACEHOLDER}` in all paths.
*   **Scope:** Generate blocks for *all* files needing changes/additions/deletions based on your analysis (source, tests, docs like `CHANGELOG.md`, config). Use context snippets for style/location guidance, but generate required changes even for unretrieved files if logical (e.g., `CHANGELOG.md`).
*   Use appropriate `lang` identifiers (e.g., `python`, `javascript`, `markdown`, `yaml`, `json`, `text`, `diff` - prefer full files over diffs).

**Completion Signal:** After the *absolute final* action block for the *entire request*, output the exact line on a new line:
`{COMPLETION_SIGNAL_MODIFY}`

**Consistency Goal:** Prioritize maintaining the style, libraries, patterns, and architectural choices observed in the provided **Code Context Snippets**.

**Ready:** Acknowledge understanding by responding *only* with: `{READY_SIGNAL}`. Do not add any other text.
"""

# --- Regex for Parsing LLM Output ---
# This regex captures FILENAME blocks (for writing/updating files) and
# ACTION: DELETE blocks from the LLM's structured output.
# It expects the PROJECT_DIR_PLACEHOLDER in the paths.
ACTION_BLOCK_REGEX = re.compile(
    r"(?:(?:^|\n)\s*```\s*\n)?"  # Optional leading ``` whitespace
    r"^(FILENAME:\s*("  # Start G1 (FILENAME block), Start G2 (Filename Path)
    + re.escape(PROJECT_DIR_PLACEHOLDER)  # Match the literal placeholder
    + r"/[^\n]+)"  # Match the rest of the relative path, End G2
    r"\s*```(?:[a-zA-Z0-9_.-]+)?\n"  # Code block start ```lang or ```\n (optional lang)
    r"(.*?)\n"  # Group 3: Capture the file content (non-greedy)
    r"(?:(?:^|\n)\s*```\s*\n)?"  # Optional trailing ``` whitespace
    r"--- END OF FILE ---\s*$)"  # Match the mandatory separator at end of line/block
    # --- OR ---
    r"|"
    # Group 4: Match ACTION: DELETE block
    r"^(ACTION:\s*DELETE\s*("  # Start G4 (DELETE block), Start G5 (Delete Path)
    + re.escape(PROJECT_DIR_PLACEHOLDER)  # Match the literal placeholder
    + r"/[^\n]+)"  # Match the rest of the relative path, End G5
    r"\s*$)",  # Must be the whole line
    re.DOTALL
    | re.IGNORECASE
    | re.MULTILINE,  # Flags: DOTALL for G3, MULTILINE for ^/$ anchors, IGNORECASE for keywords
)


# --- Helper Functions ---


def configure_api() -> Optional[str]:
    """
    Configures the Google Generative AI API key.

    Priority:
    1. GOOGLE_API_KEY environment variable.
    2. Value loaded from a .env file (if python-dotenv is installed).
    3. Prompts user via getpass if not found.

    Returns:
        The API key if configured successfully, None otherwise.
    """
    api_key = os.getenv(API_KEY_ENV_VAR)

    if api_key:
        print(f"Using Google API Key from environment variable {API_KEY_ENV_VAR}.")
    else:
        print(f"Environment variable {API_KEY_ENV_VAR} not found or empty.")
        print("Attempting to prompt for API key...")
        try:
            api_key = getpass("Please enter your Google API Key: ")
            if not api_key:
                print("API Key is required. Exiting.")
                return None
            else:
                # Optionally set it as an environment variable for the current process
                # This helps LangChain components find it automatically.
                os.environ[API_KEY_ENV_VAR] = api_key
                print("API Key received and set for this session.")
        except Exception as e:
            print(f"Error reading API key: {e}")
            return None

    # Configure the base genai library (LangChain components often use the env var directly)
    try:
        genai.configure(api_key=api_key)
        print("Google Generative AI SDK configured successfully.")
        return api_key
    except Exception as e:
        print(f"Error configuring Google Generative AI SDK: {e}")
        print("Please ensure your API key is valid and has the necessary permissions.")
        return None


def get_project_path_from_user(prompt_message: str) -> Optional[Path]:
    """
    Prompts the user for a project directory path and validates it.

    Args:
        prompt_message: The message to display when prompting the user.

    Returns:
        A resolved Path object to the directory if valid, None otherwise.
    """
    while True:
        path_str = input(f"{prompt_message} ").strip()
        if not path_str:
            print("Path cannot be empty.")
            continue
        try:
            path = Path(path_str).resolve()
            # For generation, the directory might not exist yet, which is okay.
            # For modification/indexing, it MUST exist. We'll check existence later.
            return path
        except Exception as e:
            print(f"Error resolving path '{path_str}': {e}")
            # Ask if user wants to try again
            retry = input("Invalid path. Try again? (y/N): ").strip().lower()
            if retry != "y":
                return None


# --- RAG Specific Functions ---


def get_embeddings_model() -> Optional[GoogleGenerativeAIEmbeddings]:
    """Initializes and returns the Google Generative AI Embeddings model."""
    try:
        # Assumes API key is set in environment (e.g., via configure_api)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        print(f"Initialized Embeddings model: {EMBEDDING_MODEL_NAME}")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings model '{EMBEDDING_MODEL_NAME}': {e}")
        print("Ensure the API key is valid and has permissions for this model.")
        return None


def get_llm() -> Optional[ChatGoogleGenerativeAI]:
    """Initializes and returns the Google Generative AI Chat model."""
    try:
        # Assumes API key is set in environment
        llm = ChatGoogleGenerativeAI(
            model=LLM_NAME,
            temperature=0.2,  # Lower temp for more deterministic code generation/modification
            convert_system_message_to_human=True,  # Can sometimes help compatibility
            # safety_settings=... # Consider adding safety settings if needed
        )
        print(f"Initialized LLM: {LLM_NAME}")
        return llm
    except Exception as e:
        print(f"Error initializing LLM '{LLM_NAME}': {e}")
        print("Ensure the API key is valid and has permissions for this model.")
        return None


def create_or_load_vector_store(
    project_path: Path, force_reindex: bool = False
) -> Optional[Chroma]:
    """
    Creates or loads a Chroma vector store for the given project path.
    Indexes code, documentation, and configuration files found within the project.

    Args:
        project_path: The root directory of the project to index.
        force_reindex: If True, delete any existing index and re-create it.

    Returns:
        A Chroma vector store instance if successful, None otherwise.
    """
    if not project_path.is_dir():
        print(f"Error: Project directory not found: {project_path}")
        return None

    persist_directory = project_path / VECTOR_STORE_SUBDIR
    embeddings = get_embeddings_model()
    if not embeddings:
        return None  # Error already printed

    vector_store: Optional[Chroma] = None

    # --- Attempt to Load Existing Store ---
    if persist_directory.exists() and not force_reindex:
        print(f"Attempting to load existing vector store from: {persist_directory}")
        try:
            vector_store = Chroma(
                persist_directory=str(persist_directory),
                embedding_function=embeddings,
            )
            # Perform a quick check to ensure it's functional
            vector_store.similarity_search("test query", k=1)
            print("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            print(
                f"Warning: Error loading existing vector store (will attempt re-index): {e}"
            )
            print(
                "This might happen if the underlying data format changed or became corrupted."
            )
            vector_store = None  # Ensure we proceed to re-indexing

    # --- Create New Store or Re-index ---
    if persist_directory.exists():
        if force_reindex or vector_store is None:  # Need to re-index
            print(
                f"Removing existing vector store for re-indexing: {persist_directory}"
            )
            try:
                shutil.rmtree(persist_directory)
            except OSError as e:
                print(f"Error removing existing directory '{persist_directory}': {e}")
                print("Proceeding might cause issues. Please check permissions.")
                # Decide if we should exit or just warn and continue
                # return None # Safer option
        else:
            # Should not happen if loading worked, but safeguard
            print("Error: Store exists but wasn't loaded and re-index not forced.")
            return None

    print(f"\nCreating new vector store for project: {project_path.name}")
    print(f"Indexing files in: {project_path}")
    print("This might take a while for large projects...")

    # --- Define File Patterns ---
    # Broad patterns to include common code, config, and doc files. Adjust as needed.
    include_globs: List[str] = [
        "**/*.py",
        "**/*.js",
        "**/*.ts",
        "**/*.jsx",
        "**/*.tsx",
        "**/*.java",
        "**/*.go",
        "**/*.rb",
        "**/*.php",
        "**/*.rs",
        "**/*.cs",
        "**/*.swift",
        "**/*.html",
        "**/*.css",
        "**/*.scss",
        "**/*.less",
        "**/*.md",
        "**/*.rst",
        "**/*.txt",
        "**/*.json",
        "**/*.yaml",
        "**/*.yml",
        "**/*.toml",
        "**/Dockerfile",
        "**/*.sh",
        "**/*.bash",
        "**/*.zsh",
        "**/*.sql",
        "**/*.sql.schema",
        "**/.env.example",
        "**/requirements*.txt",
        "**/pyproject.toml",
        "**/package.json",
        "**/package-lock.json",
        "**/yarn.lock",
        "**/pom.xml",
        "**/build.gradle",
        "**/settings.gradle",
        "**/go.mod",
        "**/go.sum",
        "**/Gemfile",
        "**/Gemfile.lock",
        "**/composer.json",
        "**/composer.lock",
        "**/Cargo.toml",
        "**/Cargo.lock",
        "**/*.csproj",
        "**/*.sln",
        "**/Podfile",
        "**/Podfile.lock",
        "**/.gitignore",
        "**/.dockerignore",
        # Add more specific config files if needed: e.g., "**/nginx.conf", "**/settings.py"
    ]
    # Patterns to exclude binary files, build artifacts, caches, env files, etc.
    exclude_patterns: List[str] = [
        f"**/{VECTOR_STORE_SUBDIR}/**",  # Exclude the vector store itself!
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/.git/**",
        "**/dist/**",
        "**/build/**",
        "**/target/**",
        "**/.next/**",
        "**/.nuxt/**",
        "**/.venv/**",
        "**/venv/**",
        "**/env/**",  # Virtual environments
        "**/.vscode/**",
        "**/.idea/**",
        "**/*.code-workspace",  # IDE folders
        "**/*.pyc",
        "**/*.pyo",
        "**/*.class",
        "**/*.jar",
        "**/*.war",
        "**/*.ear",
        "**/*.log",
        "**/*.lock",
        "**/*.swp",
        "**/*.swo",
        "**/*.so",
        "**/*.dll",
        "**/*.exe",
        "**/*.bin",  # Compiled objects/binaries
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.gif",
        "**/*.bmp",
        "**/*.ico",
        "**/*.svg",
        "**/*.pdf",
        "**/*.doc",
        "**/*.docx",
        "**/*.xls",
        "**/*.xlsx",
        "**/*.ppt",
        "**/*.pptx",
        "**/*.zip",
        "**/*.tar",
        "**/*.gz",
        "**/*.rar",
        "**/*.7z",
        "**/.env",  # Exclude actual .env files containing secrets
        "**/.DS_Store",  # macOS specific
        # Optionally exclude tests if they pollute context (can be very project-specific)
        # "**/test/**", "**/tests/**", "**/*_test.py", "**/*.spec.js", "**/*_test.go",
    ]

    # --- Load Documents ---
    all_docs: List[Document] = []
    print("Searching for files matching include patterns...")
    # Use a set to avoid processing the same absolute path twice if matched by multiple globs
    processed_paths = set()

    for glob_pattern in include_globs:
        try:
            loader = DirectoryLoader(
                str(project_path),
                glob=glob_pattern,
                recursive=True,
                show_progress=False,  # Set to True for verbose loading progress
                use_multithreading=True,
                loader_cls=TextLoader,  # Simple text loader
                loader_kwargs={
                    "encoding": "utf-8",
                    "autodetect_encoding": True,
                },  # Handle encoding issues
                silent_errors=True,  # Ignore files TextLoader can't read (e.g., binary)
            )
            loaded_docs = loader.load()

            # Filter out excluded patterns and duplicates
            for doc in loaded_docs:
                try:
                    abs_path = Path(doc.metadata["source"]).resolve()
                    if abs_path in processed_paths:
                        continue  # Skip already processed file

                    # Check relative path against exclude patterns
                    relative_path = abs_path.relative_to(project_path)
                    if not any(
                        relative_path.match(pattern) for pattern in exclude_patterns
                    ):
                        all_docs.append(doc)
                        processed_paths.add(abs_path)
                        # print(f"  + Included: {relative_path}") # Verbose
                    # else: # Verbose
                    #     print(f"  - Excluded: {relative_path}")

                except ValueError:
                    # Handle cases where path isn't relative (shouldn't usually happen here)
                    print(
                        f"  Warning: Could not make path relative for {doc.metadata.get('source', 'Unknown')}. Skipping."
                    )
                except Exception as path_e:
                    print(
                        f"  Warning: Error processing path metadata for {doc.metadata.get('source', 'Unknown')}: {path_e}. Skipping."
                    )

            # print(f"  Found {len(filtered_docs)} new documents for glob '{glob_pattern}'") # Optional Verbose
        except Exception as e:
            print(f"  Warning: Error loading documents for glob '{glob_pattern}': {e}")

    if not all_docs:
        print("\nError: No documents found to index.")
        print("Check the project path, ensure it contains supported file types,")
        print("and review the include/exclude patterns in the script.")
        return None

    print(f"\nTotal unique, non-excluded documents loaded: {len(all_docs)}.")
    print("Splitting documents into chunks...")

    # --- Split Documents ---
    # Using RecursiveCharacterTextSplitter - good general purpose splitter for code.
    # Consider language-specific splitters (e.g., Language.PYTHON) for more semantic chunking
    # if performance/retrieval quality needs improvement.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,  # Use default separators suitable for code
        # separators=["\n\n", "\n", " ", ""] # Default separators
    )

    try:
        splits = text_splitter.split_documents(all_docs)
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return None

    # --- Add Relative Path Metadata (Crucial for Prompts) ---
    # Add a 'file_path' metadata field using the placeholder format for easy use in prompts.
    for split in splits:
        try:
            relative_path = (
                Path(split.metadata["source"]).relative_to(project_path).as_posix()
            )
            # Store path with placeholder for LLM context, and original relative path for reference
            split.metadata["file_path_placeholder"] = (
                f"{PROJECT_DIR_PLACEHOLDER}/{relative_path}"
            )
            split.metadata["relative_path"] = relative_path
        except (ValueError, KeyError, Exception) as meta_e:
            # Fallback if path isn't relative or 'source' key missing
            source = split.metadata.get("source", "Unknown Source")
            print(
                f"  Warning: Error setting relative path metadata for chunk from '{source}': {meta_e}"
            )
            split.metadata["file_path_placeholder"] = (
                f"{PROJECT_DIR_PLACEHOLDER}/{source}"  # Less ideal fallback
            )
            split.metadata["relative_path"] = source

    if not splits:
        print("Error: Documents loaded but failed to split into chunks.")
        return None

    print(f"Split into {len(splits)} chunks.")
    print(f"Creating vector store at: {persist_directory}")

    # --- Create Chroma Vector Store ---
    try:
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(persist_directory),  # Persist to disk
            # collection_metadata={"hnsw:space": "cosine"} # Optional: Specify distance metric
        )
        print("Vector store created and persisted successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
        # Attempt cleanup of potentially corrupted store directory
        try:
            if persist_directory.exists():
                shutil.rmtree(persist_directory)
                print(
                    f"Cleaned up potentially corrupted store directory: {persist_directory}"
                )
        except OSError as cleanup_e:
            print(
                f"Error cleaning up store directory after creation failure: {cleanup_e}"
            )
        return None


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Formats retrieved documents into a string suitable for the LLM prompt context.

    Args:
        docs: A list of LangChain Document objects retrieved from the vector store.

    Returns:
        A formatted string containing the content and file paths of the documents.
    """
    formatted_docs: List[str] = []
    for i, doc in enumerate(docs):
        # Use the placeholder path we added during indexing
        placeholder_path = doc.metadata.get("file_path_placeholder", "Unknown Path")
        content = doc.page_content
        formatted_docs.append(
            f"--- Relevant Code Context Snippet ---\n"
            f"File Path: {placeholder_path}\n\n"
            f"{content}\n"
            f"--- End Snippet ---"
        )
    # Provide a message if retrieval somehow failed or returned empty
    return (
        "\n\n".join(formatted_docs)
        if formatted_docs
        else "No relevant code context snippets were retrieved. Please attempt the modification based on the request alone, using standard best practices."
    )


# --- File Parsing and Application Function ---


def parse_and_apply_changes(
    response_text: str, project_root_dir: Path
) -> Tuple[int, int]:
    """
    Parses LLM response text for FILENAME and ACTION: DELETE blocks and applies
    the changes to the file system relative to the specified project root directory.

    Args:
        response_text: The raw text output from the LLM.
        project_root_dir: The absolute path to the root of the target project directory.

    Returns:
        A tuple containing: (number_of_files_written, number_of_files_deleted).
    """
    files_written_count = 0
    files_deleted_count = 0
    # Track paths processed in this specific response to avoid duplicate operations
    # (e.g., LLM outputting the same file twice)
    applied_relative_paths_in_turn = set()

    # Ensure the project root exists for file operations (might be needed if creating new)
    try:
        project_root_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"Error: Could not create or access project directory {project_root_dir}: {e}"
        )
        print("Please check permissions and path.")
        return 0, 0  # Cannot proceed

    project_root_abs = project_root_dir.resolve()
    response_text = response_text.strip()

    # Iterate through all matches of the regex in the response
    for match in ACTION_BLOCK_REGEX.finditer(response_text):
        action_type: Optional[str] = None
        placeholder_path: Optional[str] = None
        file_content: Optional[str] = None
        relative_path_str: Optional[str] = (
            None  # Actual relative path after removing placeholder
        )

        # --- Extract Action Details ---
        if match.group(1):  # Matched FILENAME block (create/update)
            action_type = "WRITE"
            placeholder_path = match.group(2).strip()
            file_content = match.group(
                3
            )  # Content includes leading/trailing whitespace within block
            # Clean up content slightly - remove potential surrounding ``` markdown
            file_content = file_content.strip()
            if file_content.startswith("```"):
                file_content = file_content[file_content.find("\n") + 1 :]
            if file_content.endswith("```"):
                file_content = file_content[:-3].rstrip()

        elif match.group(4):  # Matched ACTION: DELETE block
            action_type = "DELETE"
            placeholder_path = match.group(5).strip()
            file_content = None  # No content for delete actions
        else:
            # This should not happen with the current regex structure but is a safeguard
            print(
                "Warning: Regex match found but did not correspond to WRITE or DELETE action. Skipping."
            )
            continue

        # --- Validate Placeholder Path and Derive Relative Path ---
        if not placeholder_path or not placeholder_path.startswith(
            PROJECT_DIR_PLACEHOLDER + "/"
        ):
            print(
                f"Warning: Skipping action due to invalid or missing placeholder path format: '{placeholder_path or 'None'}'"
            )
            continue

        # Extract the path part *after* "<PROJECT_ROOT>/"
        relative_path_str = placeholder_path[
            len(PROJECT_DIR_PLACEHOLDER) + 1 :
        ].strip()  # +1 for the '/'
        if not relative_path_str:
            print(
                f"Warning: Skipping action due to empty relative path derived from: {placeholder_path}"
            )
            continue

        # Normalize path separators for consistency
        relative_path_str = relative_path_str.replace("\\", "/")

        # Prevent processing the same relative path multiple times within this *single* LLM response
        if relative_path_str in applied_relative_paths_in_turn:
            print(
                f"Warning: Skipping duplicate action for path '{relative_path_str}' in this response."
            )
            continue

        # --- Resolve Target Path and Perform Security Check ---
        try:
            # Resolve the final target path by joining the project root and the relative path
            target_file_path = (project_root_abs / relative_path_str).resolve()

            # **CRITICAL Security Check:** Ensure the resolved path is strictly *within* the project root.
            # This prevents directory traversal attacks (e.g., "../../etc/passwd").
            if (
                project_root_abs != target_file_path
                and project_root_abs not in target_file_path.parents
            ):
                print(
                    f"Security Warning: Skipping {action_type} action for path outside designated project root!"
                )
                print(f"  Attempted path: {relative_path_str}")
                print(f"  Resolved path: {target_file_path}")
                print(f"  Project root:  {project_root_abs}")
                continue

        except (
            ValueError
        ) as e:  # Handles invalid characters in path components on some OS
            print(
                f"Warning: Skipping {action_type} action due to invalid path characters in '{relative_path_str}': {e}"
            )
            continue
        except Exception as e:  # Catch other potential path resolution errors
            print(
                f"Warning: Skipping {action_type} action due to unexpected path error for '{relative_path_str}': {e}"
            )
            continue

        # --- Apply the Action (Write or Delete) ---
        applied_relative_paths_in_turn.add(
            relative_path_str
        )  # Mark as processed for this turn
        try:
            if action_type == "WRITE":
                if file_content is None:
                    print(
                        f"Warning: Skipping WRITE action for '{relative_path_str}' because content is missing."
                    )
                    continue  # Should not happen if regex matched group 3

                # Create parent directories if they don't exist
                target_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write the file content (overwrite if exists)
                with open(target_file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
                print(f"  Applied Change (Wrote/Updated): {relative_path_str}")
                files_written_count += 1

            elif action_type == "DELETE":
                if target_file_path.is_file():
                    target_file_path.unlink()
                    print(f"  Applied Change (Deleted File): {relative_path_str}")
                    files_deleted_count += 1
                    # Optionally: Clean up empty parent directories after deletion
                    # try:
                    #     target_file_path.parent.rmdir() # Only removes if empty
                    #     print(f"  - Cleaned up empty parent directory: {target_file_path.parent.relative_to(project_root_abs)}")
                    # except OSError:
                    #     pass # Directory not empty or other error, ignore
                elif target_file_path.is_dir():
                    # Safety: By default, do NOT delete directories specified by ACTION: DELETE.
                    # This requires explicit user confirmation or a different command if needed.
                    print(
                        f"Warning: Skipping DELETE action because path is a directory: {relative_path_str}"
                    )
                    print(
                        f"  (To delete directories, manual intervention or code modification is required.)"
                    )
                    # To enable recursive directory deletion (USE WITH EXTREME CAUTION):
                    # confirmed = input(f"Confirm deletion of directory '{relative_path_str}' and ALL its contents? (y/N): ").strip().lower() == 'y'
                    # if confirmed:
                    #     shutil.rmtree(target_file_path)
                    #     print(f"  Applied Change (Deleted Directory): {relative_path_str}")
                    #     files_deleted_count += 1 # Or count separately
                    # else:
                    #     print(f"  Skipped directory deletion: {relative_path_str}")
                elif not target_file_path.exists():
                    # File/directory not found, maybe already deleted or path typo in LLM output
                    print(
                        f"Info: File/directory to delete not found (perhaps already deleted?): {relative_path_str}"
                    )
                else:
                    # Exists but isn't a file or directory (e.g., symlink, device) - skip
                    print(
                        f"Warning: Path exists but is not a regular file or directory. Skipping deletion: {relative_path_str}"
                    )

        except OSError as e:
            print(
                f"Error applying {action_type} to {target_file_path.relative_to(project_root_abs)}: {e}"
            )
        except Exception as e:
            print(
                f"An unexpected error occurred applying {action_type} to {target_file_path.relative_to(project_root_abs)}: {e}"
            )

    # --- End of Loop ---
    if not applied_relative_paths_in_turn and response_text:
        # Check if the response contained text but nothing matched the regex
        if not ACTION_BLOCK_REGEX.search(response_text):
            print(
                "\nWarning: The LLM response contained text, but no valid FILENAME or ACTION: DELETE blocks were found."
            )
            print(
                "         Please review the raw LLM output below to see what was generated."
            )
            print("--- LLM Raw Output (Unparsed) ---")
            print(response_text)
            print("--- End Raw Output ---")

    return files_written_count, files_deleted_count


# --- Main Workflow Functions ---


def generate_new_project(project_path: Path) -> None:
    """
    Drives the process for generating a new project using the iterative, non-RAG approach.

    Args:
        project_path: The target directory path provided by the user where the project
                      files will be generated.
    """
    print("\n--- Generating New Project ---")
    print(f"Target project directory: {project_path}")

    # API key should be configured already by main execution flow
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        # Should have been caught earlier, but double-check
        print("Error: API Key not configured. Cannot proceed.")
        return

    # --- Get Project Description ---
    print(
        "\nPlease describe the project you want to generate (e.g., 'a simple Python Flask web server with a single route /hello that returns JSON'):"
    )
    user_request = input("> ").strip()
    if not user_request:
        print("Error: Project description cannot be empty.")
        return

    # Create the target directory if it doesn't exist
    try:
        project_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured project directory exists: {project_path}")
    except OSError as e:
        print(f"Error creating project directory {project_path}: {e}")
        return

    # --- Initialize Model and Chat ---
    try:
        print(f"\nInitializing model: {LLM_NAME}")
        # Use the base google.generativeai library for iterative chat
        # Note: Safety settings might differ from LangChain wrapper, adjust if needed
        model = genai.GenerativeModel(LLM_NAME)
        chat = model.start_chat(history=[])  # Start with empty history
        print("Model initialized, starting generation chat...")

        # --- Send Setup Prompt and Wait for READY ---
        print(f"\nSending setup prompt to {LLM_NAME}...")
        setup_response = chat.send_message(SETUP_PROMPT_GENERATE)

        if setup_response.text.strip() != READY_SIGNAL:
            print(
                f"\nWarning: Expected '{READY_SIGNAL}' after setup prompt, but received:"
            )
            print(f"'{setup_response.text[:200].strip()}...'")
            print(
                "Attempting to proceed, but the model might not follow instructions correctly."
            )
            # Optionally, could exit here if READY is strictly required
            # return
        else:
            print(f"Model acknowledged setup ({READY_SIGNAL}).")

        # --- Send Project Request ---
        print("\nSending project request...")
        response = chat.send_message(user_request)
        full_response_text = response.text.strip()

        # --- Iterative Generation Loop ---
        is_complete = False
        success = False  # Track if completion signal was properly received
        turn_counter = 1  # Start after initial request
        total_files_written = 0
        total_files_deleted = 0  # Should ideally be 0 for generation

        while not is_complete:
            print("-" * 20)
            print(f"Processing Generation Turn {turn_counter}")

            response_to_parse = full_response_text

            # Check for explicit completion signal FIRST
            if COMPLETION_SIGNAL_GENERATE in response_to_parse:
                print(f"\nCompletion signal '{COMPLETION_SIGNAL_GENERATE}' detected.")
                # Parse only text *before* the signal
                response_to_parse = response_to_parse.split(COMPLETION_SIGNAL_GENERATE)[
                    0
                ].rstrip()
                is_complete = True
                success = True  # Mark as successful completion

            # Parse and apply changes found in the (potentially truncated) response
            written, deleted = parse_and_apply_changes(response_to_parse, project_path)
            total_files_written += written
            total_files_deleted += deleted

            if deleted > 0:
                print(
                    f"Warning: {deleted} file(s) were unexpectedly deleted during generation in this turn."
                )

            # If not complete yet, send "next" to get more files
            if not is_complete:
                # Check if the last response actually contained any actions
                if (
                    written == 0 and deleted == 0 and turn_counter > 1
                ):  # Only warn if it wasn't the first response
                    # If the response was non-empty but unparsable, parse_and_apply prints a warning.
                    # If it was empty, assume completion.
                    if not response_to_parse:
                        print(
                            "Warning: Received empty response after 'next'. Assuming completion (signal might be missing)."
                        )
                        is_complete = True
                        success = False  # Signal was missing
                        break
                    else:
                        # parse_and_apply_changes already warned if response wasn't empty but unparsable
                        print(
                            "Assuming completion as previous response yielded no file actions (signal might be missing)."
                        )
                        is_complete = True
                        success = False  # Signal was missing
                        break

                print("\nSending 'next' to request more files...")
                turn_counter += 1
                try:
                    response = chat.send_message("next")
                    full_response_text = response.text.strip()
                except Exception as e:
                    print(
                        f"Error communicating with Gemini on turn {turn_counter}: {e}"
                    )
                    # Check for feedback which might indicate blocked content etc.
                    if hasattr(response, "prompt_feedback"):
                        print(f"Prompt Feedback: {response.prompt_feedback}")
                    is_complete = True  # Stop on error
                    success = False
                    break

        # --- End of Generation Loop ---
        print("-" * 20)
        if success:
            print("\nProject generation finished successfully.")
        else:
            print(
                "\nProject generation finished, but the completion signal may be missing or errors occurred."
            )
            print("Please review the generated files carefully.")
        print(f"Total files written: {total_files_written}")
        if total_files_deleted > 0:
            print(
                f"Warning: {total_files_deleted} files were unexpectedly deleted during the generation process."
            )
        print(f"Project generated in: {project_path}")

    except Exception as e:
        print(f"\nAn unexpected error occurred during project generation: {e}")
        # Consider adding more detailed traceback logging if needed for debugging
        import traceback

        traceback.print_exc()


def modify_existing_project_rag(project_path: Path) -> None:
    """
    Drives the process for modifying an existing project using the RAG approach.

    Args:
        project_path: The path to the root directory of the existing project.
    """
    print("\n--- Modifying Existing Project ---")
    print(f"Target project directory: {project_path}")

    if not project_path.is_dir():
        print(f"Error: Project directory not found: {project_path}")
        print("Cannot modify a non-existent project.")
        return

    # API key should be configured already
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        print("Error: API Key not configured. Cannot proceed.")
        return

    # --- Ensure Vector Store is Ready ---
    force_reindex_input = (
        input(f"Re-index '{project_path.name}' before modifying? (y/N): ")
        .strip()
        .lower()
    )
    force_reindex = force_reindex_input == "y"

    print("\nEnsuring vector store is up-to-date...")
    vector_store = create_or_load_vector_store(project_path, force_reindex)
    if not vector_store:
        print(
            "\nFailed to create or load vector store. Cannot proceed with modification."
        )
        return

    # --- Get User's Change Request ---
    print(
        "\nPlease describe the changes you want to make to the project (be specific):"
    )
    print(
        "Example: 'Add a new endpoint /status that returns {'status': 'ok'}. Update the README.'"
    )
    user_request = input("> ").strip()
    if not user_request:
        print("Error: Change description cannot be empty.")
        return

    # --- Setup RAG Chain Components ---
    print("\nSetting up RAG chain...")
    llm = get_llm()
    if not llm:
        return  # Error already printed

    # Set up the retriever from the loaded/created vector store
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",  # Default, others include "mmr"
            search_kwargs={"k": RETRIEVER_K},  # Retrieve top K chunks
        )
        print(
            f"Retriever configured to fetch top {RETRIEVER_K} relevant document chunks."
        )
    except Exception as e:
        print(f"Error creating retriever from vector store: {e}")
        return

    # --- Define the RAG Prompt Template ---
    # This combines the setup instructions with placeholders for the actual
    # user question (change request) and the retrieved context.
    # Note: The SETUP_PROMPT_MODIFY_RAG already contains the {READY_SIGNAL} instruction
    # for the LLM, but we don't need to explicitly check for it here as LangChain handles
    # the single invocation differently than the iterative chat.
    rag_prompt_template_text: str = f"""
{SETUP_PROMPT_MODIFY_RAG}

---

**Now, fulfill the following request based on the provided context:**

**Change Request:**
{{question}}

**Relevant Code Context Retrieved:**
{{context}}

**Generated Changes (FILENAME/ACTION blocks only - End with '{COMPLETION_SIGNAL_MODIFY}'):**
"""
    try:
        rag_prompt = PromptTemplate(
            template=rag_prompt_template_text,
            input_variables=["question", "context"],
        )
    except Exception as e:
        print(f"Error creating prompt template: {e}")
        return

    # --- Define the RAG Chain using LangChain Expression Language (LCEL) ---
    # This defines the flow:
    # 1. Retrieve context based on the input question (RunnableParallel allows passing question through).
    # 2. Format the retrieved documents.
    # 3. Populate the prompt template with the question and formatted context.
    # 4. Send the populated prompt to the LLM.
    # 5. Parse the LLM's output as a string.
    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs_for_prompt,
                "question": RunnablePassthrough(),
            }
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain configured.")

    # --- Invoke RAG Chain and Process Output ---
    print("\nInvoking RAG chain to generate changes (this may take a moment)...")
    try:
        # The 'invoke' method takes the initial input (the user request string)
        # and triggers the entire chain execution.
        llm_response_text = rag_chain.invoke(user_request)

        # --- Debug: Print Raw Output ---
        # Useful for troubleshooting if parsing fails or results are unexpected
        # print("\n--- RAG Chain Raw Output ---")
        # print(llm_response_text)
        # print("--- End Raw Output ---")

        print("\nParsing and applying generated changes...")
        response_to_parse = llm_response_text.strip()

        # Check for completion signal and remove it before parsing
        final_response_text_to_parse = response_to_parse
        completion_received = False
        # Use rstrip for cleaner check in case of trailing whitespace after signal
        if response_to_parse.rstrip().endswith(COMPLETION_SIGNAL_MODIFY):
            # Remove signal and any trailing whitespace/newlines before it
            final_response_text_to_parse = response_to_parse.rstrip()[
                : -len(COMPLETION_SIGNAL_MODIFY)
            ].rstrip()
            completion_received = True
            print(f"Completion signal '{COMPLETION_SIGNAL_MODIFY}' found.")
        else:
            print(
                f"\nWarning: Completion signal '{COMPLETION_SIGNAL_MODIFY}' was NOT found at the end of the LLM response."
            )
            print(
                "         The model may not have finished generating all intended changes."
            )
            print("         Proceeding with parsing the output received so far.")
            # Proceed to parse whatever was received

        # Parse and apply the file operations specified in the final response text
        written, deleted = parse_and_apply_changes(
            final_response_text_to_parse, project_path
        )

        # --- End of Modification ---
        print("-" * 20)
        if completion_received:
            print("\nProject modification process finished.")
            if written == 0 and deleted == 0:
                print(
                    "Note: The LLM indicated completion, but no file changes were parsed."
                )
                print(
                    "      This might mean the request required no file changes, or the output format was incorrect."
                )
                print(
                    "      Please review the raw LLM output (enable debug printing in the script if needed)."
                )
        else:
            print(
                "\nProject modification process finished, but the completion signal was missing."
            )
            print(
                "Please review the changes applied and the raw LLM output (if enabled) carefully."
            )
        print(f"Total files written/updated: {written}")
        print(f"Total files deleted: {deleted}")
        print(f"Project located at: {project_path}")

    except Exception as e:
        print(f"\nAn error occurred during RAG chain execution or processing: {e}")
        # Consider adding more specific error handling based on potential LangChain/API errors
        import traceback

        traceback.print_exc()  # Print full traceback for debugging


# --- Main Execution ---


def main() -> None:
    """Main function to parse arguments and run the selected mode."""

    # --- Basic CLI Argument Parsing (Optional) ---
    # You could extend this to take mode and project path via CLI args
    # Example: parser.add_argument("mode", choices=['generate', 'modify', 'index'], help="Operation mode.")
    #          parser.add_argument("path", help="Target project directory path.")
    parser = argparse.ArgumentParser(
        description="Autonomous Project Generator & Modifier using Gemini and RAG.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python project_generator.py  (Interactive mode)
  # Future CLI examples if arguments are added:
  # python project_generator.py generate ./my_new_flask_app
  # python project_generator.py modify ./existing_project
  # python project_generator.py index ./existing_project
""",
    )
    # Add arguments here if needed in the future
    args = parser.parse_args()

    print("\nWelcome to the Gemini Project Manager (RAG Enabled)")
    print("===================================================")
    print(f"Using LLM: {LLM_NAME}")
    print(f"Using Embeddings: {EMBEDDING_MODEL_NAME}")
    print("---------------------------------------------------")

    # --- Configure API Key ---
    if not configure_api():
        sys.exit(1)  # Exit if API key configuration fails

    # --- Interactive Mode Selection ---
    mode: Optional[str] = None
    project_path: Optional[Path] = None

    # If CLI arguments were implemented, you would check args here first.
    # Since they aren't fully implemented, we go straight to interactive mode.

    while True:  # Loop until a valid mode and path are selected or user quits
        print("\nChoose an action:")
        print("  1. Generate a new project")
        print("  2. Modify an existing project (RAG Single-Shot)")
        print("  3. Re-index an existing project's Vector Store")
        print("  Q. Quit")
        choice = input("> ").strip().lower()

        if choice == "1":
            mode = "generate"
            project_path = get_project_path_from_user(
                "Enter the path for the NEW project directory:"
            )
            if project_path:
                # Confirm potentially overwriting if dir exists (though generation aims at new)
                if project_path.exists():
                    confirm = (
                        input(
                            f"Directory '{project_path}' already exists. Continue and potentially overwrite files? (y/N): "
                        )
                        .strip()
                        .lower()
                    )
                    if confirm != "y":
                        project_path = None  # Reset path to re-prompt
                        continue  # Go back to main menu
                break  # Valid path, proceed
            else:
                continue  # Path prompt failed, back to main menu

        elif choice == "2":
            mode = "modify"
            project_path = get_project_path_from_user(
                "Enter the path to the EXISTING project directory:"
            )
            if project_path and project_path.is_dir():
                break  # Valid existing directory path, proceed
            elif project_path:
                print(
                    f"Error: Directory not found or is not a directory: {project_path}"
                )
                project_path = None  # Reset path
                continue  # Back to main menu
            else:
                continue  # Path prompt failed, back to main menu

        elif choice == "3":
            mode = "index"
            project_path = get_project_path_from_user(
                "Enter the path to the EXISTING project directory to index:"
            )
            if project_path and project_path.is_dir():
                break  # Valid existing directory path, proceed
            elif project_path:
                print(
                    f"Error: Directory not found or is not a directory: {project_path}"
                )
                project_path = None  # Reset path
                continue  # Back to main menu
            else:
                continue  # Path prompt failed, back to main menu

        elif choice in ["q", "quit"]:
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or Q.")

    # --- Execute Selected Mode ---
    if mode and project_path:
        if mode == "generate":
            generate_new_project(project_path)
        elif mode == "modify":
            modify_existing_project_rag(project_path)
        elif mode == "index":
            print(f"\n--- Indexing Project Only ---")
            print(f"Project directory: {project_path}")
            # Force re-indexing when chosen explicitly from the menu
            create_or_load_vector_store(project_path, force_reindex=True)
            print("\nIndexing complete.")
    else:
        # Should not happen with the loop logic, but safeguard
        print("Exiting due to missing mode or project path.")

    print("\nOperation finished. Exiting.")


if __name__ == "__main__":
    main()
