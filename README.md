# Insurance Agent

This application is an educational tool designed to demonstrate the steps involved in building an AI-powered insurance product comparison assistant. It leverages Azure AI services to ingest, process, extract information from, and analyze insurance product documents (PDFs).

## Features

*   **Document Ingestion**: Upload or link PDF insurance policy documents.
*   **PDF to Markdown Conversion**: Uses [Azure AI Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) to convert PDFs into structured Markdown.
*   **Question & Category Configuration**: Manually define or use AI to suggest relevant questions and categories for comparison.
*   **AI-Powered Data Extraction**: Employs [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service) models to extract answers to your configured questions from the Markdown content.
*   **AI-Assisted Correction**: Provides an AI-driven review of extracted answers, suggesting corrections based on the source documents.
*   **Advanced Data Analysis**: Utilizes the [Azure AI SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/sdk-generative-overview) (specifically its agent capabilities, often involving Code Interpreter) to perform complex analyses, generate visualizations (plots, tables), and provide textual explanations. This requires an [Azure AI Studio Project](https://learn.microsoft.com/en-us/azure/ai-studio/what-is-ai-studio).
*   **Data Export**: Export extracted comparison data to Excel.
*   **Configurable Settings**: Manage Azure service credentials through a UI or environment variables.

## Setup

1.  **Prerequisites**:
    *   Python 3.9+
    *   `pip` (or `pip3`) for package management.

2.  **Clone Repository**:
    ```bash
    git clone https://github.com/aymenfurter/insurance-agent.git
    cd insurance-agent
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables & Configuration**:
    *   Create a `.env` file in the root directory of the project (e.g., `insurance-comparison-agent/.env`).
    *   Populate it with your Azure service credentials. The application will load these at startup.
    *   **Important**: Settings configured via the UI's "⚙️ Settings" tab are saved in `data/settings.json` and will **override** environment variables if present and non-empty.

    **Lean `.env` file example**:
    ```env
    # Azure OpenAI Service Configuration
    AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
    AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"

    AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT="YOUR_REASONING_MODEL_DEPLOYMENT"
    AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT="YOUR_NONREASONING_MODEL_DEPLOYMENT"

    # Azure Document Intelligence Configuration
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="YOUR_DOCUMENT_INTELLIGENCE_ENDPOINT"
    AZURE_DOCUMENT_INTELLIGENCE_KEY="YOUR_DOCUMENT_INTELLIGENCE_KEY"

    # Default insurance products for testing/demo
    DEFAULT_PRODUCT_1_NAME="YOUR_DEFAULT_PRODUCT_1_NAME"
    DEFAULT_PRODUCT_1_URLS="YOUR_DEFAULT_PRODUCT_1_URLS"

    DEFAULT_PRODUCT_2_NAME="YOUR_DEFAULT_PRODUCT_2_NAME"
    DEFAULT_PRODUCT_2_URLS="YOUR_DEFAULT_PRODUCT_2_URLS"

    DEFAULT_PRODUCT_3_NAME="YOUR_DEFAULT_PRODUCT_3_NAME"
    DEFAULT_PRODUCT_3_URLS="YOUR_DEFAULT_PRODUCT_3_URLS"

    PROJECT_CONNECTION_STRING="YOUR_PROJECT_CONNECTION_STRING"
    MODEL_DEPLOYMENT_NAME="YOUR_MODEL_DEPLOYMENT_NAME"

    # Default sample categories and questions for the UI
    DEFAULT_SAMPLE_CATEGORIES="YOUR_DEFAULT_SAMPLE_CATEGORIES"
    DEFAULT_SAMPLE_QUESTIONS="YOUR_DEFAULT_SAMPLE_QUESTIONS"
    ```

    *   **Azure Services Required**:

        | Service                                                                 | Purpose                                                                                                                                                              | Key `.env` Variables / UI Settings                                                                                                                               |
        | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        | **[Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)**                | Powers core LLM tasks: category/question suggestions, answer extraction, self-correction, and summarization.                                                      | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT`, `AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT` |
        | **[Azure AI Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)**         | Converts PDF documents into Markdown format by extracting text, layout, and tables.                                                                                  | `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`, `AZURE_DOCUMENT_INTELLIGENCE_KEY`.                                                                                       |
        | **[Azure AI SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/sdk-generative-overview) / [Azure AI Studio Project](https://learn.microsoft.com/en-us/azure/ai-studio/what-is-ai-studio)** | Enables advanced data analysis in Step 4 using Code Interpreter. This requires an Azure AI Project.                                                                | `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP_NAME`, `AZURE_AI_PROJECT_NAME` (or `PROJECT_CONNECTION_STRING`), `AZURE_OPENAI_AGENT_MODEL_DEPLOYMENT`.             |

## Running the Application

1.  Ensure your `.env` file is correctly configured or that settings are present in `data/settings.json`.
2.  Navigate to the project's root directory.
3.  Run the application:
    ```bash
    python app.py
    ```
4.  Open your web browser and go to the URL provided in the console (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`).

## Application Workflow

The application is structured into several tabs, representing a step-by-step process:

1.  **STEP 1: Document Ingestion & Preparation**:
    *   Add insurance products by providing a name and one or more PDF URLs.
    *   Process these products to convert the PDFs into AI-readable Markdown format using Azure Document Intelligence.
    *   Review and edit the generated Markdown pages side-by-side with the original PDF page.

2.  **STEP 2: Configure Questions & Categories**:
    *   Define categories relevant to the insurance products (e.g., "Fire Damage", "Liability Coverage").
    *   Create specific questions for each category that you want to find answers for in the documents.
    *   Alternatively, use the AI-powered suggestion feature to generate categories and questions based on the content of the ingested documents.

3.  **STEP 3: Data Extraction & Correction**:
    *   Initiate AI-driven data extraction. The system will attempt to answer your configured questions for each product using the Markdown content and Azure OpenAI.
    *   Review the extracted answers.
    *   Utilize the "AI Self-Correction Review" feature, where another AI agent reviews the initial extractions against the source document and suggests corrections.
    *   Apply the suggested corrections to refine the dataset.

4.  **STEP 4: Analysis & Export**:
    *   Select from predefined analysis templates or provide a custom natural language prompt for analysis.
    *   The Azure AI Agent Service (with Code Interpreter) will perform the analysis, generating:
        *   Visualizations (plots).
        *   Data tables (HTML format).
        *   Textual explanations and insights in Markdown.
    *   View the results of different analyses.
    *   Export all extracted and corrected data for all products into a single Excel file for offline use or further analysis.

5.  **⚙️ Settings**:
    *   Configure and save your Azure service credentials (OpenAI Endpoint, API Key, Document Intelligence Endpoint, etc.).
    *   These settings are saved locally and will override environment variables if set.

## Notes

*   **Educational Tool**: This application is primarily for educational purposes to demonstrate a possible pipeline. It is not a production-ready system.
*   **Cost**: Using Azure services (OpenAI, Document Intelligence, AI Agent Service) will incur costs based on your usage. Monitor your Azure subscription.
*   **Error Handling**: Basic error handling is implemented. For production, more robust error management and logging would be necessary.
*   **Security**: If exposing the Gradio app publicly (e.g., `share=True`), be mindful of security implications, especially regarding API key handling if not using managed identities or similar secure practices. The current setup is intended for local or controlled environment use.
