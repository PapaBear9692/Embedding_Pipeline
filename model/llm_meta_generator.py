from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from app_config import GOOGLE_API_KEY

class LLMMetaGenerator:
    """Generates structured metadata for documents using Gemini."""

    def __init__(self):
        self.model_name = "gemini-2.5-flash-lite"
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
            )
            print(f"[LLMMetaGenerator] Loaded model: {self.model_name}")
        except Exception as e:
            print(f"[LLMMetaGenerator] Error loading Gemini model: {e}")
            raise

        # Build the prompt chain
        self._setup_chain()

    def _setup_chain(self):
        prompt_template = """
        You are a embedding metadata extraction assistant. Analyze the given text and return a compact JSON with:
        - topic (main subject)
        - entities (list of key names, brands, editions, etc.)
        - summary (1 sentence summary of the content)
        
        Example output:
        {{
          "topic": "Paracetamol Medication",
          "entities": ["PainRelief 500mg", "Dosage", "Ingredients"],
          "summary": "Describes usage and side effects of PainRelief 500mg tablets."
        }}
        
        Text:
        {chunk_text}
        """

        prompt = PromptTemplate(
            input_variables=["chunk_text"],
            template=prompt_template
        )

        self.chain = prompt | self.llm | StrOutputParser()

    def generate_metadata(self, chunk_text: str) -> dict:
        try:
            response = self.chain.invoke({"chunk_text": chunk_text}).strip()

            # Handle code blocks or malformed JSON
            if response.startswith("```"):
                response = response.split("```")[1].replace("json", "").strip()

            metadata = json.loads(response)
            return metadata

        except json.JSONDecodeError:
            print("[LLMMetaGenerator] JSON parse failed. Raw response:", response)
            # fallback minimal metadata
            return {"topic": "Unknown", "entities": [], "summary": "Failed to parse metadata"}
        except Exception as e:
            print(f"[LLMMetaGenerator] Error generating metadata: {e}")
            return {"topic": "Error", "entities": [], "summary": str(e)}
