import os
from ..base import VannaBase

DEFAULT_MODEL = "gemini-1.5-pro"

class GoogleGeminiChat(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)

        # default temperature - can be overrided using config
        self.config = config
        self.temperature = 0.2

        if "temperature" in config:
            self.temperature = config["temperature"]

        model_name = DEFAULT_MODEL
        if "model" in config:
            model_name = config["model"]
        if "model_name" in config:
            model_name = config["model_name"]

        self.google_api_key = None

        api_key = config.get("api_key", os.getenv("GOOGLE_API_KEY"))

        if api_key:
            """
            If Google api_key is provided through config
            or set as an environment variable, assign it.
            """
            self.google_api_key = api_key
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.chat_model = genai.GenerativeModel(model_name)
        else:
            # Authenticate using VertexAI
            from vertexai.generative_models import GenerativeModel
            self.chat_model = GenerativeModel(model_name)

    def system_message(self, message: str) -> any:
        return message

    def user_message(self, message: str) -> any:
        return message

    def assistant_message(self, message: str) -> any:
        return message

    def submit_prompt(self, prompt, **kwargs) -> str:
        response = self.chat_model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
            },
        )
        return response.text
