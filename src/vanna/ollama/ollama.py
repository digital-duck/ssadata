import json
import re

from httpx import Timeout

from ..base import VannaBase
from ..exceptions import DependencyError

class Ollama(VannaBase):
    def __init__(self, config=None):

        try:
            ollama = __import__("ollama")
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install ollama"
            )

        if not config or 'model' not in config.keys():
            raise ValueError("config must contain at least Ollama model")
    
        self.host = config.get("ollama_host", "http://localhost:11434")
        self.model = config.get("model", "llama3.1")  # use Llama3.1 as default
        if ":" not in self.model:
            self.model += ":latest"

        self.ollama_timeout = config.get("ollama_timeout", 240.0)
        self.ollama_client = ollama.Client(self.host, timeout=Timeout(self.ollama_timeout))
        self.keep_alive = config.get('keep_alive', None)
        self.ollama_options = {
            'gpu' : config.get("gpu", True),          # Enable GPU
            'num_ctx': config.get("num_ctx", 2048),   # Context window size
            'top_p': config.get("top_p", 0.9),        # Nucleus sampling
            'top_k': config.get("top_k", 40),         # Top tokens to sample
            'seed': config.get("seed", 42),           # RNG seed
            'stop': config.get("stop", ['STOP']),     # Stop sequences
            'temperature' : config.get("temperature", 0.2),       # Creativity (0-1)
            'repeat_penalty': config.get("repeat_penalty", 1.1),  # Repetition penalty
            'num_predict': config.get("num_predict", 500),       # Max tokens to generate
        }
        self.num_ctx = self.ollama_options.get('num_ctx', 2048)

        @staticmethod
        def __pull_model_if_ne(ollama_client, model):
            model_response = ollama_client.list()
            model_lists = [model_element['model'] for model_element in
                          model_response.get('models', [])]
            if model not in model_lists:
                ollama_client.pull(model)

        # self.__pull_model_if_ne(self.ollama_client, self.model)

    # @staticmethod
    # def list_models():
    #     """
    #     List all available models in the Ollama instance.
    #     Returns a list of model names.
    #     """
    #     try:
    #         ollama_ = __import__("ollama")
    #     except ImportError:
    #         raise DependencyError(
    #             "You need to install required dependencies to execute this method, run command:"
    #             " \npip install ollama"
    #         )
    
    #     client = ollama_.Client("http://localhost:11434")
    #     response = client.list()
    #     return [model['model'] for model in response.get('models', [])]

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        print_prompt = kwargs.get("print_prompt",False)
        print_response = kwargs.get("print_response",False)
        self.log(
            f"Ollama parameters:\n"
            f"model={self.model},\n"
            f"options={self.ollama_options},\n"
            f"keep_alive={self.keep_alive}")
        if print_prompt:
            self.log(f"Prompt Content:\n{json.dumps(prompt)}")
      
        response_dict = self.ollama_client.chat(model=self.model,
                                                messages=prompt,
                                                stream=False,
                                                options=self.ollama_options,
                                                keep_alive=self.keep_alive)

        if print_response:
            self.log(f"Ollama Response:\n{str(response_dict)}")

        return response_dict['message']['content']
  

