from typing import Literal, Dict, Any, Union
from pydantic import BaseModel
from jinja2 import Template
import re
import fitz
import os
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from .utils import ImageData
from .constants import SUPPORTED_MODELS
import logging

logger = logging.getLogger(__name__)


class ImageDescription(BaseModel):
    """Model Schema for image description."""

    text_detected: Literal["Yes", "No"]
    tables_detected: Literal["Yes", "No"]
    images_detected: Literal["Yes", "No"]
    latex_equations_detected: Literal["Yes", "No"]
    extracted_text: str
    confidence_score_text: float


class UnsupportedModelError(BaseException):
    """Custom exception for unsupported model names"""
    pass


class LLMError(BaseException):
    """Custom exception for Vision LLM errors"""
    pass


class LLM:
    # Load prompts at class level
    try:
        from importlib.resources import files

        _image_analysis_prompt = Template(
            files("vision_parse").joinpath("image_analysis.j2").read_text()
        )
        _md_prompt_template = Template(
            files("vision_parse").joinpath("markdown_prompt.j2").read_text()
        )
    except Exception as e:
        raise FileNotFoundError(f"Failed to load prompt files: {str(e)}")

    def __init__(
        self,
        model_name: str,
        api_key: Union[str, None],
        temperature: float,
        top_p: float,
        openai_config: Union[Dict, None],
        image_mode: Literal["url", "base64", None],
        custom_prompt: Union[str, None],
        detailed_extraction: bool,
        enable_concurrency: bool,
        device: Literal["cuda", "mps", None],
        num_workers: int,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.openai_config = openai_config or {}
        self.temperature = temperature
        self.top_p = top_p
        self.image_mode = image_mode
        self.custom_prompt = custom_prompt
        self.detailed_extraction = detailed_extraction
        self.enable_concurrency = enable_concurrency
        self.device = device
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.provider = self._get_provider_name(model_name)
        self._init_llm()

    def _get_provider_name(self, model_name: str) -> str:
        """Get the provider name for a given model name."""
        if self.provider != "openai":
            raise UnsupportedModelError(f"Only OpenAI models are supported.")
        try:
            return SUPPORTED_MODELS[model_name]
        except KeyError:
            supported_models = ", ".join(SUPPORTED_MODELS.keys())
            raise UnsupportedModelError(
                f"Model '{model_name}' is not supported. Supported models are: {supported_models}"
            )

    def _init_llm(self) -> None:
        """Initialize the LLM client."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI is not installed. Please install it using pip install 'vision-parse[openai]'."
            )

        try:
            self.aclient = openai.AsyncClient(
                api_key=self.api_key,
                timeout=self.openai_config.get("OPENAI_TIMEOUT", 240.0),
            )
        except Exception as e:
            raise LLMError(f"Unable to initialize OpenAI client: {str(e)}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _get_response(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Process base64-encoded image through OpenAI vision models."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_encoded}"
                            },
                        },
                    ],
                }
            ]

            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature if not structured else 0.0,
                top_p=self.top_p if not structured else 0.4,
                **self.kwargs,
            )

            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```",
                r"\1",
                response.choices[0].message.content,
                flags=re.DOTALL,
            )
        except Exception as e:
            raise LLMError(f"OpenAI Model processing failed: {str(e)}")

    async def generate_markdown(
        self, base64_encoded: str, pix: fitz.Pixmap, page_number: int
    ) -> Any:
        """
        Generate markdown formatted text from a base64-encoded image
        using the OpenAI model.
        """
        extracted_images = []
        if self.detailed_extraction:
            try:
                response = await self._get_response(
                    base64_encoded,
                    self._image_analysis_prompt.render(),
                    structured=True,
                )
                json_response = ImageDescription.model_validate_json(response)

                if json_response.text_detected.strip() == "No":
                    return ""

                if (
                    json_response.images_detected.strip() == "Yes"
                    and self.image_mode is not None
                ):
                    extracted_images = ImageData.extract_images(
                        pix, self.image_mode, page_number
                    )

                prompt = self._md_prompt_template.render(
                    extracted_text=json_response.extracted_text,
                    tables_detected=json_response.tables_detected,
                    latex_equations_detected=json_response.latex_equations_detected,
                    confidence_score_text=float(json_response.confidence_score_text),
                    custom_prompt=self.custom_prompt,
                )

            except Exception:
                logger.warning(
                    "Detailed extraction failed. Falling back to simple extraction."
                )
                self.detailed_extraction = False

        if not self.detailed_extraction:
            prompt = self._md_prompt_template.render(
                extracted_text="",
                tables_detected="Yes",
                latex_equations_detected="No",
                confidence_score_text=0.0,
                custom_prompt=self.custom_prompt,
            )

        markdown_content = await self._get_response(
            base64_encoded, prompt, structured=False
        )

        if extracted_images:
            if self.image_mode == "url":
                for image_data in extracted_images:
                    markdown_content += (
                        f"\n\n![{image_data.image_url}]({image_data.image_url})"
                    )
            elif self.image_mode == "base64":
                for image_data in extracted_images:
                    markdown_content += (
                        f"\n\n![{image_data.image_url}]({image_data.base64_encoded})"
                    )

        return markdown_content
